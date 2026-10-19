@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> order: array<u32>;

@group(1) @binding(0) var<uniform> radix_params: RadixParams;
@group(1) @binding(1) var<storage, read> radix_input: array<vec2u>;
@group(1) @binding(2) var<storage, read_write> radix_output: array<vec2u>;
@group(1) @binding(3) var<storage, read_write> radix_histogram: array<u32>;
@group(1) @binding(4) var<storage, read_write> radix_block_offsets: array<u32>;
@group(1) @binding(5) var<storage, read_write> radix_local_ranks: array<u32>;
@group(1) @binding(6) var<storage, read_write> radix_order: array<u32>;

const quad_corners = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1));
const culled_position = vec4f(2, 2, 0, 1);
const cov_blur = 0.3;
const gaussian_extent = 3.0;

struct Gaussian {
    position: vec4f,
    scale: vec4f,
    rotation: vec4f,
    color: vec4f,
}

struct Camera {
    view_projection: mat4x4f,
    view: mat4x4f,
    viewport: vec2f,
    projection_scale: vec2f,
}

struct VSInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) offset: vec2f,
    @location(1) @interpolate(flat) conic: vec3f,
    @location(2) @interpolate(flat) color: vec4f,
}

struct RadixParams {
    count: u32,
    group_count: u32,
    block_count: u32,
    bit_shift: u32,
}

var<workgroup> digit_masks: array<atomic<u32>, 128>;
var<workgroup> scan_values: array<u32, 256>;
var<workgroup> digit_totals: array<u32, 16>;

fn quat_to_matrix(q: vec4f) -> mat3x3f {
    let d = 2.0 * q.xyz;
    let xx = q.x * d.x; let yy = q.y * d.y; let zz = q.z * d.z;
    let xy = q.x * d.y; let xz = q.x * d.z; let yz = q.y * d.z;
    let wx = q.w * d.x; let wy = q.w * d.y; let wz = q.w * d.z;
    return mat3x3f(
        vec3f(1 - yy - zz, xy + wz, xz - wy),
        vec3f(xy - wz, 1 - xx - zz, yz + wx),
        vec3f(xz + wy, yz - wx, 1 - xx - yy),
    );
}

@vertex
fn vs_main(input: VSInput) -> VSOutput {
    var output: VSOutput;

    let gaussian = gaussians[order[input.instance_index]];
    let clip_pos = camera.view_projection * vec4f(gaussian.position.xyz, 1);
    let view_pos = (camera.view * vec4f(gaussian.position.xyz, 1)).xyz;

    output.position = culled_position;
    if (clip_pos.w <= 0.0 || clip_pos.z < 0.0 || clip_pos.z > clip_pos.w) { return output; }

    let f = camera.projection_scale;

    let J = mat3x2f(vec2f(f.x, 0), vec2f(0, f.y), -f * view_pos.xy / view_pos.z);
    let W = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let R = quat_to_matrix(gaussian.rotation);
    let M = mat3x3f(R[0] * gaussian.scale.x, R[1] * gaussian.scale.y, R[2] * gaussian.scale.z);

    let B = J * W * M;
    let pixel_to_clip = 2.0 * clip_pos.w / camera.viewport;
    let blur = cov_blur * pixel_to_clip * pixel_to_clip;
    let cov2d = B * transpose(B) + mat2x2f(blur.x, 0, 0, blur.y);

    let cov2d_det = determinant(cov2d);
    if (cov2d_det <= 0.0) { return output; }

    let cov2d_xx = cov2d[0][0];
    let cov2d_xy = cov2d[0][1];
    let cov2d_yy = cov2d[1][1];

    output.offset = quad_corners[input.vertex_index] * gaussian_extent * sqrt(vec2f(cov2d_xx, cov2d_yy));
    output.position = vec4f(clip_pos.xy + output.offset, clip_pos.zw);
    output.conic = vec3f(cov2d_yy, -cov2d_xy, cov2d_xx) / cov2d_det;
    output.color = gaussian.color;

    return output;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4f {
    let p = input.offset;
    let q = input.conic.x * p.x * p.x + 2.0 * input.conic.y * p.x * p.y + input.conic.z * p.y * p.y;
    if (q > gaussian_extent * gaussian_extent) { discard; }
    let alpha = min(0.99, input.color.a * exp(-0.5 * q));
    if (alpha < 1.0 / 255.0) { discard; }
    return vec4f(input.color.rgb * alpha, alpha);
}

fn prefix_sum(value: u32, lane: u32) -> vec2u {
    scan_values[lane] = value;
    workgroupBarrier();
    for (var step = 1u; step < 256u; step *= 2u) {
        var add = 0u;
        if (lane >= step) {
            add = scan_values[lane - step];
        }
        workgroupBarrier();
        scan_values[lane] += add;
        workgroupBarrier();
    }
    return vec2u(scan_values[lane] - value, scan_values[255]);
}

fn depth_key(depth: f32) -> u32 {
    let bits = select(bitcast<u32>(depth), 0u, depth == 0.0);
    let mask = select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
    return ~(bits ^ mask);
}

@compute @workgroup_size(256)
fn radix_histogram_rank(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    if (lane < 128u) { atomicStore(&digit_masks[lane], 0u); }
    workgroupBarrier();
    let index = group.x * 256u + lane;
    var digit = 0u;
    let word = lane / 32u;
    let bit = lane % 32u;
    if (index < radix_params.count) {
        var key: u32;
        if (radix_params.bit_shift == 0u) {
            let direction = vec3f(camera.view[0].z, camera.view[1].z, camera.view[2].z);
            key = depth_key(dot(direction, gaussians[index].position.xyz));
            radix_output[index] = vec2u(key, index);
        } else {
            key = radix_input[index].x;
        }
        digit = (key >> radix_params.bit_shift) & 15u;
        atomicOr(&digit_masks[digit * 8u + word], 1u << bit);
    }
    workgroupBarrier();
    if (index < radix_params.count) {
        var rank = countOneBits(atomicLoad(&digit_masks[digit * 8u + word]) & ((1u << bit) - 1u));
        for (var w = 0u; w < word; w++) {
            rank += countOneBits(atomicLoad(&digit_masks[digit * 8u + w]));
        }
        radix_local_ranks[index] = rank;
    }
    if (lane < 16u) {
        var count = 0u;
        for (var w = 0u; w < 8u; w++) {
            count += countOneBits(atomicLoad(&digit_masks[lane * 8u + w]));
        }
        radix_histogram[lane * radix_params.group_count + group.x] = count;
    }
}

@compute @workgroup_size(256)
fn radix_scan_blocks(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    let index = group.x * 256u + lane;
    let digit = group.y;
    var value = 0u;
    if (index < radix_params.group_count) { value = radix_histogram[digit * radix_params.group_count + index]; }
    let result = prefix_sum(value, lane);
    if (index < radix_params.group_count) { radix_histogram[digit * radix_params.group_count + index] = result.x; }
    if (lane == 0u) { radix_block_offsets[digit * radix_params.block_count + group.x] = result.y; }
}
@compute @workgroup_size(16)
fn radix_scan_sums(@builtin(local_invocation_index) digit: u32) {
    var total = 0u;
    for (var block = 0u; block < radix_params.block_count; block++) {
        total += radix_block_offsets[digit * radix_params.block_count + block];
    }
    digit_totals[digit] = total;
    workgroupBarrier();
    var offset = 0u;
    for (var previous_digit = 0u; previous_digit < digit; previous_digit++) { offset += digit_totals[previous_digit]; }
    for (var block = 0u; block < radix_params.block_count; block++) {
        let index = digit * radix_params.block_count + block;
        let count = radix_block_offsets[index];
        radix_block_offsets[index] = offset;
        offset += count;
    }
}

@compute @workgroup_size(256)
fn radix_scatter(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= radix_params.count) { return; }
    let key_index = radix_input[index];
    let digit = (key_index.x >> radix_params.bit_shift) & 15u;
    let group = index / 256u;
    let base = radix_block_offsets[digit * radix_params.block_count + group / 256u];
    let offset = radix_histogram[digit * radix_params.group_count + group] + radix_local_ranks[index];
    if (radix_params.bit_shift == 28u) {
        radix_order[base + offset] = key_index.y;
    } else {
        radix_output[base + offset] = key_index;
    }
}
