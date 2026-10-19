@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> order: array<u32>;

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

const quad_corners = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1));
const culled_position = vec4f(2, 2, 0, 1);
const eps2d = 0.3;

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

@vertex fn vs_main(input: VSInput) -> VSOutput {
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
    let blur = eps2d * pixel_to_clip * pixel_to_clip;
    let cov = B * transpose(B) + mat2x2f(blur.x, 0, 0, blur.y);

    let det_cov = determinant(cov);
    if (det_cov <= 0.0) { return output; }

    let a = cov[0][0];
    let b = cov[0][1];
    let c = cov[1][1];

    output.offset = quad_corners[input.vertex_index] * 3.0 * sqrt(vec2f(a, c));
    output.position = vec4f(clip_pos.xy + output.offset, clip_pos.zw);
    output.conic = vec3f(c, -b, a) / det_cov;
    output.color = gaussian.color;

    return output;
}

@fragment fn fs_main(input: VSOutput) -> @location(0) vec4f {
    let p = input.offset;
    let q = input.conic.x * p.x * p.x + 2.0 * input.conic.y * p.x * p.y + input.conic.z * p.y * p.y;
    if (q > 9.0) { discard; }
    let alpha = min(0.99, input.color.a * exp(-0.5 * q));
    if (alpha < 1.0 / 255.0) { discard; }
    return vec4f(input.color.rgb * alpha, alpha);
}
