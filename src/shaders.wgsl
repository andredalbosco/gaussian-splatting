struct Gaussian {
    position: vec4f,
    scale: vec4f,
    rotation: vec4f,
    color: vec4f,
}

struct Camera {
    view_projection: mat4x4f,
    viewport: vec4f,
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

@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> order: array<u32>;

const quad_corners = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1));

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn projected_axis(gaussian: Gaussian, clip: vec4f, local_axis: vec3f) -> vec2f {
    let axis = camera.view_projection * vec4f(quat_rotate(gaussian.rotation, local_axis), 0);
    return (axis.xy * clip.w - clip.xy * axis.w) / (clip.w * clip.w) * camera.viewport.xy * 0.5;
}

@vertex fn vs_main(input: VSInput) -> VSOutput {
    let gaussian = gaussians[order[input.instance_index]];
    let clip = camera.view_projection * vec4f(gaussian.position.xyz, 1);
    var output: VSOutput;
    output.position = vec4f(2, 2, 0, 1);
    if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) { return output; }

    // Columns of B = J * R_camera * R_object * S; covariance = B * transpose(B).
    let ax = projected_axis(gaussian, clip, vec3f(gaussian.scale.x, 0, 0));
    let ay = projected_axis(gaussian, clip, vec3f(0, gaussian.scale.y, 0));
    let az = projected_axis(gaussian, clip, vec3f(0, 0, gaussian.scale.z));
    let x = vec3f(ax.x, ay.x, az.x);
    let y = vec3f(ax.y, ay.y, az.y);
    let a = dot(x, x) + 0.3;
    let b = dot(x, y);
    let c = dot(y, y) + 0.3;
    let determinant = a * c - b * b;
    if (determinant <= 0.0) { return output; }

    output.offset = quad_corners[input.vertex_index] * 3.0 * sqrt(vec2f(a, c));
    output.position = vec4f(clip.xy / clip.w + output.offset * 2.0 / camera.viewport.xy, clip.z / clip.w, 1);
    output.conic = vec3f(c, -b, a) / determinant;
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
