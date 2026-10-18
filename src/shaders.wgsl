struct Gaussian {
    position: vec4f,
    scale: vec4f,
    rotation: vec4f,
    color: vec4f,
}

struct Camera {
    view_projection: mat4x4f,
    position: vec4f,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: f32,
}

const unit_sphere = Sphere(vec3f(0), 1.0);

struct Ellipsoid {
    center: vec3f,
    scale: vec3f,
    rotation: vec4f,
}

struct VSInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(2) world_pos: vec3f,
    @location(0) @interpolate(flat) color: vec3f,
    @location(1) @interpolate(flat) ellipsoid_center: vec3f,
    @location(3) @interpolate(flat) ellipsoid_scale: vec3f,
    @location(4) @interpolate(flat) ellipsoid_rotation: vec4f,
}

struct FSOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> camera: Camera;

const box_corners = array(
    vec3f(-1, -1, -1), vec3f(1, -1, -1), vec3f(1, 1, -1), vec3f(-1, 1, -1),
    vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(-1, 1, 1),
);

const box_indices = array(
    0u, 2u, 1u, 0u, 3u, 2u,
    4u, 5u, 6u, 4u, 6u, 7u,
    0u, 4u, 7u, 0u, 7u, 3u,
    1u, 2u, 6u, 1u, 6u, 5u,
    0u, 1u, 5u, 0u, 5u, 4u,
    3u, 7u, 6u, 3u, 6u, 2u,
);

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn ray_intersect_sphere(ray: Ray, sphere: Sphere) -> f32 {
    let offset = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let b = dot(offset, ray.direction) / a;
    let closest = offset - b * ray.direction;
    let delta = (sphere.radius * sphere.radius - dot(closest, closest)) / a;
    if (delta < 0.0) { return -1.0; }

    let sqrt_delta = sqrt(delta);
    var t = -b - sqrt_delta;
    if (t <= 0.0) { t = -b + sqrt_delta; }
    if (t <= 0.0) { return -1.0; }
    return t;
}

fn ray_intersect_ellipsoid(ray: Ray, ellipsoid: Ellipsoid) -> f32 {
    let inverse_rotation = vec4f(-ellipsoid.rotation.xyz, ellipsoid.rotation.w);
    let local_ray = Ray(
        quat_rotate(inverse_rotation, ray.origin - ellipsoid.center) / ellipsoid.scale,
        quat_rotate(inverse_rotation, ray.direction) / ellipsoid.scale,
    );
    return ray_intersect_sphere(local_ray, unit_sphere);
}

@vertex fn vs_main(input: VSInput) -> VSOutput {
    let gaussian = gaussians[input.instance_index];
    let local = box_corners[box_indices[input.vertex_index]] * gaussian.scale.xyz;
    let world_pos = gaussian.position.xyz + quat_rotate(gaussian.rotation, local);

    var output: VSOutput;
    output.position = camera.view_projection * vec4f(world_pos, 1.0);
    output.color = gaussian.color.rgb;
    output.ellipsoid_center = gaussian.position.xyz;
    output.ellipsoid_scale = gaussian.scale.xyz;
    output.ellipsoid_rotation = gaussian.rotation;
    output.world_pos = world_pos;

    return output;
}

@fragment fn fs_main(input: VSOutput) -> FSOutput {
    let ray = Ray(camera.position.xyz, normalize(input.world_pos - camera.position.xyz));
    let ellipsoid = Ellipsoid(input.ellipsoid_center, input.ellipsoid_scale, input.ellipsoid_rotation);
    let t = ray_intersect_ellipsoid(ray, ellipsoid);
    if (t <= 0.0) { discard; }

    let hit = ray.origin + t * ray.direction;
    let clip = camera.view_projection * vec4f(hit, 1.0);
    let depth = clip.z / clip.w;
    if (clip.w <= 0.0 || depth < 0.0 || depth > 1.0) { discard; }

    var output: FSOutput;
    output.color = vec4f(input.color, 1.0);
    output.depth = depth;
    return output;
}
