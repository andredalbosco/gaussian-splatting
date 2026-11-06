override golus_billboard = false;
override object_space_rays = false;

struct Gaussian {
    position: vec4f,
    scale: vec4f,
    rotation: vec4f,
    color: vec4f,
}

struct Camera {
    view_projection: mat4x4f,
    position: vec4f,
    inverse_view_projection: mat4x4f,
    settings: vec4u,
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
    @location(2) ray_direction: vec3f,
    @location(0) @interpolate(flat) color: vec3f,
    @location(1) @interpolate(flat) ellipsoid_center: vec3f,
    @location(3) @interpolate(flat) ellipsoid_scale: vec3f,
    @location(4) @interpolate(flat) ellipsoid_rotation: vec4f,
    @location(5) @interpolate(flat) local_origin: vec3f,
    @location(6) local_direction: vec3f,
}

struct FSOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> camera: Camera;

const quad_corners = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 1));

const clip_planes = array(
    vec4f(1, 0, 0, 1), vec4f(-1, 0, 0, 1),
    vec4f(0, 1, 0, 1), vec4f(0, -1, 0, 1),
    vec4f(0, 0, 1, 0), vec4f(0, 0, -1, 1),
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

fn quad_vertex(input: VSInput) -> VSOutput {
    let gaussian = gaussians[input.instance_index];
    let clip = camera.view_projection * vec4f(gaussian.position.xyz, 1);
    let axis_x = camera.view_projection * vec4f(quat_rotate(gaussian.rotation, vec3f(gaussian.scale.x, 0, 0)), 0);
    let axis_y = camera.view_projection * vec4f(quat_rotate(gaussian.rotation, vec3f(0, gaussian.scale.y, 0)), 0);
    let axis_z = camera.view_projection * vec4f(quat_rotate(gaussian.rotation, vec3f(0, 0, gaussian.scale.z)), 0);

    var output: VSOutput;
    output.position = vec4f(2, 2, 0, 1);
    for (var i = 0u; i < 6u; i++) {
        let plane = clip_planes[i];
        let extent = length(vec3f(dot(axis_x, plane), dot(axis_y, plane), dot(axis_z, plane)));
        if (dot(clip, plane) + extent < 0.0) { return output; }
    }

    let offset = camera.position.xyz - gaussian.position.xyz;
    let radius = max(gaussian.scale.x, max(gaussian.scale.y, gaussian.scale.z));
    let distance_squared = dot(offset, offset);
    if (golus_billboard && distance_squared > radius * radius) {
        let forward = offset * inverseSqrt(distance_squared);
        let right = normalize(cross(forward, normalize(camera.inverse_view_projection[1].xyz)));
        let up = cross(right, forward);
        let half_size = radius * sqrt(distance_squared / (distance_squared - radius * radius));
        let corner = quad_corners[input.vertex_index] * 2.0 - 1.0;
        let world = gaussian.position.xyz + (right * corner.x + up * corner.y) * half_size;
        let quad_clip = camera.view_projection * vec4f(world, 1);
        output.position = vec4f(quad_clip.xy, 0, quad_clip.w);
        output.ray_direction = world - camera.position.xyz;
    } else {
        let x = vec3f(axis_x.x, axis_y.x, axis_z.x);
        let y = vec3f(axis_x.y, axis_y.y, axis_z.y);
        let w = vec3f(axis_x.w, axis_y.w, axis_z.w);
        var lower = vec2f(-1);
        var upper = vec2f(1);
        if (clip.w > length(w)) {
            let a = clip.w * clip.w - dot(w, w);
            let center = (clip.xy * clip.w - vec2f(dot(x, w), dot(y, w))) / a;
            let dx = clip.w * x - clip.x * w;
            let dy = clip.w * y - clip.y * w;
            let cx = cross(x, w);
            let cy = cross(y, w);
            let radius = sqrt(max(vec2f(dot(dx, dx) - dot(cx, cx), dot(dy, dy) - dot(cy, cy)), vec2f(0))) / a;
            lower = center - radius;
            upper = center + radius;
        }
        let ndc = mix(clamp(lower, vec2f(-1), vec2f(1)),
                      clamp(upper, vec2f(-1), vec2f(1)), quad_corners[input.vertex_index]);
        let world = camera.inverse_view_projection * vec4f(ndc, 1, 1);

        output.position = vec4f(ndc, 0, 1);
        output.ray_direction = world.xyz - camera.position.xyz * world.w;
    }
    output.color = gaussian.color.rgb;
    output.ellipsoid_center = gaussian.position.xyz;
    output.ellipsoid_scale = gaussian.scale.xyz;
    output.ellipsoid_rotation = gaussian.rotation;
    if (object_space_rays) {
        let inverse_rotation = vec4f(-gaussian.rotation.xyz, gaussian.rotation.w);
        output.local_origin = quat_rotate(inverse_rotation, offset) / gaussian.scale.xyz;
        output.local_direction = quat_rotate(inverse_rotation, output.ray_direction) / gaussian.scale.xyz;
    }

    return output;
}

@vertex fn vs_main(input: VSInput) -> VSOutput {
    return quad_vertex(input);
}

@vertex fn vs_quad_wire(input: VSInput) -> VSOutput {
    let indices = array(0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u, 1u, 2u);
    var output = quad_vertex(VSInput(indices[input.vertex_index], input.instance_index));
    output.color = vec3f(0.2, 0.9, 1.0);
    return output;
}

@vertex fn vs_box_wire(input: VSInput) -> VSOutput {
    let corners = array(
        vec3f(-1, -1, -1), vec3f(1, -1, -1), vec3f(1, 1, -1), vec3f(-1, 1, -1),
        vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(-1, 1, 1),
    );
    let indices = array(
        0u, 1u, 1u, 2u, 2u, 3u, 3u, 0u,
        4u, 5u, 5u, 6u, 6u, 7u, 7u, 4u,
        0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u,
    );
    let gaussian = gaussians[input.instance_index];
    let local = corners[indices[input.vertex_index]] * gaussian.scale.xyz;
    let world = gaussian.position.xyz + quat_rotate(gaussian.rotation, local);
    var output: VSOutput;
    output.position = camera.view_projection * vec4f(world, 1);
    output.color = vec3f(1.0, 0.65, 0.15);
    return output;
}

@fragment fn fs_wire(input: VSOutput) -> @location(0) vec4f {
    return vec4f(input.color, 1);
}

@fragment fn fs_main(input: VSOutput) -> FSOutput {
    let ellipsoid = Ellipsoid(input.ellipsoid_center, input.ellipsoid_scale, input.ellipsoid_rotation);
    var hit: vec3f;
    if (object_space_rays) {
        let t = ray_intersect_sphere(Ray(input.local_origin, input.local_direction), unit_sphere);
        if (t <= 0.0) { discard; }
        hit = camera.position.xyz + t * input.ray_direction;
    } else {
        let ray = Ray(camera.position.xyz, normalize(input.ray_direction));
        let t = ray_intersect_ellipsoid(ray, ellipsoid);
        if (t <= 0.0) { discard; }
        hit = ray.origin + t * ray.direction;
    }
    let clip = camera.view_projection * vec4f(hit, 1.0);
    let depth = clip.z / clip.w;
    if (clip.w <= 0.0 || depth < 0.0 || depth > 1.0) { discard; }

    var output: FSOutput;
    var color = input.color;
    if (camera.settings.x == 1u) {
        let inverse_rotation = vec4f(-ellipsoid.rotation.xyz, ellipsoid.rotation.w);
        let local_normal = quat_rotate(inverse_rotation, hit - ellipsoid.center) / (ellipsoid.scale * ellipsoid.scale);
        let normal = normalize(quat_rotate(ellipsoid.rotation, local_normal));
        color *= 0.25 + 0.75 * max(dot(normal, normalize(vec3f(-0.4, 0.7, 1))), 0.0);
    }
    output.color = vec4f(color, 1.0);
    output.depth = depth;
    return output;
}
