override gaussian_falloff = false;
override fragment_rays = false;
override normalize_ray = false;

struct Ellipsoid {
    position: vec4f,
    scale: vec4f,
    rotation: vec4f,
    color: vec4f,
}
struct Camera {
    view_projection: mat4x4f,
    position: vec4f,
    inverse_view_projection: mat4x4f,
}
struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) direction: vec3f,
    @location(1) local_direction: vec3f,
    @location(2) @interpolate(flat) local_origin: vec3f,
    @location(3) offset: vec2f,
    @location(4) @interpolate(flat) conic: vec3f,
    @location(5) @interpolate(flat) center: vec3f,
    @location(6) @interpolate(flat) scale: vec3f,
    @location(7) @interpolate(flat) rotation: vec4f,
    @location(8) @interpolate(flat) color: vec4f,
}
struct Fragment {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}
@group(0) @binding(0) var<storage, read> ellipsoids: array<Ellipsoid>;
@group(0) @binding(1) var<uniform> camera: Camera;

const corners = array(
    vec3f(-1, -1, -1), vec3f(1, -1, -1), vec3f(1, 1, -1), vec3f(-1, 1, -1),
    vec3f(-1, -1, 1), vec3f(1, -1, 1), vec3f(1, 1, 1), vec3f(-1, 1, 1),
);
const faces = array(
    0u, 2u, 1u, 0u, 3u, 2u, 4u, 5u, 6u, 4u, 6u, 7u,
    0u, 4u, 7u, 0u, 7u, 3u, 1u, 2u, 6u, 1u, 6u, 5u,
    0u, 1u, 5u, 0u, 5u, 4u, 3u, 7u, 6u, 3u, 6u, 2u,
);
const edges = array(0u, 1u, 1u, 2u, 2u, 3u, 3u, 0u,
                    4u, 5u, 5u, 6u, 6u, 7u, 7u, 4u,
                    0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);
const quad = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1));
const quad_edges = array(0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u);

fn rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn world_point(ellipsoid: Ellipsoid, local: vec3f) -> vec3f {
    return ellipsoid.position.xyz + rotate(ellipsoid.rotation, local * ellipsoid.scale.xyz);
}
fn ray_vertex(ellipsoid: Ellipsoid, position: vec4f, direction: vec3f) -> Vertex {
    let inverse_rotation = vec4f(-ellipsoid.rotation.xyz, ellipsoid.rotation.w);
    var output: Vertex;
    output.position = position;
    output.direction = direction;
    output.center = ellipsoid.position.xyz;
    output.scale = ellipsoid.scale.xyz;
    output.rotation = ellipsoid.rotation;
    output.color = ellipsoid.color;
    if (!fragment_rays) {
        output.local_origin = rotate(inverse_rotation, camera.position.xyz - ellipsoid.position.xyz) / ellipsoid.scale.xyz;
        output.local_direction = rotate(inverse_rotation, direction) / ellipsoid.scale.xyz;
    }
    return output;
}
@vertex fn vs_box(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let ellipsoid = ellipsoids[instance];
    let world = world_point(ellipsoid, corners[faces[index]]);
    return ray_vertex(ellipsoid, camera.view_projection * vec4f(world, 1), world - camera.position.xyz);
}
fn quad_vertex(ellipsoid: Ellipsoid, index: u32) -> Vertex {
    let clip = camera.view_projection * vec4f(ellipsoid.position.xyz, 1);
    let ax = camera.view_projection * vec4f(rotate(ellipsoid.rotation, vec3f(ellipsoid.scale.x, 0, 0)), 0);
    let ay = camera.view_projection * vec4f(rotate(ellipsoid.rotation, vec3f(0, ellipsoid.scale.y, 0)), 0);
    let az = camera.view_projection * vec4f(rotate(ellipsoid.rotation, vec3f(0, 0, ellipsoid.scale.z)), 0);
    let x = vec3f(ax.x, ay.x, az.x);
    let y = vec3f(ax.y, ay.y, az.y);
    let w = vec3f(ax.w, ay.w, az.w);
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
    let ndc = mix(clamp(lower, vec2f(-1), vec2f(1)), clamp(upper, vec2f(-1), vec2f(1)), quad[index] * 0.5 + 0.5);
    let world = camera.inverse_view_projection * vec4f(ndc, 1, 1);
    return ray_vertex(ellipsoid, vec4f(ndc, 0, 1), world.xyz - camera.position.xyz * world.w);
}
@vertex fn vs_quad(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    return quad_vertex(ellipsoids[instance], index);
}
@vertex fn vs_box_wire(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let ellipsoid = ellipsoids[instance];
    var output: Vertex;
    output.position = camera.view_projection * vec4f(world_point(ellipsoid, corners[edges[index]]), 1);
    return output;
}
@vertex fn vs_quad_wire(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    return quad_vertex(ellipsoids[instance], quad_edges[index]);
}

@vertex fn vs_projected_box(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let ellipsoid = ellipsoids[instance];
    var lower = vec2f(1e30);
    var upper = vec2f(-1e30);
    var crosses_camera = false;
    for (var i = 0u; i < 8u; i++) {
        let clip = camera.view_projection * vec4f(world_point(ellipsoid, corners[i]), 1);
        crosses_camera = crosses_camera || clip.w <= 0.0;
        lower = min(lower, clip.xy / clip.w);
        upper = max(upper, clip.xy / clip.w);
    }
    if (crosses_camera) { lower = vec2f(-1); upper = vec2f(1); }
    let ndc = mix(clamp(lower, vec2f(-1), vec2f(1)), clamp(upper, vec2f(-1), vec2f(1)), quad[index] * 0.5 + 0.5);
    let world = camera.inverse_view_projection * vec4f(ndc, 1, 1);
    return ray_vertex(ellipsoid, vec4f(ndc, 0, 1), world.xyz - camera.position.xyz * world.w);
}
@vertex fn vs_golus(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let ellipsoid = ellipsoids[instance];
    let offset = camera.position.xyz - ellipsoid.position.xyz;
    let radius = max(ellipsoid.scale.x, max(ellipsoid.scale.y, ellipsoid.scale.z));
    let distance_squared = dot(offset, offset);
    if (distance_squared <= radius * radius) { return quad_vertex(ellipsoid, index); }
    let forward = offset * inverseSqrt(distance_squared);
    let right = normalize(cross(forward, normalize(camera.inverse_view_projection[1].xyz)));
    let up = cross(right, forward);
    let half_size = radius * sqrt(distance_squared / (distance_squared - radius * radius));
    let world = ellipsoid.position.xyz + (right * quad[index].x + up * quad[index].y) * half_size;
    let clip = camera.view_projection * vec4f(world, 1);
    return ray_vertex(ellipsoid, vec4f(clip.xy, 0, clip.w), world - camera.position.xyz);
}

// Columns of J * R * S, with J differentiating the perspective divide.
fn projected_axis(ellipsoid: Ellipsoid, clip: vec4f, local_axis: vec3f) -> vec2f {
    let axis = camera.view_projection * vec4f(rotate(ellipsoid.rotation, local_axis), 0);
    return (axis.xy * clip.w - clip.xy * axis.w) / (clip.w * clip.w);
}
@vertex fn vs_splat(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let ellipsoid = ellipsoids[instance];
    let clip = camera.view_projection * vec4f(ellipsoid.position.xyz, 1);
    let ax = projected_axis(ellipsoid, clip, vec3f(ellipsoid.scale.x, 0, 0));
    let ay = projected_axis(ellipsoid, clip, vec3f(0, ellipsoid.scale.y, 0));
    let az = projected_axis(ellipsoid, clip, vec3f(0, 0, ellipsoid.scale.z));
    let x = vec3f(ax.x, ay.x, az.x);
    let y = vec3f(ax.y, ay.y, az.y);
    let a = dot(x, x);
    let b = dot(x, y);
    let c = dot(y, y);
    let support = select(1.0, 3.0, gaussian_falloff);
    var output: Vertex;
    output.offset = quad[index] * sqrt(vec2f(a, c)) * support;
    output.position = vec4f(clip.xy / clip.w + output.offset, clip.z / clip.w, 1);
    output.color = ellipsoid.color;
    output.conic = vec3f(c, -b, a) / (a * c - b * b);
    return output;
}
@fragment fn fs_surface(input: Vertex) -> Fragment {
    var o = input.local_origin;
    var d = input.local_direction;
    if (fragment_rays) {
        let inverse_rotation = vec4f(-input.rotation.xyz, input.rotation.w);
        o = rotate(inverse_rotation, camera.position.xyz - input.center) / input.scale;
        d = rotate(inverse_rotation, input.direction) / input.scale;
    }
    var inverse_length = 1.0;
    var a = dot(d, d);
    if (normalize_ray) {
        inverse_length = inverseSqrt(a);
        d *= inverse_length;
        a = 1.0;
    }
    let b = dot(o, d) / a;
    let closest = o - b * d;
    let delta = (1.0 - dot(closest, closest)) / a;
    if (delta < 0.0) { discard; }
    var t = -b - sqrt(delta);
    if (t <= 0.0) { t = -b + sqrt(delta); }
    if (t <= 0.0) { discard; }
    let hit = camera.position.xyz + (t * inverse_length) * input.direction;
    let clip = camera.view_projection * vec4f(hit, 1);
    let depth = clip.z / clip.w;
    if (clip.w <= 0.0 || depth < 0.0 || depth > 1.0) { discard; }
    let normal = normalize(rotate(input.rotation, (o + t * d) / input.scale));
    let lighting = 0.25 + 0.75 * max(dot(normal, normalize(vec3f(-0.4, 0.7, 1))), 0.0);
    return Fragment(vec4f(input.color.rgb * lighting, 1), depth);
}
@fragment fn fs_splat(input: Vertex) -> @location(0) vec4f {
    let p = input.offset;
    let q = input.conic.x * p.x * p.x + 2.0 * input.conic.y * p.x * p.y + input.conic.z * p.y * p.y;
    if (q > select(1.0, 9.0, gaussian_falloff)) { discard; }
    let alpha = select(1.0, input.color.a * exp(-0.5 * q), gaussian_falloff);
    return vec4f(input.color.rgb * alpha, alpha);
}
@fragment fn fs_wire() -> @location(0) vec4f {
    return vec4f(0.3, 0.85, 0.85, 1);
}
