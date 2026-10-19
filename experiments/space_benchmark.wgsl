@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read_write> results: array<Projected>;
@group(1) @binding(0) var<storage, read> projected: array<Projected>;

struct Input { position: vec4f, scale: vec4f, rotation: vec4f, color: vec4f }
struct Camera {
    view_projection: mat4x4f,
    view: mat4x4f,
    viewport_focal: vec4f,
    projection_count: vec4f,
}
struct Projected { clip_pos: vec4f, covariance: vec4f, color: vec4f }
struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) offset: vec2f,
    @location(1) @interpolate(flat) conic: vec3f,
    @location(2) @interpolate(flat) color: vec4f,
}
// 0: pixel covariance / NDC output; 1: pixel / clip; 2: NDC / clip; 3: clip / clip.
override method: u32 = 0u;
const corners = array(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));

fn quat_to_matrix(q: vec4f) -> mat3x3f {
    let d = 2.0 * q.xyz;
    let xx = q.x*d.x; let yy = q.y*d.y; let zz = q.z*d.z;
    let xy = q.x*d.y; let xz = q.x*d.z; let yz = q.y*d.z;
    let wx = q.w*d.x; let wy = q.w*d.y; let wz = q.w*d.z;
    return mat3x3f(vec3f(1-yy-zz,xy+wz,xz-wy),
                   vec3f(xy-wz,1-xx-zz,yz+wx),
                   vec3f(xz+wy,yz-wx,1-xx-yy));
}
fn project(index: u32) -> Projected {
    let g = inputs[index];
    let clip_pos = camera.view_projection*g.position;
    let view_pos = (camera.view*g.position).xyz;
    let pixel_to_ndc = 2.0/camera.viewport_focal.xy;
    var f = camera.viewport_focal.zw/view_pos.z;
    var blur = vec2f(0.3);
    if (method == 2u) {
        f = camera.projection_count.xy/view_pos.z;
        blur = 0.3*pixel_to_ndc*pixel_to_ndc;
    }
    if (method == 3u) {
        // clip_pos.w = view_pos.z for our perspective projection.
        f = camera.projection_count.xy;
        let pixel_to_clip = pixel_to_ndc*clip_pos.w;
        blur = 0.3*pixel_to_clip*pixel_to_clip;
    }
    let J = mat3x2f(vec2f(f.x,0),vec2f(0,f.y),-f*view_pos.xy/view_pos.z);
    let W = mat3x3f(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz);
    let R = quat_to_matrix(g.rotation);
    let M = mat3x3f(R[0]*g.scale.x,R[1]*g.scale.y,R[2]*g.scale.z);
    let B = J*W*M;
    let C = B*transpose(B)+mat2x2f(blur.x,0,0,blur.y);
    return Projected(clip_pos,vec4f(C[0][0],C[0][1],C[1][1],0),g.color);
}
@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < u32(camera.projection_count.z)) { results[id.x] = project(id.x); }
}
fn vertex(p: Projected, v: u32) -> Vertex {
    var out: Vertex;
    out.position = vec4f(2,2,0,1);
    if (p.clip_pos.w <= 0 || p.clip_pos.z < 0 || p.clip_pos.z > p.clip_pos.w) { return out; }
    let C = p.covariance.xyz;
    let det = C.x*C.z-C.y*C.y;
    if (det <= 0) { return out; }
    out.offset = corners[v]*3.0*sqrt(C.xz);
    out.conic = vec3f(C.z,-C.y,C.x)/det;
    out.color = p.color;
    if (method == 0u) {
        out.position = vec4f(p.clip_pos.xy/p.clip_pos.w+out.offset*2.0/camera.viewport_focal.xy,
                             p.clip_pos.z/p.clip_pos.w,1);
    } else if (method == 1u) {
        out.position = vec4f(p.clip_pos.xy+out.offset*(2.0/camera.viewport_focal.xy)*p.clip_pos.w,p.clip_pos.zw);
    } else if (method == 2u) {
        out.position = vec4f(p.clip_pos.xy+out.offset*p.clip_pos.w,p.clip_pos.zw);
    } else {
        out.position = vec4f(p.clip_pos.xy+out.offset,p.clip_pos.zw);
    }
    return out;
}
@vertex fn vs_direct(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    return vertex(project(i),v);
}
// Keep the complete vertex work but shift all primitives outside the clip volume.
@vertex fn vs_math(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    var out = vertex(project(i),v);
    out.position.x = abs(out.position.x)+2.0*out.position.w;
    return out;
}
@fragment fn fs_main(in: Vertex) -> @location(0) vec4f {
    let p = in.offset;
    let q = in.conic.x*p.x*p.x+2.0*in.conic.y*p.x*p.y+in.conic.z*p.y*p.y;
    if (q > 9) { discard; }
    let alpha = min(0.99,in.color.a*exp(-0.5*q));
    if (alpha < 1.0/255.0) { discard; }
    return vec4f(in.color.rgb*alpha,alpha);
}
