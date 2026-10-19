@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read_write> results: array<Projected>;
@group(1) @binding(0) var<storage, read> projected: array<Projected>;

// Input is either quaternion/scale or six precomputed world covariance entries.
struct Input { position: vec4f, scale: vec4f, rotation: vec4f, color: vec4f }
struct Camera {
    view_projection: mat4x4f,
    view: mat4x4f,
    viewport_focal: vec4f,
    limits_count: vec4f,
}
struct Projected { clip_pos: vec4f, covariance: vec4f, color: vec4f }
struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) offset: vec2f,
    @location(1) @interpolate(flat) conic: vec3f,
    @location(2) @interpolate(flat) color: vec4f,
}
override method: u32 = 0u;
const corners = array(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn rotation_matrix(q: vec4f) -> mat3x3f {
    let d = 2.0 * q.xyz;
    let xx = q.x*d.x; let yy = q.y*d.y; let zz = q.z*d.z;
    let xy = q.x*d.y; let xz = q.x*d.z; let yz = q.y*d.z;
    let wx = q.w*d.x; let wy = q.w*d.y; let wz = q.w*d.z;
    return mat3x3f(vec3f(1-yy-zz,xy+wz,xz-wy),
                   vec3f(xy-wz,1-xx-zz,yz+wx),
                   vec3f(xz+wy,yz-wx,1-xx-yy));
}
fn scaled_rotation(g: Input) -> mat3x3f {
    let R = rotation_matrix(g.rotation);
    return mat3x3f(R[0]*g.scale.x,R[1]*g.scale.y,R[2]*g.scale.z);
}
fn view_rotation() -> mat3x3f {
    return mat3x3f(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz);
}
fn packed_covariance(g: Input) -> mat3x3f {
    let a = g.scale.xyz; let b = g.rotation.xyz;
    return mat3x3f(vec3f(a.x,a.y,a.z),vec3f(a.y,b.x,b.y),vec3f(a.z,b.y,b.z));
}
fn covariance_world(g: Input) -> mat3x3f {
    if (method == 5u || method == 7u || method == 9u || method == 10u) {
        return packed_covariance(g);
    }
    let M = scaled_rotation(g);
    if (method == 6u) {
        // PlayCanvas constructs only the six unique entries using row dot products.
        let N = transpose(M);
        let a = vec3f(dot(N[0],N[0]),dot(N[0],N[1]),dot(N[0],N[2]));
        let b = vec3f(dot(N[1],N[1]),dot(N[1],N[2]),dot(N[2],N[2]));
        return mat3x3f(vec3f(a.x,a.y,a.z),vec3f(a.y,b.x,b.y),vec3f(a.z,b.y,b.z));
    }
    return M * transpose(M);
}
fn quotient_axis(g: Input, clip_pos: vec4f, v: vec3f) -> vec2f {
    let a = camera.view_projection * vec4f(quat_rotate(g.rotation,v),0);
    return (a.xy*clip_pos.w-clip_pos.xy*a.w)/(clip_pos.w*clip_pos.w)*camera.viewport_focal.xy*0.5;
}
fn covariance_2d(g: Input, clip_pos: vec4f) -> vec3f {
    if (method == 0u) {
        let ax = quotient_axis(g,clip_pos,vec3f(g.scale.x,0,0));
        let ay = quotient_axis(g,clip_pos,vec3f(0,g.scale.y,0));
        let az = quotient_axis(g,clip_pos,vec3f(0,0,g.scale.z));
        let x = vec3f(ax.x,ay.x,az.x); let y = vec3f(ax.y,ay.y,az.y);
        return vec3f(dot(x,x),dot(x,y),dot(y,y));
    }
    if (method == 1u || method == 2u) {
        let ndc = clip_pos.xy/clip_pos.w;
        let s = camera.viewport_focal.xy*(0.5/clip_pos.w);
        let vp = camera.view_projection;
        let jx = (vec3f(vp[0].x,vp[1].x,vp[2].x)-ndc.x*vec3f(vp[0].w,vp[1].w,vp[2].w))*s.x;
        let jy = (vec3f(vp[0].y,vp[1].y,vp[2].y)-ndc.y*vec3f(vp[0].w,vp[1].w,vp[2].w))*s.y;
        var x: vec3f; var y: vec3f;
        if (method == 1u) {
            let R = rotation_matrix(g.rotation);
            x = (jx*R)*g.scale.xyz;
            y = (jy*R)*g.scale.xyz;
        } else {
            let inverse_q = vec4f(-g.rotation.xyz,g.rotation.w);
            x = quat_rotate(inverse_q,jx)*g.scale.xyz;
            y = quat_rotate(inverse_q,jy)*g.scale.xyz;
        }
        return vec3f(dot(x,x),dot(x,y),dot(y,y));
    }
    let p = (camera.view*g.position).xyz;
    let inv_z = 1.0/p.z;
    var uv = p.xy*inv_z;
    // gsplat perspective clamp; inactive for the matched timing scenes.
    if (method == 4u || method == 5u) {
        uv = clamp(uv,-camera.limits_count.xy,camera.limits_count.xy);
    }
    let f = camera.viewport_focal.zw*inv_z;
    let W = view_rotation();
    if (method == 3u) {
        let M = W*scaled_rotation(g);
        let x = vec3f(M[0].x,M[1].x,M[2].x);
        let y = vec3f(M[0].y,M[1].y,M[2].y);
        let z = vec3f(M[0].z,M[1].z,M[2].z);
        let bx = x-uv.x*z; let by = y-uv.y*z;
        return vec3f(dot(bx,bx)*f.x*f.x,dot(bx,by)*f.x*f.y,dot(by,by)*f.y*f.y);
    }
    let C = covariance_world(g);
    let J = mat3x2f(vec2f(f.x,0),vec2f(0,f.y),-f*uv);
    if (method == 4u || method == 5u) {
        // gsplat: covariance world -> camera, rectangular J * C_camera * J^T.
        let C_camera = W*C*transpose(W);
        let cov = J*C_camera*transpose(J);
        return vec3f(cov[0][0],cov[0][1],cov[1][1]);
    }
    if (method == 6u || method == 7u) {
        // PlayCanvas/original 3DGS convention: T = W^T * J^T.
        let Jt = mat3x3f(vec3f(f.x,0,-f.x*uv.x),vec3f(0,f.y,-f.y*uv.y),vec3f(0));
        let T = transpose(W)*Jt;
        let cov = transpose(T)*C*T;
        return vec3f(cov[0][0],cov[0][1],cov[1][1]);
    }
    if (method == 10u) {
        let V = W*C*transpose(W);
        let hx = V[0][2]-uv.x*V[2][2];
        let hy = V[1][2]-uv.y*V[2][2];
        return vec3f((V[0][0]-uv.x*(V[0][2]+hx))*f.x*f.x,
                     (V[0][1]-uv.x*V[1][2]-uv.y*hx)*f.x*f.y,
                     (V[1][1]-uv.y*(V[1][2]+hy))*f.y*f.y);
    }
    // Symmetric quadratic form: only evaluate three output entries.
    let T = J*W;
    let x = vec3f(T[0].x,T[1].x,T[2].x);
    let y = vec3f(T[0].y,T[1].y,T[2].y);
    let Cx = C*x; let Cy = C*y;
    return vec3f(dot(x,Cx),dot(x,Cy),dot(y,Cy));
}
fn project(index: u32) -> Projected {
    let g = inputs[index];
    let clip_pos = camera.view_projection*g.position;
    let cov = covariance_2d(g,clip_pos)+vec3f(0.3,0,0.3);
    return Projected(clip_pos,vec4f(cov,0),g.color);
}
@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < u32(camera.limits_count.z)) { results[id.x] = project(id.x); }
}
fn vertex(p: Projected, vertex_index: u32) -> Vertex {
    var out: Vertex;
    out.position = vec4f(2,2,0,1);
    if (p.clip_pos.w <= 0 || p.clip_pos.z < 0 || p.clip_pos.z > p.clip_pos.w) { return out; }
    let cov = p.covariance.xyz;
    let det = cov.x*cov.z-cov.y*cov.y;
    if (det <= 0) { return out; }
    out.offset = corners[vertex_index]*3.0*sqrt(cov.xz);
    out.position = vec4f(p.clip_pos.xy/p.clip_pos.w+out.offset*2.0/camera.viewport_focal.xy,p.clip_pos.z/p.clip_pos.w,1);
    out.conic = vec3f(cov.z,-cov.y,cov.x)/det;
    out.color = p.color;
    return out;
}
@vertex fn vs_direct(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    return vertex(project(i),v);
}
@vertex fn vs_prepared(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    return vertex(projected[i],v);
}
@fragment fn fs_main(in: Vertex) -> @location(0) vec4f {
    let p = in.offset;
    let q = in.conic.x*p.x*p.x+2.0*in.conic.y*p.x*p.y+in.conic.z*p.y*p.y;
    if (q > 9) { discard; }
    let alpha = min(0.99,in.color.a*exp(-0.5*q));
    if (alpha < 1.0/255.0) { discard; }
    return vec4f(in.color.rgb*alpha,alpha);
}

// Diagnostic: keep covariance-dependent positions, but rasterize no fragments.
fn offscreen_vertex(p: Projected, v: u32) -> Vertex {
    var out = vertex(p,v);
    let cov = p.covariance.xyz;
    let xy = vec2f(cov.x,cov.z)+cov.y*0.1+p.clip_pos.xy*0.01+out.offset*0.001;
    out.position = vec4f(vec2f(2)+abs(xy)*1e-6,0.5,1);
    return out;
}
@vertex fn vs_math(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    return offscreen_vertex(project(i),v);
}
@vertex fn vs_math_prepared(@builtin(vertex_index) v: u32,@builtin(instance_index) i: u32) -> Vertex {
    return offscreen_vertex(projected[i],v);
}
