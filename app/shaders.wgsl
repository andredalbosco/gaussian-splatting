@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> gaussians3d: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read> gaussians2d: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read> gaussianIndex: array<u32>;

struct Globals {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
}

struct Gaussian3D {
    mean3d: vec3f,
    cov3d: mat3x3f,
    color: vec3f,
    opacity: f32,
}

struct Gaussian2D {
    mean2d: vec2f,
    cov2d: mat2x2f,
    color: vec3f,
    opacity: f32,
}

const vertices = array(
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
    vec2f(0, 0.5),
);

struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clipPos: vec3f,
    @location(1) mean2d: vec2f,
    @location(2) cov2dInv: vec4f,
    @location(3) color: vec3f,
    @location(4) opacity: f32,
    @location(5) xy: vec2f,
}

const quadVertices = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(1, 1),
    vec2f(-1, -1),
    vec2f(1, 1),
    vec2f(-1, 1),
);

fn getRotationMatrix(M: mat4x4f) -> mat3x3f {
    return mat3x3f(M[0].xyz, M[1].xyz, M[2].xyz);
}

fn getJacobianMatrix(mean3d: vec3f) -> mat3x3f {
    let u = (globals.viewMatrix * vec4f(mean3d, 1)).xyz;
    let l = length(u);
    var J = transpose(mat3x3f(
        vec3f(1 / u.z, 0, -u.x / (u.z * u.z)),
        vec3f(0, 1 / u.z, -u.y / (u.z * u.z)),
        u / l,
    ));
    J[0] *= globals.projectionMatrix[0][0];
    J[1] *= globals.projectionMatrix[1][1];
    return J;
}

fn eigenBasis2x2(cov: mat2x2f) -> mat2x2f {
    let a = cov[0][0];
    let b = cov[0][1];
    let c = cov[1][1];

    let t = a + c;
    let d = a * c - b * b;
    let s = sqrt(max(0.0, t * t * 0.25 - d));

    let l1 = t * 0.5 + s;
    let l2 = t * 0.5 - s;

    var e1 = vec2f(b, l1 - a);
    if (abs(b) < 1e-5) {
        e1 = vec2f(1, 0);
    }
    e1 = normalize(e1);
    let e2 = vec2f(-e1.y, e1.x);

    let axis1 = sqrt(max(l1, 1e-5)) * e1;
    let axis2 = sqrt(max(l2, 1e-5)) * e2;

    return mat2x2f(axis1, axis2);
}

fn inverse2x2(m: mat2x2f) -> mat2x2f {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[1][0];
    let d = m[1][1];
    let invDet = 1.0 / (a * d - b * c);
    return mat2x2f(
        vec2f( d, -b) * invDet,
        vec2f(-c,  a) * invDet,
    );
}

fn upper2x2(m: mat3x3f) -> mat2x2f {
    return mat2x2f(m[0].xy, m[1].xy);
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // let gaussian = gaussians3d[input.instanceIndex];
    // let gaussian = gaussians3d[ % 3];
    let gaussian = gaussians3d[gaussianIndex[input.instanceIndex]];

    let R = getRotationMatrix(globals.viewMatrix);
    var J = getJacobianMatrix(gaussian.mean3d);

    var meanClip = globals.projectionMatrix * globals.viewMatrix * vec4f(gaussian.mean3d, 1);
    meanClip = meanClip.xyzw / meanClip.w;
    output.mean2d = meanClip.xy;

    let cov2d = upper2x2(J * R * gaussian.cov3d * transpose(R) * transpose(J));
    let cov2dInv = inverse2x2(cov2d);
    output.cov2dInv = vec4f(cov2dInv[0].xy, cov2dInv[1].xy);

    let cov2dBasis = eigenBasis2x2(cov2d);
    // output.position = vec4f(quadVertices[input.vertexIndex], 0, 1);
    const kSupport = 3.0;
    output.position = vec4f(meanClip.xy + kSupport * cov2dBasis * quadVertices[input.vertexIndex], meanClip.z, 1);
    output.xy = kSupport * quadVertices[input.vertexIndex];

    output.color = gaussian.color;
    output.clipPos = output.position.xyz / output.position.w;
    output.opacity = gaussian.opacity;

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let cov2dInv = mat2x2f(input.cov2dInv.xy, input.cov2dInv.zw);
    let x = input.clipPos.xy - input.mean2d;
    // let alpha = input.opacity * exp(-0.5 * dot(x, cov2dInv * x));
    let alpha = input.opacity * exp(-0.5 * dot(input.xy, input.xy));
    return vec4f(input.color * alpha, alpha);
}
