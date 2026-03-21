@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;

struct Globals {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

struct Gaussian {
    mean: vec3f,
    cov: mat3x3f,
    color: vec3f,
    opacity: f32,
};

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
    @location(0) color: vec3f,
}

const quadVertices = array(
    vec3f(-1, -1, 0),
    vec3f(1, -1, 0),
    vec3f(1, 1, 0),
    vec3f(-1, -1, 0),
    vec3f(1, 1, 0),
    vec3f(-1, 1, 0),
);

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let gaussian = gaussians[input.instanceIndex];
    var worldPos = vec4f(0.1 * quadVertices[input.vertexIndex], 1) + vec4f(gaussian.mean, 0);
    output.position = globals.projectionMatrix * globals.viewMatrix * worldPos;
    output.color = gaussian.color;
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(input.color, 1);
}
