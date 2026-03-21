@group(0) @binding(0) var<uniform> globals: Globals;

struct Globals {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

const vertices = array(
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
    vec2f(0, 0.5),
);

struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var worldPos = vec4f(vertices[input.vertexIndex], 0, 1);
    output.position = globals.projectionMatrix * globals.viewMatrix * worldPos;
    return output;
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1);
}
