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
    output.position = vec4f(vertices[input.vertexIndex], 0, 1);
    return output;
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1);
}
