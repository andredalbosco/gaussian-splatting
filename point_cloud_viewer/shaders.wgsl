struct Globals {
    viewProjection: mat4x4f,
    viewport: vec2f,
    pointSize: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) localPosition: vec2f,
}

const quadVertices = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(1, 1),
    vec2f(-1, -1),
    vec2f(1, 1),
    vec2f(-1, 1),
);

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @location(0) pointPosition: vec4f,
    @location(1) pointColor: vec4f,
) -> VertexOutput {
    var output: VertexOutput;
    let clip = globals.viewProjection * vec4f(pointPosition.xyz, 1);
    let offset = quadVertices[vertexIndex] * pointPosition.w * globals.pointSize * vec2f(2 / globals.viewport.x, 2 / globals.viewport.y);
    output.position = vec4f(clip.xy + offset * clip.w, clip.zw);
    output.color = pointColor.rgb;
    output.localPosition = quadVertices[vertexIndex];
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    if (dot(input.localPosition, input.localPosition) > 1) {
        discard;
    }
    return vec4f(input.color, 1.0);
}
