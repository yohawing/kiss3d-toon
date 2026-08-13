struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

struct TriangleVertex {
    position: vec3<f32>,
    depth_bias: f32,
    color: vec4<f32>,
}

@group(0) @binding(1)
var<storage, read> vertices: array<TriangleVertex>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];
    var clip = frame.proj * frame.view * vec4(vertex.position, 1.0);
    clip.z *= 1.0 - clamp(vertex.depth_bias, 0.0, 0.9999);

    var out: VertexOutput;
    out.clip_position = clip;
    out.color = vertex.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
