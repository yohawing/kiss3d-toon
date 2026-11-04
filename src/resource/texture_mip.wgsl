@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(positions[index], 0.0, 1.0);
}

// Preserve the existing 2x2 box and one-pixel-axis behavior. sRGB views decode
// RGB on load and encode on render-target write; alpha/data stay linear.
@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let origin = vec2<u32>(position.xy) * 2u;
    let size = textureDimensions(source);
    var sum = vec4(0.0);
    var count = 0.0;
    for (var y = 0u; y < 2u; y++) {
        for (var x = 0u; x < 2u; x++) {
            let p = origin + vec2(x, y);
            if (all(p < size)) {
                sum += textureLoad(source, vec2<i32>(p), 0);
                count += 1.0;
            }
        }
    }
    return sum / count;
}
