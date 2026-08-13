//! Simplistic mtl loader.

use crate::loader::obj;
use crate::loader::obj::{bounded_warning_text, Words};
use std::io::Result as IoResult;
use std::mem;
use std::path::Path;
use std::str::FromStr;

fn error(line: usize, err: &str) -> ! {
    panic!("At line {}: {}", line, err)
}

/// Parses a mtl file.
pub fn parse_file(path: &Path) -> IoResult<Vec<MtlMaterial>> {
    obj::read_bounded_text_file(path, "MTL").map(|sfile| parse(&sfile))
}

/// Parses a string representing a mtl file.
pub fn parse(string: &str) -> Vec<MtlMaterial> {
    let string = string.strip_prefix('\u{feff}').unwrap_or(string);
    let mut res = Vec::new();
    let mut curr_material = MtlMaterial::new_default("".to_string());

    for (l, line) in string.lines().enumerate() {
        let mut words = obj::split_words(line);
        let Some(w) = words.next() else {
            continue;
        };
        if w.is_empty() || w.as_bytes()[0] == b'#' {
            continue;
        }

        let mut p = obj::split_words(line).peekable();
        let _ = p.next();

        if p.peek().is_none() {
            continue;
        }

        match w {
            // texture name
            "newmtl" => {
                let old = mem::replace(
                    &mut curr_material,
                    MtlMaterial::new_default(parse_name(l, words)),
                );

                if !old.name.is_empty() {
                    res.push(old);
                }
            }
            // ambient color
            "Ka" => curr_material.ambient = parse_color(l, words),
            // diffuse color
            "Kd" => curr_material.diffuse = parse_color(l, words),
            // specular color
            "Ks" => curr_material.specular = parse_color(l, words),
            // shininess
            "Ns" => curr_material.shininess = parse_scalar(l, words),
            // alpha
            "d" => curr_material.alpha = parse_scalar(l, words),
            // Transparency is the inverse of dissolve/opacity in the
            // Wavefront MTL convention (`Tr 0` is opaque, `Tr 1` is clear).
            "Tr" => curr_material.alpha = 1.0 - parse_scalar(l, words),
            // ambient map
            "map_Ka" => curr_material.ambient_texture = Some(parse_name(l, words)),
            // diffuse texture map
            "map_Kd" => curr_material.diffuse_texture = Some(parse_name(l, words)),
            // specular texture map
            "map_Ks" => curr_material.specular_texture = Some(parse_name(l, words)),
            // specular texture map
            "map_d" | "map_opacity" => curr_material.opacity_map = Some(parse_name(l, words)),
            _ => {
                log::warn!(
                    "unknown line {} ignored: `{}'",
                    l,
                    bounded_warning_text(line)
                );
            }
        }
    }

    if !curr_material.name.is_empty() {
        res.push(curr_material);
    }

    res
}

fn parse_name<'a>(_: usize, ws: Words<'a>) -> String {
    let res: Vec<&'a str> = ws.collect();
    res.join(" ")
}

fn parse_color(l: usize, mut ws: Words) -> [f32; 3] {
    let sx = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 0."));
    let sy = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 1."));
    let sz = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 2."));

    let x: Result<f32, _> = FromStr::from_str(sx);
    let y: Result<f32, _> = FromStr::from_str(sy);
    let z: Result<f32, _> = FromStr::from_str(sz);

    let x = finite_or_error(
        l,
        sx,
        x.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sx, e)[..])),
    );
    let y = finite_or_error(
        l,
        sy,
        y.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sy, e)[..])),
    );
    let z = finite_or_error(
        l,
        sz,
        z.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sz, e)[..])),
    );

    [x, y, z]
}

fn parse_scalar(l: usize, mut ws: Words) -> f32 {
    let token = ws
        .next()
        .unwrap_or_else(|| error(l, "1 component was expected, found 0."));
    let value: f32 = token
        .parse()
        .unwrap_or_else(|e| error(l, &format!("failed to parse as f32: {}", e)[..]));
    finite_or_error(l, token, value)
}

fn finite_or_error(line: usize, token: &str, value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        error(line, &format!("`{token}` must be finite"))
    }
}

/// Material information read from a `.mtl` file.
#[derive(Clone)]
pub struct MtlMaterial {
    /// Name of the material.
    pub name: String,
    /// Path to the ambient texture.
    pub ambient_texture: Option<String>,
    /// Path to the diffuse texture.
    pub diffuse_texture: Option<String>,
    /// Path to the specular texture.
    pub specular_texture: Option<String>,
    /// Path to the opacity map.
    pub opacity_map: Option<String>,
    /// The ambient color.
    pub ambient: [f32; 3],
    /// The diffuse color.
    pub diffuse: [f32; 3],
    /// The specular color.
    pub specular: [f32; 3],
    /// The shininess.
    pub shininess: f32,
    /// Alpha blending.
    pub alpha: f32,
}

impl MtlMaterial {
    /// Creates a new mtl material with a name and default values.
    pub fn new_default(name: String) -> MtlMaterial {
        MtlMaterial {
            name,
            shininess: 60.0,
            alpha: 1.0,
            ambient_texture: None,
            diffuse_texture: None,
            specular_texture: None,
            opacity_map: None,
            ambient: [1.0, 1.0, 1.0],
            diffuse: [1.0, 1.0, 1.0],
            specular: [1.0, 1.0, 1.0],
        }
    }

    /// Creates a new mtl material.
    pub fn new(
        name: String,
        shininess: f32,
        alpha: f32,
        ambient: [f32; 3],
        diffuse: [f32; 3],
        specular: [f32; 3],
        ambient_texture: Option<String>,
        diffuse_texture: Option<String>,
        specular_texture: Option<String>,
        opacity_map: Option<String>,
    ) -> MtlMaterial {
        MtlMaterial {
            name,
            ambient,
            diffuse,
            specular,
            ambient_texture,
            diffuse_texture,
            specular_texture,
            opacity_map,
            shininess,
            alpha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;
    use crate::loader::obj::{bounded_warning_text, MAX_WARNING_TEXT_BYTES};
    use crate::scene::AlphaMode;

    #[test]
    #[should_panic(expected = "must be finite")]
    fn rejects_non_finite_color_component() {
        let _ = parse("newmtl bad\nKd NaN 0 0\n");
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn rejects_non_finite_scalar() {
        let _ = parse("newmtl bad\nd inf\n");
    }

    #[test]
    fn unknown_line_warning_text_is_bounded_and_control_safe() {
        let source = format!(
            "prefix\u{1b}[31m{}",
            "x".repeat(MAX_WARNING_TEXT_BYTES + 32)
        );
        let warning = bounded_warning_text(&source);

        assert!(warning.len() <= MAX_WARNING_TEXT_BYTES);
        assert!(!warning.chars().any(char::is_control));
        assert!(warning.starts_with("prefix�"));
    }

    #[test]
    fn transparency_is_converted_to_alpha() {
        let materials = parse("newmtl glass\nTr 0.25\n");

        assert_eq!(materials.len(), 1);
        assert!((materials[0].alpha - 0.75).abs() < f32::EPSILON);
        assert!(AlphaMode::default().is_transparent(materials[0].alpha));
        assert!(!AlphaMode::default().is_transparent(1.0));
    }
}
