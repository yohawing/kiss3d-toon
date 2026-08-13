//! Simplistic obj loader.

use crate::loader::mtl;
use crate::loader::mtl::MtlMaterial;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::GPUVec;
use crate::resource::{AllocationType, BufferType, GpuMesh3d};
use glamx::{Vec2, Vec3};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Read;
use std::io::Result as IoResult;
use std::iter::Filter;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::str::Split;
use std::sync::{Arc, RwLock};

/// The type of vertex coordinates.
pub type Coord = Vec3;
/// The type of normals.
pub type Normal = Vec3;
/// The type of texture coordinates.
pub type UV = Vec2;

/// Iterator through words.
pub type Words<'a> = Filter<Split<'a, fn(char) -> bool>, fn(&&str) -> bool>;

pub(crate) const MAX_WARNING_TEXT_BYTES: usize = 256;
pub(crate) const MAX_TEXT_SOURCE_BYTES: u64 = 1024 * 1024 * 1024;

pub(crate) fn read_bounded_text_file(path: &Path, format: &str) -> IoResult<String> {
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_TEXT_SOURCE_BYTES {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{format} source exceeds {MAX_TEXT_SOURCE_BYTES} bytes"),
        ));
    }

    let mut text = String::new();
    file.take(MAX_TEXT_SOURCE_BYTES.saturating_add(1))
        .read_to_string(&mut text)?;
    if text.len() as u64 > MAX_TEXT_SOURCE_BYTES {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{format} source exceeds {MAX_TEXT_SOURCE_BYTES} bytes"),
        ));
    }
    Ok(text)
}

// TODO: replace by split_whitespaces from rust 1.1
/// Returns an iterator through all the words of a string.
pub fn split_words(s: &str) -> Words<'_> {
    fn is_not_empty(s: &&str) -> bool {
        !s.is_empty()
    }
    let is_not_empty: fn(&&str) -> bool = is_not_empty; // coerce to fn pointer

    fn is_whitespace(c: char) -> bool {
        c.is_whitespace()
    }
    let is_whitespace: fn(char) -> bool = is_whitespace; // coerce to fn pointer!s.is_empty())

    s.split(is_whitespace).filter(is_not_empty)
}

fn error(line: usize, err: &str) -> ! {
    panic!("At line {}: {}", line, err)
}

pub(crate) fn bounded_warning_text(value: &str) -> String {
    let mut bounded = String::new();
    for character in value.chars() {
        let character = if character.is_control() {
            '\u{fffd}'
        } else {
            character
        };
        if bounded.len() + character.len_utf8() > MAX_WARNING_TEXT_BYTES {
            break;
        }
        bounded.push(character);
    }
    bounded
}

fn warn(line: usize, err: &str) {
    log::warn!("at line {}: {}", line, bounded_warning_text(err))
}

/// Parses an OBJ file and returns the meshes it contains.
///
/// Loads a Wavefront OBJ file from disk and parses it into GPU meshes.
/// Each object group in the OBJ file becomes a separate mesh.
///
/// # Arguments
/// * `path` - Path to the .obj file
/// * `mtl_base_dir` - Directory containing the .mtl material files
/// * `basename` - Base name for material lookups
///
/// # Returns
/// `Ok(Vec)` containing tuples of (name, mesh, optional material) for each object in the file,
/// or `Err` if the file couldn't be read
///
/// # Example
/// ```no_run
/// # use kiss3d::loader::obj;
/// # use std::path::Path;
/// let meshes = obj::parse_file(
///     Path::new("model.obj"),
///     Path::new("materials/"),
///     "model"
/// ).unwrap();
/// for (name, mesh, material) in meshes {
///     println!("Loaded object: {}", name);
/// }
/// ```
pub fn parse_file(
    path: &Path,
    mtl_base_dir: &Path,
    basename: &str,
) -> IoResult<Vec<(String, GpuMesh3d, Option<MtlMaterial>)>> {
    read_bounded_text_file(path, "OBJ").map(|sfile| parse(&sfile, mtl_base_dir, basename))
}

/// Index triplet for a face vertex: (coord_idx, uv_idx, normal_idx)
/// Uses i32 to handle negative (relative) indices, or i32::MAX for missing
#[derive(Clone, Copy)]
struct FaceVertex {
    coord: i32,
    uv: i32,
    normal: i32,
}

impl FaceVertex {
    fn new(coord: i32, uv: i32, normal: i32) -> Self {
        Self { coord, uv, normal }
    }
}

/// Parses an OBJ-formatted string and returns the meshes it contains.
///
/// Parses the OBJ format from a string in memory rather than from a file.
///
/// # Arguments
/// * `string` - String containing OBJ-formatted data
/// * `mtl_base_dir` - Directory containing the .mtl material files
/// * `basename` - Base name for material lookups
///
/// # Returns
/// A vector of tuples containing (object name, GPU mesh, optional material)
pub fn parse(
    string: &str,
    mtl_base_dir: &Path,
    basename: &str,
) -> Vec<(String, GpuMesh3d, Option<MtlMaterial>)> {
    let string = string.strip_prefix('\u{feff}').unwrap_or(string);
    let mut coords: Vec<Coord> = Vec::new();
    let mut normals: Vec<Normal> = Vec::new();
    let mut uvs: Vec<UV> = Vec::new();
    let mut groups: HashMap<String, usize> = HashMap::new();
    let mut groups_ids: Vec<Vec<FaceVertex>> = Vec::new();
    let mut curr_group: usize = 0;
    let mut ignore_normals = false;
    let mut ignore_uvs = false;
    let mut mtllib = HashMap::new();
    let mut group2mtl = HashMap::new();
    let mut curr_mtl = None::<MtlMaterial>;

    groups_ids.push(Vec::new());
    let _ = groups.insert(basename.to_string(), 0);

    for (l, line) in string.lines().enumerate() {
        let mut words = split_words(line);
        let Some(w) = words.next() else {
            continue;
        };
        if w.is_empty() || w.as_bytes()[0] == b'#' {
            continue;
        }

        match w {
            "v" => {
                let mut coordinate = parse_v_or_vn(l, &mut words);
                if let Some(w_token) = words.next() {
                    if words.next().is_none() {
                        let w: f32 = w_token.parse().unwrap_or_else(|parse_error| {
                            error(
                                l,
                                &format!(
                                    "failed to parse homogeneous vertex w `{}` as a f32: {}",
                                    w_token, parse_error
                                ),
                            )
                        });
                        if !w.is_finite() || w == 0.0 {
                            error(l, "vertex homogeneous w must be finite and non-zero");
                        }
                        coordinate /= w;
                        if !coordinate.is_finite() {
                            error(
                                l,
                                "homogeneous vertex divide produced a non-finite coordinate",
                            );
                        }
                    } else {
                        warn(
                            l,
                            "additional vertex attributes (including colors) are not supported; keeping position only",
                        );
                    }
                }
                coords.push(coordinate);
            }
            "vn" => {
                if !ignore_normals {
                    normals.push(parse_v_or_vn(l, words))
                }
            }
            "f" => parse_f(
                l,
                words,
                &coords[..],
                &uvs[..],
                &normals[..],
                &mut ignore_uvs,
                &mut ignore_normals,
                &mut groups_ids,
                curr_group,
            ),
            "vt" => {
                if !ignore_uvs {
                    uvs.push(parse_vt(l, words))
                }
            }
            "g" => {
                curr_group = parse_g(l, words, basename, &mut groups, &mut groups_ids);
                let _ = curr_mtl
                    .as_ref()
                    .map(|mtl| group2mtl.insert(curr_group, mtl.clone()));
            }
            "mtllib" => parse_mtllib(l, words, mtl_base_dir, &mut mtllib),
            "usemtl" => {
                curr_group = parse_usemtl(
                    l,
                    words,
                    curr_group,
                    &mtllib,
                    &mut group2mtl,
                    &mut groups,
                    &mut groups_ids,
                    &mut curr_mtl,
                )
            }
            _ => {
                log::warn!(
                    "unknown line {} ignored: `{}'",
                    l,
                    bounded_warning_text(line)
                );
            }
        }
    }

    if ignore_uvs {
        log::warn!("some texture coordinates are missing. Dropping texture coordinates infos for every vertex.");
    }

    if ignore_normals {
        log::warn!("some normals are missing. Dropping normals infos for every vertex.");
    }

    reformat(
        coords,
        if ignore_normals { None } else { Some(normals) },
        if ignore_uvs { None } else { Some(uvs) },
        groups_ids,
        groups,
        group2mtl,
    )
}

fn parse_usemtl<'a>(
    l: usize,
    ws: Words<'a>,
    curr_group: usize,
    mtllib: &HashMap<String, MtlMaterial>,
    group2mtl: &mut HashMap<usize, MtlMaterial>,
    groups: &mut HashMap<String, usize>,
    groups_ids: &mut Vec<Vec<FaceVertex>>,
    curr_mtl: &mut Option<MtlMaterial>,
) -> usize {
    let mname: Vec<&'a str> = ws.collect();
    let mname = mname.join(" ");
    let none = "None";
    if mname[..] != none[..] {
        match mtllib.get(&mname) {
            None => {
                *curr_mtl = None;
                warn(l, &format!("could not find the material {}", mname)[..]);

                curr_group
            }
            Some(m) => {
                if !group2mtl.contains_key(&curr_group) {
                    let _ = group2mtl.insert(curr_group, m.clone());
                    *curr_mtl = Some(m.clone());
                    curr_group
                } else {
                    // multiple usemtls for one group
                    // NOTE: this is a violation of the obj specification, but we support it anyway
                    let mut g = curr_group.to_string();
                    g.push_str(&mname[..]);

                    let new_group = parse_g(
                        l,
                        split_words(&g[..]),
                        "auto_generated_group_",
                        groups,
                        groups_ids,
                    );

                    let _ = group2mtl.insert(new_group, m.clone());
                    *curr_mtl = Some(m.clone());

                    new_group
                }
            }
        }
    } else {
        *curr_mtl = None;
        curr_group
    }
}

fn parse_mtllib<'a>(
    l: usize,
    ws: Words<'a>,
    mtl_base_dir: &Path,
    mtllib: &mut HashMap<String, MtlMaterial>,
) {
    let filename: Vec<&'a str> = ws.collect();
    let filename = filename.join(" ");

    let mut path = PathBuf::new();
    path.push(mtl_base_dir);
    path.push(filename);

    let ms = mtl::parse_file(&path);

    match ms {
        Ok(ms) => {
            for m in ms.into_iter() {
                let _ = mtllib.insert(m.name.to_string(), m);
            }
        }
        Err(err) => warn(l, &format!("{}", err)[..]),
    }
}

fn parse_v_or_vn<'a, I>(l: usize, mut ws: I) -> Vec3
where
    I: Iterator<Item = &'a str>,
{
    let sx = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 0."));
    let sy = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 1."));
    let sz = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 2."));

    let x = parse_f32(l, sx);
    let y = parse_f32(l, sy);
    let z = parse_f32(l, sz);

    Vec3::new(x, y, z)
}

fn parse_f32(line: usize, token: &str) -> f32 {
    let value: f32 = FromStr::from_str(token).unwrap_or_else(|error_value| {
        error(
            line,
            &format!("failed to parse `{}` as a f32: {}", token, error_value),
        )
    });
    if !value.is_finite() {
        error(line, &format!("`{token}` must be finite"));
    }
    value
}

fn parse_f<'a>(
    l: usize,
    ws: Words<'a>,
    coords: &[Vec3],
    uvs: &[Vec2],
    normals: &[Vec3],
    ignore_uvs: &mut bool,
    ignore_normals: &mut bool,
    groups_ids: &mut [Vec<FaceVertex>],
    curr_group: usize,
) {
    // Four formats possible: v   v/t   v//n   v/t/n
    let face_start = groups_ids[curr_group].len();
    let mut i = 0;
    for word in ws {
        let mut coord_idx = i32::MAX;
        let mut uv_idx = i32::MAX;
        let mut normal_idx = i32::MAX;

        for (idx, w) in word.split('/').enumerate() {
            if idx == 0 || !w.is_empty() {
                let parsed_idx: Result<i32, _> = FromStr::from_str(w);
                match parsed_idx {
                    Ok(id) => {
                        let adjusted = id - 1; // OBJ indices are 1-based
                        match idx {
                            0 => coord_idx = adjusted,
                            1 => uv_idx = adjusted,
                            2 => normal_idx = adjusted,
                            _ => {}
                        }
                    }
                    Err(e) => error(l, &format!("failed to parse `{}' as a i32: {}", w, e)[..]),
                }
            }
        }

        if i > 2 {
            // on the fly triangulation as triangle fan
            let g = &mut groups_ids[curr_group];
            let p1 = g[face_start];
            let p2 = g[g.len() - 1];
            g.push(p1);
            g.push(p2);
        }

        if uv_idx == i32::MAX {
            *ignore_uvs = true;
        }

        if normal_idx == i32::MAX {
            *ignore_normals = true;
        }

        // Handle relative indices
        let x = if coord_idx < 0 {
            coords.len() as i32 + coord_idx + 1
        } else {
            coord_idx
        };

        let y = if uv_idx < 0 {
            uvs.len() as i32 + uv_idx + 1
        } else {
            uv_idx
        };

        let z = if normal_idx < 0 {
            normals.len() as i32 + normal_idx + 1
        } else {
            normal_idx
        };

        assert!(x >= 0 && y >= 0 && z >= 0);
        groups_ids[curr_group].push(FaceVertex::new(x, y, z));

        i += 1;
    }

    // there is not enough vertex to form a triangle. Complete it.
    if i < 2 {
        for _ in 0usize..3 - i {
            let last = *groups_ids[curr_group].last().unwrap();
            groups_ids[curr_group].push(last);
        }
    }
}

fn parse_vt(l: usize, mut ws: Words) -> UV {
    let sx = ws
        .next()
        .unwrap_or_else(|| error(l, "at least 2 components were expected, found 0."));
    let sy = ws
        .next()
        .unwrap_or_else(|| error(l, "at least 2 components were expected, found 1."));

    let x = parse_f32(l, sx);
    let y = parse_f32(l, sy);

    Vec2::new(x, y)
}

fn parse_g<'a>(
    _: usize,
    ws: Words<'a>,
    prefix: &str,
    groups: &mut HashMap<String, usize>,
    groups_ids: &mut Vec<Vec<FaceVertex>>,
) -> usize {
    let suffix: Vec<&'a str> = ws.collect();
    let suffix = suffix.join(" ");
    let name = if suffix.is_empty() {
        prefix.to_string()
    } else {
        format!("{}/{}", prefix, suffix)
    };

    match groups.entry(name) {
        Entry::Occupied(entry) => *entry.into_mut(),
        Entry::Vacant(entry) => {
            groups_ids.push(Vec::new());

            let val = groups_ids.len() - 1;
            *entry.insert(val)
        }
    }
}

fn reformat(
    coords: Vec<Coord>,
    normals: Option<Vec<Normal>>,
    uvs: Option<Vec<UV>>,
    groups_ids: Vec<Vec<FaceVertex>>,
    groups: HashMap<String, usize>,
    group2mtl: HashMap<usize, MtlMaterial>,
) -> Vec<(String, GpuMesh3d, Option<MtlMaterial>)> {
    let mut vt2id: HashMap<[i32; 3], VertexIndex> = HashMap::new();
    let mut vertex_ids: Vec<VertexIndex> = Vec::new();
    let mut resc: Vec<Coord> = Vec::new();
    let mut resn: Option<Vec<Normal>> = normals.as_ref().map(|_| Vec::new());
    let mut resu: Option<Vec<UV>> = uvs.as_ref().map(|_| Vec::new());
    let mut resfs: Vec<Vec<[VertexIndex; 3]>> = Vec::new();
    let mut allfs: Vec<[VertexIndex; 3]> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut mtls: Vec<Option<MtlMaterial>> = Vec::new();

    for (name, i) in groups.into_iter() {
        names.push(name);
        mtls.push(group2mtl.get(&i).cloned());

        for fv in groups_ids[i].iter() {
            let point_key = [fv.coord, fv.uv, fv.normal];
            let idx = match vt2id.get(&point_key) {
                Some(i) => {
                    vertex_ids.push(*i);
                    None
                }
                None => {
                    let idx = resc.len() as VertexIndex;

                    resc.push(coords[fv.coord as usize]);

                    let _ = resu
                        .as_mut()
                        .map(|l| l.push((*uvs.as_ref().unwrap())[fv.uv as usize]));
                    let _ = resn
                        .as_mut()
                        .map(|l| l.push((*normals.as_ref().unwrap())[fv.normal as usize]));

                    vertex_ids.push(idx);

                    Some(idx)
                }
            };

            let _ = idx.map(|i| vt2id.insert(point_key, i));
        }

        let mut resf = Vec::with_capacity(vertex_ids.len() / 3);

        assert!(vertex_ids.len().is_multiple_of(3));

        for f in vertex_ids[..].chunks(3) {
            resf.push([f[0], f[1], f[2]]);
            allfs.push([f[0], f[1], f[2]]);
        }

        resfs.push(resf);
        vertex_ids.clear();
    }

    let resn = resn.unwrap_or_else(|| GpuMesh3d::compute_normals_array(&resc[..], &allfs[..]));
    let resn = Arc::new(RwLock::new(GPUVec::new(
        resn,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));
    let resu = resu.unwrap_or_else(|| std::iter::repeat_n(Vec2::ZERO, resc.len()).collect());
    let resu = Arc::new(RwLock::new(GPUVec::new(
        resu,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));
    let resc = Arc::new(RwLock::new(GPUVec::new(
        resc,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));

    let mut meshes = Vec::new();
    for ((fs, name), mtl) in resfs.into_iter().zip(names).zip(mtls) {
        if !fs.is_empty() {
            let fs = Arc::new(RwLock::new(GPUVec::new(
                fs,
                BufferType::ElementArray,
                AllocationType::StaticDraw,
            )));
            let mesh =
                GpuMesh3d::new_with_gpu_vectors(resc.clone(), fs, resn.clone(), resu.clone());
            meshes.push((name, mesh, mtl))
        }
    }

    meshes
}

#[cfg(test)]
mod tests {
    use super::{parse, parse_f, parse_file, split_words, MAX_TEXT_SOURCE_BYTES};
    use glamx::{Vec2, Vec3};
    use std::fs::{self, File};
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn polygon_triangulation_keeps_each_face_anchor_and_attributes() {
        let coords = vec![Vec3::ZERO; 9];
        let uvs = vec![Vec2::ZERO; 9];
        let normals = vec![Vec3::ZERO; 9];
        let mut ignore_uvs = false;
        let mut ignore_normals = false;
        let mut groups = vec![Vec::new()];

        parse_f(
            0,
            split_words("1/1/1 2/2/2 3/3/3 4/4/4 5/5/5"),
            &coords,
            &uvs,
            &normals,
            &mut ignore_uvs,
            &mut ignore_normals,
            &mut groups,
            0,
        );
        parse_f(
            0,
            split_words("-4/-4/-4 -3/-3/-3 -2/-2/-2 -1/-1/-1"),
            &coords,
            &uvs,
            &normals,
            &mut ignore_uvs,
            &mut ignore_normals,
            &mut groups,
            0,
        );

        let coordinates: Vec<_> = groups[0].iter().map(|vertex| vertex.coord).collect();
        let uvs: Vec<_> = groups[0].iter().map(|vertex| vertex.uv).collect();
        let normals: Vec<_> = groups[0].iter().map(|vertex| vertex.normal).collect();
        let expected = vec![0, 1, 2, 0, 2, 3, 0, 3, 4, 5, 6, 7, 5, 7, 8];
        assert_eq!(coordinates, expected);
        assert_eq!(uvs, expected);
        assert_eq!(normals, expected);
        assert!(!ignore_uvs);
        assert!(!ignore_normals);
        assert_eq!(
            groups[0].first().copied().map(|vertex| vertex.coord),
            Some(0)
        );
    }

    #[test]
    fn mixed_attributes_recompute_without_indexing_missing_values() {
        let obj = "v 0 0 0\n\
v 1 0 0\n\
v 0 1 0\n\
v 0 0 1\n\
v 1 0 1\n\
v 0 1 1\n\
vt 0 0\n\
vt 1 0\n\
vt 0 1\n\
vn 0 0 1\n\
f 1/1/1 2/2/1 3/3/1\n\
f 4 5 6\n";

        let meshes = parse(obj, Path::new("."), "mixed-normals");
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].1.normals().read().unwrap().len(), 6);
        assert_eq!(meshes[0].1.uvs().read().unwrap().len(), 6);
    }

    #[test]
    fn vertex_color_components_keep_position_with_explicit_unsupported_warning_contract() {
        let obj = "v 0 0 0 1 0 0\n\
v 1 0 0 0 1 0\n\
v 0 1 0 0 0 1\n\
f 1 2 3\n";

        let meshes = parse(obj, Path::new("."), "vertex-colors");
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].1.coords().read().unwrap().len(), 3);
    }

    #[test]
    fn homogeneous_vertex_w_is_applied_before_mesh_build() {
        let obj = "v 2 4 6 2\n\
v 1 0 0\n\
v 0 1 0\n\
f 1 2 3\n";

        let meshes = parse(obj, Path::new("."), "homogeneous-w");
        let coordinates = meshes[0].1.coords().read().unwrap();
        let coordinates = coordinates.data().as_ref().unwrap();
        assert_eq!(coordinates[0], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn small_nonzero_homogeneous_w_is_valid() {
        let obj = "v 1e-8 2e-8 3e-8 1e-8\n\
v 1 0 0\n\
v 0 1 0\n\
f 1 2 3\n";

        let meshes = parse(obj, Path::new("."), "small-homogeneous-w");
        let coordinates = meshes[0].1.coords().read().unwrap();
        let coordinates = coordinates.data().as_ref().unwrap();
        assert_eq!(coordinates[0], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    #[should_panic(expected = "homogeneous vertex divide produced a non-finite coordinate")]
    fn homogeneous_divide_overflow_is_rejected() {
        let obj = "v 3.4e38 0 0 1e-6\n\
v 1 0 0\n\
v 0 1 0\n\
f 1 2 3\n";

        let _ = parse(obj, Path::new("."), "overflowing-homogeneous-w");
    }

    #[test]
    fn non_finite_vertex_attributes_are_rejected_before_mesh_build() {
        for (source, label) in [
            ("v NaN 0 0\n", "non-finite-position"),
            ("vn 0 inf 0\n", "non-finite-normal"),
            ("vt 0 -inf\n", "non-finite-uv"),
        ] {
            let result = std::panic::catch_unwind(|| parse(source, Path::new("."), label));
            assert!(result.is_err(), "{} should be rejected", label);
        }
    }

    #[test]
    fn obj_and_mtl_file_loaders_reject_oversized_text_before_reading() {
        let path = std::env::temp_dir().join(format!(
            "kiss3d-text-budget-{}-{}.obj",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after epoch")
                .as_nanos()
        ));
        let file = File::create(&path).expect("temporary source should be creatable");
        file.set_len(MAX_TEXT_SOURCE_BYTES + 1)
            .expect("temporary source should be sparse-expandable");
        drop(file);

        let obj_error = parse_file(&path, Path::new("."), "oversized")
            .err()
            .expect("oversized OBJ must be rejected");
        assert!(obj_error.to_string().contains("source exceeds"));
        let mtl_error = crate::loader::mtl::parse_file(&path)
            .err()
            .expect("oversized MTL must be rejected");
        assert!(mtl_error.to_string().contains("source exceeds"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn utf8_bom_does_not_drop_first_obj_or_mtl_declaration() {
        let meshes = parse(
            "\u{feff}v 0 0 0\n\
v 1 0 0\n\
v 0 1 0\n\
f 1 2 3\n",
            Path::new("."),
            "bom",
        );
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].1.coords().read().unwrap().len(), 3);

        let materials = crate::loader::mtl::parse("\u{feff}newmtl bom\nKd 1 0 0\n");
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0].name, "bom");
    }
}
