//! Lightweight FBX loader backed by `ufbx`.
//!
//! Supports triangle meshes, node hierarchy animation baked at the source FPS,
//! linear four-weight skinning, UV0, normals, and scalar PBR factors. Textures,
//! morph animation, curves/surfaces, and FBX animation interpolation details are
//! intentionally outside this bounded loader slice.

use crate::color::Color;
use crate::resource::{GpuMesh3d, MaterialManager3d, SkinVertexData, TextureManager};
use crate::scene::{
    AnimationChannel, AnimationClip, AnimationPlayer, Interpolation as SceneInterpolation,
    Object3d, SceneNode3d, Skin3d,
};
use glamx::{Mat3, Mat4, Pose3, Quat, Vec2, Vec3};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::path::Path;
use std::rc::Rc;

const DEFAULT_ANIMATION_FPS: f64 = 30.0;
const MAX_ANIMATION_FPS: f64 = 60.0;
const MAX_BAKED_NODE_SAMPLES: usize = 2_000_000;

/// A loaded FBX subtree and its baked animation clips.
pub struct FbxModel {
    pub root: SceneNode3d,
    pub player: AnimationPlayer,
}

/// Read-only FBX animation metadata for editor Timeline projection.
pub struct FbxAnimationMetadata {
    pub clips: Vec<FbxAnimationClipMetadata>,
}

pub struct FbxAnimationClipMetadata {
    pub label: String,
    pub duration: f32,
    pub tracks: Vec<FbxAnimationTrackMetadata>,
}

pub struct FbxAnimationTrackMetadata {
    pub node_index: usize,
    pub node_label: String,
    pub key_times: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaterialFactors {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
}

impl Default for MaterialFactors {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

#[derive(Clone)]
struct CpuSkin {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
    vertex_joints: Vec<[u32; 4]>,
    vertex_weights: Vec<[f32; 4]>,
}

#[derive(Clone)]
struct SkinTemplate {
    cluster_to_joint: Vec<Option<u32>>,
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
}

struct CpuPrimitive {
    node_index: usize,
    positions: Vec<Vec3>,
    faces: Vec<[u32; 3]>,
    normals: Option<Vec<Vec3>>,
    uvs: Option<Vec<Vec2>>,
    material: MaterialFactors,
    skin: Option<CpuSkin>,
}

struct ClipLayout {
    label: String,
    duration: f32,
    source_times: Vec<f64>,
    key_times: Vec<f32>,
    nodes: Vec<usize>,
}

/// Loads an FBX file, including supported skeletal/node animations.
pub fn load(path: &Path) -> Result<FbxModel, String> {
    let scene = load_scene(path, true)?;
    build_model(&scene)
}

/// Extracts the same baked clip layout used by [`load`] without building GPU data.
pub fn animation_metadata(path: &Path) -> Result<FbxAnimationMetadata, String> {
    let scene = load_scene(path, true)?;
    let layouts = animation_layouts(&scene)?;
    let clips = layouts
        .into_iter()
        .map(|layout| {
            let key_times = layout.key_times;
            FbxAnimationClipMetadata {
                label: layout.label,
                duration: layout.duration,
                tracks: layout
                    .nodes
                    .into_iter()
                    .map(|node_index| FbxAnimationTrackMetadata {
                        node_index,
                        node_label: node_label(&scene.nodes[node_index], node_index),
                        key_times: key_times.clone(),
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(FbxAnimationMetadata { clips })
}

fn load_scene(path: &Path, load_animation: bool) -> Result<ufbx::SceneRoot, String> {
    let path_text = path
        .to_str()
        .ok_or_else(|| format!("FBX path is not valid UTF-8: {}", path.display()))?;
    ufbx::load_file(path_text, load_options(load_animation)).map_err(format_ufbx_error)
}

fn load_options<'a>(load_animation: bool) -> ufbx::LoadOpts<'a> {
    ufbx::LoadOpts {
        ignore_animation: !load_animation,
        ignore_embedded: true,
        load_external_files: false,
        generate_missing_normals: true,
        normalize_normals: true,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        target_unit_meters: 1.0,
        ..Default::default()
    }
}

fn build_model(scene: &ufbx::Scene) -> Result<FbxModel, String> {
    let primitives = extract_primitives(scene)?;
    if primitives.is_empty() {
        return Err("FBX contains no supported triangle meshes".to_string());
    }

    let mut node_map = Vec::with_capacity(scene.nodes.len());
    for node in scene.nodes.iter() {
        let (translation, rotation, scale) = transform_parts(node.local_transform)?;
        let mut scene_node =
            SceneNode3d::new(scale, Pose3::from_parts(translation, rotation), None);
        scene_node.set_visible(node.visible);
        node_map.push(scene_node);
    }

    let node_indices = node_index_map(scene);
    let mut root = SceneNode3d::empty();
    for (index, node) in scene.nodes.iter().enumerate() {
        if let Some(parent) = node.parent.as_deref() {
            let parent_index = *node_indices
                .get(&parent.element.element_id)
                .ok_or_else(|| format!("FBX node '{}' has an unknown parent", node.element.name))?;
            let child = node_map[index].clone();
            node_map[parent_index].add_child(child);
        } else {
            root.add_child(node_map[index].clone());
        }
    }

    let material = MaterialManager3d::get_global_manager(|manager| manager.get_default());
    let texture = TextureManager::get_global_manager(|manager| manager.get_default());
    for primitive in primitives {
        let mut gpu = GpuMesh3d::new(
            primitive.positions,
            primitive.faces,
            primitive.normals,
            primitive.uvs,
            false,
        );
        let skin_binding = if let Some(skin) = primitive.skin {
            gpu.set_skin_vertices(SkinVertexData::new(skin.vertex_joints, skin.vertex_weights));
            let joints = skin
                .joints
                .iter()
                .map(|index| node_map[*index].downgrade())
                .collect();
            Some(Skin3d::new(joints, skin.inverse_bind))
        } else {
            None
        };

        let factors = primitive.material;
        let mut object = Object3d::new(
            Rc::new(RefCell::new(gpu)),
            Color::new(
                factors.base_color[0],
                factors.base_color[1],
                factors.base_color[2],
                factors.base_color[3],
            ),
            texture.clone(),
            material.clone(),
        );
        object.set_metallic(factors.metallic);
        object.set_roughness(factors.roughness);
        if let Some(skin) = skin_binding {
            object.set_skin(skin);
        }
        node_map[primitive.node_index].add_object(Vec3::ONE, Pose3::IDENTITY, object);
    }

    let clips = build_animation_clips(scene, &node_map)?;
    Ok(FbxModel {
        root,
        player: AnimationPlayer::new(clips),
    })
}

fn extract_primitives(scene: &ufbx::Scene) -> Result<Vec<CpuPrimitive>, String> {
    let node_indices = node_index_map(scene);
    let mut result = Vec::new();
    for (node_index, node) in scene.nodes.iter().enumerate() {
        let Some(mesh) = node.mesh.as_deref() else {
            continue;
        };
        if mesh.num_triangles == 0 {
            continue;
        }

        let skin_deformer = mesh.skin_deformers.get(0).map(|skin| &**skin);
        if mesh.skin_deformers.len() > 1 {
            log::warn!(
                "fbx: mesh '{}' has multiple skin deformers; using the first",
                mesh.element.name
            );
        }
        let skin_template = skin_deformer
            .map(|skin| skin_template(skin, &node_indices))
            .transpose()?;
        let geometry_matrix = if skin_template.is_some() {
            Mat4::IDENTITY
        } else {
            mat4(&node.geometry_to_node)?
        };
        let normal_matrix = normal_matrix_from_mat4(geometry_matrix)?;
        let material_count = node.materials.len().max(mesh.materials.len()).max(1);
        let mut by_material: Vec<CpuPrimitive> = (0..material_count)
            .map(|material_index| CpuPrimitive {
                node_index,
                positions: Vec::new(),
                faces: Vec::new(),
                normals: mesh.vertex_normal.exists.then(Vec::new),
                uvs: mesh.vertex_uv.exists.then(Vec::new),
                material: node
                    .materials
                    .get(material_index)
                    .or_else(|| mesh.materials.get(material_index))
                    .map(|material| material_factors(material))
                    .unwrap_or_default(),
                skin: skin_template.as_ref().map(|template| CpuSkin {
                    joints: template.joints.clone(),
                    inverse_bind: template.inverse_bind.clone(),
                    vertex_joints: Vec::new(),
                    vertex_weights: Vec::new(),
                }),
            })
            .collect();
        let mut triangle_indices = Vec::new();

        for (face_index, face) in mesh.faces.iter().copied().enumerate() {
            if face.num_indices < 3 {
                continue;
            }
            ufbx::triangulate_face_vec(&mut triangle_indices, mesh, face);
            let material_index = mesh.face_material.get(face_index).copied().unwrap_or(0) as usize;
            let primitive = by_material.get_mut(material_index).ok_or_else(|| {
                format!(
                    "FBX mesh '{}' face {} references missing material {}",
                    mesh.element.name, face_index, material_index
                )
            })?;

            for triangle in triangle_indices.chunks_exact(3) {
                let base = u32::try_from(primitive.positions.len())
                    .map_err(|_| "FBX primitive exceeds u32 vertex indexing".to_string())?;
                let corner_order = if mesh.reversed_winding {
                    [triangle[0], triangle[2], triangle[1]]
                } else {
                    [triangle[0], triangle[1], triangle[2]]
                };
                for polygon_vertex in corner_order {
                    append_vertex(
                        primitive,
                        mesh,
                        skin_deformer,
                        skin_template.as_ref(),
                        geometry_matrix,
                        normal_matrix,
                        polygon_vertex,
                    )?;
                }
                primitive.faces.push([base, base + 1, base + 2]);
            }
        }

        result.extend(
            by_material
                .into_iter()
                .filter(|primitive| !primitive.faces.is_empty()),
        );
    }
    Ok(result)
}

fn append_vertex(
    primitive: &mut CpuPrimitive,
    mesh: &ufbx::Mesh,
    skin_deformer: Option<&ufbx::SkinDeformer>,
    skin_template: Option<&SkinTemplate>,
    geometry_matrix: Mat4,
    normal_matrix: Mat3,
    polygon_vertex: u32,
) -> Result<(), String> {
    let index = usize::try_from(polygon_vertex)
        .map_err(|_| "FBX polygon vertex index overflow".to_string())?;
    validate_vec3_attribute(&mesh.vertex_position, index).ok_or_else(|| {
        format!(
            "FBX mesh '{}' polygon vertex {} has no position",
            mesh.element.name, index
        )
    })?;
    let position = geometry_matrix.transform_point3(vec3(mesh.vertex_position[index])?);
    if !position.is_finite() {
        return Err(format!(
            "FBX mesh '{}' has a non-finite position",
            mesh.element.name
        ));
    }
    primitive.positions.push(position);

    if let Some(normals) = primitive.normals.as_mut() {
        validate_vec3_attribute(&mesh.vertex_normal, index).ok_or_else(|| {
            format!(
                "FBX mesh '{}' polygon vertex {} has no normal",
                mesh.element.name, index
            )
        })?;
        let normal = normal_matrix * vec3(mesh.vertex_normal[index])?;
        if !normal.is_finite() || normal.length_squared() <= f32::EPSILON {
            return Err(format!(
                "FBX mesh '{}' has an invalid transformed normal",
                mesh.element.name
            ));
        }
        normals.push(normal.normalize());
    }

    if let Some(uvs) = primitive.uvs.as_mut() {
        validate_vec2_attribute(&mesh.vertex_uv, index).ok_or_else(|| {
            format!(
                "FBX mesh '{}' polygon vertex {} has no UV",
                mesh.element.name, index
            )
        })?;
        let source_uv = mesh.vertex_uv[index];
        let uv = Vec2::new(to_f32(source_uv.x)?, to_f32(source_uv.y)?);
        if !uv.is_finite() {
            return Err(format!(
                "FBX mesh '{}' has a non-finite UV",
                mesh.element.name
            ));
        }
        uvs.push(uv);
    }

    if let (Some(skin), Some(template), Some(cpu_skin)) =
        (skin_deformer, skin_template, primitive.skin.as_mut())
    {
        let control_point = *mesh.vertex_indices.get(index).ok_or_else(|| {
            format!(
                "FBX mesh '{}' has no control-point index",
                mesh.element.name
            )
        })? as usize;
        let (joints, weights) = skin_vertex(skin, template, control_point)?;
        cpu_skin.vertex_joints.push(joints);
        cpu_skin.vertex_weights.push(weights);
    }
    Ok(())
}

fn skin_template(
    skin: &ufbx::SkinDeformer,
    node_indices: &HashMap<u32, usize>,
) -> Result<SkinTemplate, String> {
    let mut cluster_to_joint = vec![None; skin.clusters.len()];
    let mut joints = Vec::new();
    let mut inverse_bind = Vec::new();
    for (cluster_index, cluster) in skin.clusters.iter().enumerate() {
        let Some(bone) = cluster.bone_node.as_deref() else {
            continue;
        };
        let node_index = *node_indices.get(&bone.element.element_id).ok_or_else(|| {
            format!(
                "FBX skin bone '{}' is not in the node table",
                bone.element.name
            )
        })?;
        let joint_index =
            u32::try_from(joints.len()).map_err(|_| "FBX skin has too many joints".to_string())?;
        cluster_to_joint[cluster_index] = Some(joint_index);
        joints.push(node_index);
        inverse_bind.push(mat4(&cluster.geometry_to_bone)?);
    }
    if joints.is_empty() {
        return Err("FBX skin contains no valid bone clusters".to_string());
    }
    Ok(SkinTemplate {
        cluster_to_joint,
        joints,
        inverse_bind,
    })
}

fn skin_vertex(
    skin: &ufbx::SkinDeformer,
    template: &SkinTemplate,
    control_point: usize,
) -> Result<([u32; 4], [f32; 4]), String> {
    let vertex = skin
        .vertices
        .get(control_point)
        .ok_or_else(|| format!("FBX skin has no weights for vertex {control_point}"))?;
    let begin = vertex.weight_begin as usize;
    let end = begin
        .checked_add(vertex.num_weights as usize)
        .ok_or_else(|| "FBX skin weight range overflow".to_string())?;
    let source = skin
        .weights
        .get(begin..end)
        .ok_or_else(|| "FBX skin weight range is out of bounds".to_string())?;
    let mut influences: Vec<(u32, f32)> = source
        .iter()
        .filter_map(|weight| {
            let joint = template
                .cluster_to_joint
                .get(weight.cluster_index as usize)
                .copied()
                .flatten()?;
            let value = to_f32(weight.weight).ok()?;
            (value.is_finite() && value > 0.0).then_some((joint, value))
        })
        .collect();
    influences.sort_by(|left, right| right.1.total_cmp(&left.1));
    influences.truncate(4);
    let sum: f32 = influences.iter().map(|(_, weight)| *weight).sum();
    if !sum.is_finite() || sum <= f32::EPSILON {
        return Ok(([0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]));
    }
    let mut joints = [0u32; 4];
    let mut weights = [0.0f32; 4];
    for (index, (joint, weight)) in influences.into_iter().enumerate() {
        joints[index] = joint;
        weights[index] = weight / sum;
    }
    Ok((joints, weights))
}

fn build_animation_clips(
    scene: &ufbx::Scene,
    node_map: &[SceneNode3d],
) -> Result<Vec<AnimationClip>, String> {
    let layouts = animation_layouts(scene)?;
    let mut clips = Vec::with_capacity(layouts.len());
    for (clip_index, layout) in layouts.into_iter().enumerate() {
        let stack = scene
            .anim_stacks
            .get(clip_index)
            .ok_or_else(|| "FBX animation stack disappeared".to_string())?;
        let mut channels = Vec::with_capacity(layout.nodes.len() * 3);
        for node_index in layout.nodes {
            let node = &scene.nodes[node_index];
            let mut translations = Vec::with_capacity(layout.source_times.len());
            let mut rotations = Vec::with_capacity(layout.source_times.len());
            let mut scales = Vec::with_capacity(layout.source_times.len());
            for &source_time in &layout.source_times {
                let transform = ufbx::evaluate_transform(&stack.anim, node, source_time);
                let (translation, rotation, scale) = transform_parts(transform)?;
                translations.push(translation);
                rotations.push(rotation);
                scales.push(scale);
            }
            channels.push(AnimationChannel::translation(
                node_map[node_index].clone(),
                layout.key_times.clone(),
                translations,
                SceneInterpolation::Linear,
            ));
            channels.push(AnimationChannel::rotation(
                node_map[node_index].clone(),
                layout.key_times.clone(),
                rotations,
                SceneInterpolation::Linear,
            ));
            channels.push(AnimationChannel::scale(
                node_map[node_index].clone(),
                layout.key_times.clone(),
                scales,
                SceneInterpolation::Linear,
            ));
        }
        clips.push(AnimationClip::new(layout.label, channels));
    }
    Ok(clips)
}

fn animation_layouts(scene: &ufbx::Scene) -> Result<Vec<ClipLayout>, String> {
    let node_indices = node_index_map(scene);
    let fps =
        if scene.settings.frames_per_second.is_finite() && scene.settings.frames_per_second > 0.0 {
            scene.settings.frames_per_second.min(MAX_ANIMATION_FPS)
        } else {
            DEFAULT_ANIMATION_FPS
        };
    let mut layouts = Vec::with_capacity(scene.anim_stacks.len());
    for (clip_index, stack) in scene.anim_stacks.iter().enumerate() {
        if !stack.time_begin.is_finite() || !stack.time_end.is_finite() {
            return Err(format!(
                "FBX animation stack {clip_index} has a non-finite range"
            ));
        }
        let duration_f64 = (stack.time_end - stack.time_begin).max(0.0);
        let duration = to_f32(duration_f64)?;
        let frame_count = (duration_f64 * fps).ceil() as usize;
        let mut source_times = Vec::with_capacity(frame_count.saturating_add(1));
        let mut key_times: Vec<f32> = Vec::with_capacity(frame_count.saturating_add(1));
        for frame in 0..=frame_count {
            let relative = ((frame as f64) / fps).min(duration_f64);
            if key_times
                .last()
                .is_some_and(|last| (*last - relative as f32).abs() <= f32::EPSILON)
            {
                continue;
            }
            source_times.push(stack.time_begin + relative);
            key_times.push(relative as f32);
        }
        if source_times.is_empty() {
            source_times.push(stack.time_begin);
            key_times.push(0.0);
        }

        let animated_ids: HashSet<u32> = stack
            .layers
            .iter()
            .flat_map(|layer| layer.anim_props.iter())
            .filter(|prop| {
                matches!(
                    prop.prop_name.as_ref(),
                    "Lcl Translation" | "Lcl Rotation" | "Lcl Scaling"
                )
            })
            .map(|prop| prop.element.element_id)
            .collect();
        let mut nodes: Vec<usize> = animated_ids
            .into_iter()
            .filter_map(|element_id| node_indices.get(&element_id).copied())
            .collect();
        nodes.sort_unstable();
        let sample_count = nodes
            .len()
            .checked_mul(source_times.len())
            .ok_or_else(|| "FBX animation sample count overflow".to_string())?;
        if sample_count > MAX_BAKED_NODE_SAMPLES {
            return Err(format!(
                "FBX animation stack {clip_index} requires {sample_count} baked node samples (limit {MAX_BAKED_NODE_SAMPLES})"
            ));
        }
        let label = if stack.element.name.trim().is_empty() {
            format!("Animation {}", clip_index + 1)
        } else {
            stack.element.name.to_string()
        };
        layouts.push(ClipLayout {
            label,
            duration,
            source_times,
            key_times,
            nodes,
        });
    }
    Ok(layouts)
}

fn node_index_map(scene: &ufbx::Scene) -> HashMap<u32, usize> {
    scene
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.element.element_id, index))
        .collect()
}

fn node_label(node: &ufbx::Node, index: usize) -> String {
    if node.element.name.trim().is_empty() {
        format!("Node {}", index + 1)
    } else {
        node.element.name.to_string()
    }
}

fn transform_parts(transform: ufbx::Transform) -> Result<(Vec3, Quat, Vec3), String> {
    let translation = vec3(transform.translation)?;
    let scale = vec3(transform.scale)?;
    let rotation = Quat::from_xyzw(
        to_f32(transform.rotation.x)?,
        to_f32(transform.rotation.y)?,
        to_f32(transform.rotation.z)?,
        to_f32(transform.rotation.w)?,
    );
    if !rotation.is_finite() || rotation.length_squared() <= f32::EPSILON {
        return Err("FBX transform has an invalid rotation".to_string());
    }
    Ok((translation, rotation.normalize(), scale))
}

fn material_factors(material: &ufbx::Material) -> MaterialFactors {
    let defaults = MaterialFactors::default();
    let (color_map, factor_map) = if material.pbr.base_color.has_value {
        (&material.pbr.base_color, &material.pbr.base_factor)
    } else {
        (&material.fbx.diffuse_color, &material.fbx.diffuse_factor)
    };
    let color = color_map.value_vec4;
    let factor = scalar_or(factor_map, 1.0).max(0.0);
    let base_color = if color_map.has_value {
        [
            finite_or(color.x, defaults.base_color[0]) * factor,
            finite_or(color.y, defaults.base_color[1]) * factor,
            finite_or(color.z, defaults.base_color[2]) * factor,
            finite_or(color.w, defaults.base_color[3]),
        ]
    } else {
        defaults.base_color
    };
    let metallic = scalar_or(&material.pbr.metalness, defaults.metallic).clamp(0.0, 1.0);
    let roughness = scalar_or(&material.pbr.roughness, defaults.roughness).clamp(0.0, 1.0);

    if !material.textures.is_empty() {
        log::warn!(
            "fbx: material '{}' textures are not supported; using scalar factors only",
            material.element.name
        );
    }
    MaterialFactors {
        base_color,
        metallic,
        roughness,
    }
}

fn scalar_or(map: &ufbx::MaterialMap, fallback: f32) -> f32 {
    if map.has_value {
        finite_or(map.value_vec4.x, fallback)
    } else {
        fallback
    }
}

fn finite_or(value: f64, fallback: f32) -> f32 {
    to_f32(value).unwrap_or(fallback)
}

fn validate_vec3_attribute(attribute: &ufbx::VertexVec3, index: usize) -> Option<()> {
    let value_index = *attribute.indices.get(index)? as usize;
    attribute.values.get(value_index).map(|_| ())
}

fn validate_vec2_attribute(attribute: &ufbx::VertexVec2, index: usize) -> Option<()> {
    let value_index = *attribute.indices.get(index)? as usize;
    attribute.values.get(value_index).map(|_| ())
}

fn format_ufbx_error(error: ufbx::Error) -> String {
    let info = error.info();
    if info.is_empty() {
        format!("{:?}: {}", error.type_, error.description)
    } else {
        format!("{:?}: {} ({info})", error.type_, error.description)
    }
}

fn vec3(value: ufbx::Vec3) -> Result<Vec3, String> {
    let result = Vec3::new(to_f32(value.x)?, to_f32(value.y)?, to_f32(value.z)?);
    result
        .is_finite()
        .then_some(result)
        .ok_or_else(|| "FBX vector is not finite".to_string())
}

fn mat4(matrix: &ufbx::Matrix) -> Result<Mat4, String> {
    let result = Mat4::from_cols_array(&[
        to_f32(matrix.m00)?,
        to_f32(matrix.m10)?,
        to_f32(matrix.m20)?,
        0.0,
        to_f32(matrix.m01)?,
        to_f32(matrix.m11)?,
        to_f32(matrix.m21)?,
        0.0,
        to_f32(matrix.m02)?,
        to_f32(matrix.m12)?,
        to_f32(matrix.m22)?,
        0.0,
        to_f32(matrix.m03)?,
        to_f32(matrix.m13)?,
        to_f32(matrix.m23)?,
        1.0,
    ]);
    result
        .is_finite()
        .then_some(result)
        .ok_or_else(|| "FBX matrix is not finite".to_string())
}

fn normal_matrix_from_mat4(matrix: Mat4) -> Result<Mat3, String> {
    let linear = Mat3::from_mat4(matrix);
    let determinant = linear.determinant();
    if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return Err("FBX geometry transform is singular".to_string());
    }
    Ok(linear.inverse().transpose())
}

fn to_f32(value: f64) -> Result<f32, String> {
    let converted = value as f32;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err("FBX numeric value overflows f32".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE_FBX: &[u8] = br#"; FBX 7.4.0 project file
FBXHeaderExtension: { FBXHeaderVersion: 1003 FBXVersion: 7400 }
GlobalSettings: { Version: 1000 Properties70: {
P: "UpAxis", "int", "Integer", "",1
P: "FrontAxis", "int", "Integer", "",2
P: "CoordAxis", "int", "Integer", "",0
P: "UnitScaleFactor", "double", "Number", "",1
} }
Documents: { Count: 1 Document: 1, "", "Scene" { RootNode: 0 } }
Definitions: { Version: 100 Count: 2
ObjectType: "Model" { Count: 1 }
ObjectType: "Geometry" { Count: 1 }
}
Objects: {
Geometry: 101, "Geometry::Triangle", "Mesh" {
Vertices: *9 { a: 0,0,0,100,0,0,0,100,0 }
PolygonVertexIndex: *3 { a: 0,1,-3 }
LayerElementNormal: 0 { Version: 101 MappingInformationType: "ByPolygonVertex" ReferenceInformationType: "Direct" Normals: *3 { a: 0,0,1 } }
Layer: 0 { Version: 100 LayerElement: { Type: "LayerElementNormal" TypedIndex: 0 } }
}
Model: 100, "Model::Triangle", "Mesh" { Version: 232 }
}
Connections: { C: "OO",100,0 C: "OO",101,100 }
"#;

    #[test]
    fn extracts_static_triangle_in_meter_space() {
        let scene = ufbx::load_memory(TRIANGLE_FBX, load_options(true)).unwrap();
        let primitives = extract_primitives(&scene).unwrap();
        assert_eq!(primitives.len(), 1);
        assert_eq!(primitives[0].faces, vec![[0, 1, 2]]);
        assert_eq!(primitives[0].positions.len(), 3);
        assert!((primitives[0].positions[1].x - 1.0).abs() < 1e-5);
        assert!((primitives[0].positions[2].y - 1.0).abs() < 1e-5);
        assert_eq!(primitives[0].normals.as_ref().unwrap().len(), 3);
        assert!(primitives[0].skin.is_none());
    }

    #[test]
    fn static_triangle_has_no_animation_layouts() {
        let scene = ufbx::load_memory(TRIANGLE_FBX, load_options(true)).unwrap();
        assert!(animation_layouts(&scene).unwrap().is_empty());
    }

    #[test]
    fn rejects_non_fbx_memory() {
        assert!(ufbx::load_memory(b"not an fbx", load_options(true)).is_err());
    }
}
