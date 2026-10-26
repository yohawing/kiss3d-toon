//! glTF / GLB loader.
//!
//! Loads a glTF 2.0 asset (`.gltf` + external buffers/images, or self-contained
//! `.glb`) into a kiss3d [`GltfModel`]: a scene-graph subtree mirroring the glTF
//! node hierarchy, plus an [`AnimationPlayer`] holding every animation in the
//! file. Meshes become regular scene nodes; PBR materials map onto kiss3d's
//! metallic-roughness [`Object3d`] surface; skinned meshes carry a [`Skin3d`]
//! that drives GPU vertex skinning.
//!
//! Use [`SceneNode3d::add_gltf`](crate::scene::SceneNode3d::add_gltf) rather than
//! calling [`load`] directly in most cases.

use crate::color::Color;
use crate::loader::obj::bounded_warning_text;
use crate::resource::{
    GpuMesh3d, MaterialManager3d, MorphTargets, SkinVertexData, Texture, TextureManager,
};
use crate::scene::{
    AlphaMode, AnimationChannel, AnimationClip, AnimationPlayer, GltfModel, Interpolation,
    Object3d, SceneNode3d, SkeletonNode3d, Skin3d,
};
use glamx::{Mat4, Pose3, Quat, Vec2, Vec3};
use image::{DynamicImage, ImageFormat, ImageReader};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fs::File;
use std::io::{Cursor, Error as IoError, ErrorKind, Read};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

const MAX_GLTF_IMAGE_DIMENSION: u32 = 16_384;
const MAX_GLTF_IMAGE_DECODED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_GLTF_DECODED_IMAGE_BYTES: u64 = 1024 * 1024 * 1024;

/// Limits applied before parsing a CPU glTF decode source.
///
/// The struct is deliberately non-exhaustive so additional source, document,
/// or CPU-template budgets can be added without breaking callers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GltfDecodeLimits {
    /// Maximum source size accepted by either path or in-memory decode.
    pub max_source_bytes: u64,
    /// Maximum JSON document or GLB JSON chunk size.
    pub max_json_bytes: u64,
    /// Maximum aggregate declared byte length across all glTF buffers.
    pub max_declared_buffer_bytes: u64,
    /// Maximum number of images in the parsed document.
    pub max_images: u64,
    /// Maximum aggregate element count across all accessors.
    pub max_accessor_elements: u64,
    /// Maximum aggregate decoded accessor byte size.
    pub max_accessor_bytes: u64,
    /// Maximum aggregate bytes read from external buffer and image URIs.
    pub max_external_resource_bytes: u64,
    /// Maximum number of nodes materialized in the CPU template.
    pub max_nodes: u64,
    /// Maximum number of skins materialized in the CPU template.
    pub max_skins: u64,
    /// Maximum aggregate skin joint references materialized in the CPU template.
    /// This counts each `CpuSkin` definition's joints, rather than deduplicating
    /// the runtime skeleton nodes.
    pub max_skin_joint_references: u64,
    /// Maximum number of meshes materialized in the CPU template.
    pub max_meshes: u64,
    /// Maximum aggregate mesh primitives materialized in the CPU template.
    pub max_primitives: u64,
    /// Maximum number of materials materialized in the CPU template.
    pub max_materials: u64,
    /// Maximum number of animation clips materialized in the CPU template.
    pub max_animations: u64,
    /// Maximum aggregate animation channels materialized in the CPU template.
    pub max_animation_channels: u64,
    /// Maximum aggregate animation input values and, independently, output
    /// values materialized in the CPU template.
    pub max_animation_key_values: u64,
}

impl Default for GltfDecodeLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 1024 * 1024 * 1024,
            max_json_bytes: 64 * 1024 * 1024,
            max_declared_buffer_bytes: 1024 * 1024 * 1024,
            max_images: 256,
            max_accessor_elements: 100_000_000,
            max_accessor_bytes: 1024 * 1024 * 1024,
            max_external_resource_bytes: 1024 * 1024 * 1024,
            max_nodes: 262_144,
            max_skins: 65_536,
            max_skin_joint_references: 262_144,
            max_meshes: 262_144,
            max_primitives: 262_144,
            max_materials: 262_144,
            max_animations: 64,
            max_animation_channels: 65_536,
            max_animation_key_values: 1_000_000,
        }
    }
}

/// Validates the number of animation output elements for one glTF channel.
/// Vector and quaternion channels use one element per key; morph channels use
/// `components_per_key` scalar weights per key. Cubic-spline channels store
/// in-tangent, value, and out-tangent for each key.
pub fn validate_animation_output_cardinality(
    key_count: usize,
    cubic_spline: bool,
    output_count: usize,
    components_per_key: usize,
) -> Result<(), String> {
    let values_per_key = if cubic_spline { 3 } else { 1 };
    let expected = key_count
        .checked_mul(values_per_key)
        .and_then(|count| count.checked_mul(components_per_key))
        .ok_or_else(|| "glTF animation output cardinality overflowed".to_string())?;
    if output_count != expected {
        return Err(format!(
            "glTF animation output cardinality is {output_count}; expected {expected}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DecodedImageBudget {
    total_bytes: u64,
    max_total_bytes: u64,
}

impl Default for DecodedImageBudget {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            max_total_bytes: MAX_GLTF_DECODED_IMAGE_BYTES,
        }
    }
}

impl DecodedImageBudget {
    fn limits(&self) -> Result<image::Limits, gltf::Error> {
        let remaining = self.max_total_bytes.saturating_sub(self.total_bytes);
        if remaining == 0 {
            return Err(decoded_image_budget_error());
        }
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_GLTF_IMAGE_DIMENSION);
        limits.max_image_height = Some(MAX_GLTF_IMAGE_DIMENSION);
        limits.max_alloc = Some(remaining.min(MAX_GLTF_IMAGE_DECODED_BYTES));
        Ok(limits)
    }

    fn reserve(&mut self, bytes: usize) -> Result<(), gltf::Error> {
        let bytes = u64::try_from(bytes).map_err(|_| decoded_image_budget_error())?;
        let next = self
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(decoded_image_budget_error)?;
        if bytes > MAX_GLTF_IMAGE_DECODED_BYTES || next > self.max_total_bytes {
            return Err(decoded_image_budget_error());
        }
        self.total_bytes = next;
        Ok(())
    }
}

fn decoded_image_budget_error() -> gltf::Error {
    gltf::Error::Io(IoError::new(
        ErrorKind::InvalidData,
        "glTF decoded image budget exceeded",
    ))
}

#[derive(Clone, Copy, Debug)]
struct EncodedResourceBudget {
    total_bytes: u64,
    max_total_bytes: u64,
}

impl EncodedResourceBudget {
    fn new(max_total_bytes: u64) -> Self {
        Self {
            total_bytes: 0,
            max_total_bytes,
        }
    }

    fn remaining(&self) -> u64 {
        self.max_total_bytes.saturating_sub(self.total_bytes)
    }

    fn ensure_bound(&self, bytes: u64) -> Result<(), gltf::Error> {
        let next = self
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(|| encoded_resource_budget_error(u64::MAX, self.max_total_bytes))?;
        if next > self.max_total_bytes {
            return Err(encoded_resource_budget_error(next, self.max_total_bytes));
        }
        Ok(())
    }

    fn reserve(&mut self, bytes: usize) -> Result<(), gltf::Error> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| encoded_resource_budget_error(u64::MAX, self.max_total_bytes))?;
        self.ensure_bound(bytes)?;
        self.total_bytes = self
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(|| encoded_resource_budget_error(u64::MAX, self.max_total_bytes))?;
        Ok(())
    }
}

fn encoded_resource_budget_error(actual: u64, limit: u64) -> gltf::Error {
    gltf::Error::Io(IoError::new(
        ErrorKind::InvalidData,
        format!("glTF external encoded resource bytes total is {actual}; limit is {limit}"),
    ))
}

/// One renderable primitive of a glTF mesh: a shareable GPU mesh plus the index
/// of the material that shades it.
struct PrimInfo {
    mesh: Rc<RefCell<GpuMesh3d>>,
    material_index: Option<usize>,
}

/// CPU-owned geometry extracted from one triangle primitive.
///
/// This intentionally contains only plain vectors and arrays.  It is a small
/// first step toward decoding glTF off the event-loop thread; GPU/scene handles
/// remain in [`build_model`].
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuPrimitive {
    positions: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    uvs: Option<Vec<[f32; 2]>>,
    material_index: Option<usize>,
    skin_joints: Option<Vec<[u32; 4]>>,
    skin_weights: Option<Vec<[f32; 4]>>,
    morph: Option<CpuMorphTargets>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuMesh {
    primitives: Vec<CpuPrimitive>,
    weights: Vec<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuNode {
    parent: Option<usize>,
    children: Vec<usize>,
    mesh: Option<usize>,
    skin: Option<usize>,
    weights: Vec<f32>,
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuImage {
    decoded: DynamicImage,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuMaterial {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    extensions: MaterialExtensionFactors,
    /// Runtime alpha mode, including the normalized MASK cutoff when applicable.
    alpha_mode: AlphaMode,
    /// The effective MASK cutoff; other alpha modes ignore this glTF property.
    alpha_cutoff: Option<f32>,
    base_color_image: Option<usize>,
    metallic_roughness_image: Option<usize>,
    normal_image: Option<usize>,
    occlusion_image: Option<usize>,
    emissive_image: Option<usize>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuMorphTargets {
    num_targets: usize,
    positions: Vec<[f32; 4]>,
    normals: Option<Vec<[f32; 4]>>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuSkin {
    joints: Vec<usize>,
    inverse_bind: Vec<[[f32; 4]; 4]>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum CpuAnimationOutput {
    Translation(Vec<[f32; 3]>),
    Scale(Vec<[f32; 3]>),
    Rotation(Vec<[f32; 4]>),
    MorphWeights {
        values: Vec<f32>,
        num_targets: usize,
    },
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuAnimationChannel {
    target_node: usize,
    times: Vec<f32>,
    interpolation: Interpolation,
    output: CpuAnimationOutput,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CpuAnimationClip {
    name: String,
    channels: Vec<CpuAnimationChannel>,
}

/// Owned CPU-side glTF data for a future worker-thread decode/instantiate path.
///
/// Fields remain private until GPU instantiation is exposed; callers should use
/// [`load`] or [`load_from_slice`] for a renderable model today.
#[derive(Clone, Debug)]
pub struct GltfAssetTemplate {
    #[allow(dead_code)]
    nodes: Vec<CpuNode>,
    #[allow(dead_code)]
    meshes: Vec<CpuMesh>,
    #[allow(dead_code)]
    default_scene_roots: Vec<usize>,
    #[allow(dead_code)]
    images: Vec<CpuImage>,
    #[allow(dead_code)]
    materials: Vec<CpuMaterial>,
    #[allow(dead_code)]
    skins: Vec<CpuSkin>,
    #[allow(dead_code)]
    animations: Vec<CpuAnimationClip>,
}

/// Loads a glTF/GLB file into a [`GltfModel`].
///
/// The returned model's `root` is an unrooted group node holding the file's
/// default scene; attach it to your scene graph (or use
/// [`SceneNode3d::add_gltf`](crate::scene::SceneNode3d::add_gltf), which does that
/// for you). `player` is stopped initially — call
/// [`AnimationPlayer::play`](crate::scene::AnimationPlayer::play) to start one.
/// Malformed scene graphs and out-of-range geometry/animation references return
/// [`gltf::Error`] rather than being accepted as a partial model.
pub fn load(path: &Path) -> Result<GltfModel, gltf::Error> {
    let gltf = gltf::Gltf::open(path)?;
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let gltf::Gltf { document, blob } = gltf;
    let buffers = gltf::import_buffers(&document, Some(base), blob)?;
    catch_unwind(AssertUnwindSafe(|| {
        let images = import_images_with_budget(&document, Some(base), &buffers)?;
        build_model(document, buffers, images, &path.to_string_lossy())
    }))
    .map_err(|_| loader_panic_error())?
}

/// Loads a glTF/GLB from an in-memory byte slice — e.g. an `include_bytes!`-embedded,
/// self-contained `.glb` — for targets without a filesystem such as wasm. See
/// [`load`] and [`SceneNode3d::add_gltf_from_memory`](crate::scene::SceneNode3d::add_gltf_from_memory).
pub fn load_from_slice(bytes: &[u8]) -> Result<GltfModel, gltf::Error> {
    let gltf = gltf::Gltf::from_slice(bytes)?;
    let gltf::Gltf { document, blob } = gltf;
    let buffers = gltf::import_buffers(&document, None, blob)?;
    catch_unwind(AssertUnwindSafe(|| {
        let images = import_images_with_budget(&document, None, &buffers)?;
        build_model(document, buffers, images, "gltf_mem")
    }))
    .map_err(|_| loader_panic_error())?
}

/// Decodes a glTF/GLB into an owned CPU template without creating GPU resources.
/// The template is not yet consumed by [`load`]; it is the safe hand-off seam
/// for a future worker-thread decode path.
/// Image dimensions and aggregate decoded image bytes are bounded before the
/// decoded images are retained in the template.
/// The default source and JSON limits match the renderer's current parser
/// preflight. External buffer and image URI reads share the configured
/// `max_external_resource_bytes` budget.
pub fn decode_cpu(path: &Path) -> Result<GltfAssetTemplate, gltf::Error> {
    decode_cpu_with_limits(path, &GltfDecodeLimits::default())
}

/// Decodes a glTF/GLB into an owned CPU template with explicit source limits.
pub fn decode_cpu_with_limits(
    path: &Path,
    limits: &GltfDecodeLimits,
) -> Result<GltfAssetTemplate, gltf::Error> {
    let source = read_path_source_with_limits(path, limits)?;
    validate_slice_source_limits(&source, limits)?;
    let gltf = gltf::Gltf::from_slice(&source)?;
    validate_document_decode_limits(&gltf.document, limits)?;
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let gltf::Gltf { document, blob } = gltf;
    let mut resource_budget = EncodedResourceBudget::new(limits.max_external_resource_bytes);
    let buffers = import_buffers_with_budget(&document, Some(base), blob, &mut resource_budget)?;
    catch_unwind(AssertUnwindSafe(|| {
        let images = import_images_with_encoded_budget(
            &document,
            Some(base),
            &buffers,
            &mut resource_budget,
        )?;
        decode_cpu_internal(document, buffers, images)
    }))
    .map_err(|_| loader_panic_error())?
}

/// Decodes an in-memory glTF/GLB into an owned CPU template without GPU work.
/// The returned template is opaque until a future CPU-to-GPU instantiate API is
/// added; malformed input is reported as [`gltf::Error`]. Image dimensions and
/// aggregate decoded image bytes are bounded, and the default source/JSON
/// and document limits match the renderer's current parser preflight.
/// External buffer and image URI reads share the configured
/// `max_external_resource_bytes` budget.
pub fn decode_cpu_from_slice(bytes: &[u8]) -> Result<GltfAssetTemplate, gltf::Error> {
    decode_cpu_from_slice_with_limits(bytes, &GltfDecodeLimits::default())
}

/// Decodes an in-memory glTF/GLB into an owned CPU template with explicit source limits.
pub fn decode_cpu_from_slice_with_limits(
    bytes: &[u8],
    limits: &GltfDecodeLimits,
) -> Result<GltfAssetTemplate, gltf::Error> {
    validate_slice_source_limits(bytes, limits)?;
    let gltf = gltf::Gltf::from_slice(bytes)?;
    validate_document_decode_limits(&gltf.document, limits)?;
    let gltf::Gltf { document, blob } = gltf;
    let mut resource_budget = EncodedResourceBudget::new(limits.max_external_resource_bytes);
    let buffers = import_buffers_with_budget(&document, None, blob, &mut resource_budget)?;
    catch_unwind(AssertUnwindSafe(|| {
        let images =
            import_images_with_encoded_budget(&document, None, &buffers, &mut resource_budget)?;
        decode_cpu_internal(document, buffers, images)
    }))
    .map_err(|_| loader_panic_error())?
}

fn loader_panic_error() -> gltf::Error {
    gltf::Error::Io(IoError::new(
        ErrorKind::InvalidData,
        "glTF loader rejected malformed document",
    ))
}

fn invalid_decode_source_error(message: impl Into<String>) -> gltf::Error {
    gltf::Error::Io(IoError::new(ErrorKind::InvalidData, message.into()))
}

fn read_path_source_with_limits(
    path: &Path,
    limits: &GltfDecodeLimits,
) -> Result<Vec<u8>, gltf::Error> {
    let file = File::open(path).map_err(gltf::Error::Io)?;
    let read_limit = limits
        .max_source_bytes
        .checked_add(1)
        .ok_or_else(|| invalid_decode_source_error("glTF source limit overflowed"))?;
    let mut source = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut source)
        .map_err(gltf::Error::Io)?;
    let source_bytes = u64::try_from(source.len())
        .map_err(|_| invalid_decode_source_error("glTF source length does not fit u64"))?;
    validate_source_byte_limit(source_bytes, limits)?;
    Ok(source)
}

fn validate_slice_source_limits(
    bytes: &[u8],
    limits: &GltfDecodeLimits,
) -> Result<(), gltf::Error> {
    let source_bytes = u64::try_from(bytes.len())
        .map_err(|_| invalid_decode_source_error("glTF source length does not fit u64"))?;
    validate_source_byte_limit(source_bytes, limits)?;

    if bytes.len() >= 4 && bytes[..4] == *b"glTF" {
        if bytes.len() < 12 {
            return Err(invalid_decode_source_error("glTF GLB header is truncated"));
        }
        let header_tail: [u8; 8] = bytes[4..12].try_into().unwrap();
        let declared_length = validate_glb_header_limits(source_bytes, &header_tail, limits)?;
        if bytes.len() < 20 {
            return Err(invalid_decode_source_error(
                "glTF GLB JSON chunk header is truncated",
            ));
        }
        let chunk_header: [u8; 8] = bytes[12..20].try_into().unwrap();
        validate_glb_json_chunk_limits(declared_length, &chunk_header, limits)?;
    } else {
        validate_json_byte_limit(source_bytes, limits)?;
    }
    Ok(())
}

fn validate_document_decode_limits(
    document: &gltf::Document,
    limits: &GltfDecodeLimits,
) -> Result<(), gltf::Error> {
    let mut declared_buffer_bytes = 0u64;
    for buffer in document.buffers() {
        let length = u64::try_from(buffer.length()).map_err(|_| {
            invalid_document_limit_overflow(
                "declared buffer bytes",
                limits.max_declared_buffer_bytes,
            )
        })?;
        declared_buffer_bytes = declared_buffer_bytes.checked_add(length).ok_or_else(|| {
            invalid_document_limit_overflow(
                "declared buffer bytes",
                limits.max_declared_buffer_bytes,
            )
        })?;
    }
    if declared_buffer_bytes > limits.max_declared_buffer_bytes {
        return Err(invalid_document_limit_error(
            "declared buffer bytes",
            declared_buffer_bytes,
            limits.max_declared_buffer_bytes,
        ));
    }

    let image_count = u64::try_from(document.images().count())
        .map_err(|_| invalid_document_limit_overflow("image count", limits.max_images))?;
    if image_count > limits.max_images {
        return Err(invalid_document_limit_error(
            "image count",
            image_count,
            limits.max_images,
        ));
    }

    let mut accessor_elements = 0u64;
    let mut accessor_bytes = 0u64;
    for accessor in document.accessors() {
        let elements = u64::try_from(accessor.count()).map_err(|_| {
            invalid_document_limit_overflow("accessor elements", limits.max_accessor_elements)
        })?;
        let bytes_per_element = u64::try_from(accessor.size()).map_err(|_| {
            invalid_document_limit_overflow("accessor bytes", limits.max_accessor_bytes)
        })?;
        let bytes = elements.checked_mul(bytes_per_element).ok_or_else(|| {
            invalid_document_limit_overflow("accessor bytes", limits.max_accessor_bytes)
        })?;
        accessor_elements = accessor_elements.checked_add(elements).ok_or_else(|| {
            invalid_document_limit_overflow("accessor elements", limits.max_accessor_elements)
        })?;
        accessor_bytes = accessor_bytes.checked_add(bytes).ok_or_else(|| {
            invalid_document_limit_overflow("accessor bytes", limits.max_accessor_bytes)
        })?;
    }
    if accessor_elements > limits.max_accessor_elements {
        return Err(invalid_document_limit_error(
            "accessor elements",
            accessor_elements,
            limits.max_accessor_elements,
        ));
    }
    if accessor_bytes > limits.max_accessor_bytes {
        return Err(invalid_document_limit_error(
            "accessor bytes",
            accessor_bytes,
            limits.max_accessor_bytes,
        ));
    }
    validate_document_collection_limits(document, limits)
}

fn validate_document_collection_limits(
    document: &gltf::Document,
    limits: &GltfDecodeLimits,
) -> Result<(), gltf::Error> {
    validate_document_count("node count", document.nodes().count(), limits.max_nodes)?;
    validate_document_count("skin count", document.skins().count(), limits.max_skins)?;
    validate_document_count("mesh count", document.meshes().count(), limits.max_meshes)?;
    validate_document_count(
        "material count",
        document.materials().count(),
        limits.max_materials,
    )?;
    validate_document_count(
        "animation count",
        document.animations().count(),
        limits.max_animations,
    )?;

    let mut skin_joint_references = 0u64;
    let node_count = document.as_json().nodes.len();
    for (skin_index, raw_skin) in document.as_json().skins.iter().enumerate() {
        for (joint_index, joint) in raw_skin.joints.iter().enumerate() {
            if joint.value() >= node_count {
                return Err(invalid_decode_source_error(format!(
                    "glTF skin {skin_index} joint {joint_index} references node {}; node count is {node_count}",
                    joint.value()
                )));
            }
        }
    }
    for skin in document.skins() {
        let joints = u64::try_from(skin.joints().count()).map_err(|_| {
            invalid_document_limit_overflow(
                "skin joint references",
                limits.max_skin_joint_references,
            )
        })?;
        skin_joint_references = skin_joint_references.checked_add(joints).ok_or_else(|| {
            invalid_document_limit_overflow(
                "skin joint references",
                limits.max_skin_joint_references,
            )
        })?;
    }
    if skin_joint_references > limits.max_skin_joint_references {
        return Err(invalid_document_limit_error(
            "skin joint references",
            skin_joint_references,
            limits.max_skin_joint_references,
        ));
    }

    let mut primitive_count = 0u64;
    for mesh in document.meshes() {
        let primitives = u64::try_from(mesh.primitives().count()).map_err(|_| {
            invalid_document_limit_overflow("primitive count", limits.max_primitives)
        })?;
        primitive_count = primitive_count.checked_add(primitives).ok_or_else(|| {
            invalid_document_limit_overflow("primitive count", limits.max_primitives)
        })?;
    }
    if primitive_count > limits.max_primitives {
        return Err(invalid_document_limit_error(
            "primitive count",
            primitive_count,
            limits.max_primitives,
        ));
    }

    let mut channel_count = 0u64;
    let mut input_values = 0u64;
    let mut output_values = 0u64;
    for animation in document.animations() {
        for (channel_index, channel) in animation.channels().enumerate() {
            channel_count = channel_count.checked_add(1).ok_or_else(|| {
                invalid_document_limit_overflow(
                    "animation channel count",
                    limits.max_animation_channels,
                )
            })?;
            if channel_count > limits.max_animation_channels {
                return Err(invalid_document_limit_error(
                    "animation channel count",
                    channel_count,
                    limits.max_animation_channels,
                ));
            }
            let channel_result = catch_unwind(AssertUnwindSafe(|| {
                validate_document_animation_channel(document, &animation, &channel, channel_index)
            }));
            let (channel_inputs, channel_outputs) = match channel_result {
                Ok(result) => result?,
                Err(_) => {
                    return Err(invalid_decode_source_error(format!(
                        "glTF animation {} channel {channel_index} has an invalid sampler or accessor/target/node reference",
                        animation.index()
                    )))
                }
            };
            input_values = input_values.checked_add(channel_inputs).ok_or_else(|| {
                invalid_document_limit_overflow(
                    "animation input key values",
                    limits.max_animation_key_values,
                )
            })?;
            if input_values > limits.max_animation_key_values {
                return Err(invalid_document_limit_error(
                    "animation input key values",
                    input_values,
                    limits.max_animation_key_values,
                ));
            }
            output_values = output_values.checked_add(channel_outputs).ok_or_else(|| {
                invalid_document_limit_overflow(
                    "animation output key values",
                    limits.max_animation_key_values,
                )
            })?;
            if output_values > limits.max_animation_key_values {
                return Err(invalid_document_limit_error(
                    "animation output key values",
                    output_values,
                    limits.max_animation_key_values,
                ));
            }
        }
    }

    Ok(())
}

fn validate_document_count(field: &str, count: usize, limit: u64) -> Result<u64, gltf::Error> {
    let actual = u64::try_from(count).map_err(|_| invalid_document_limit_overflow(field, limit))?;
    if actual > limit {
        return Err(invalid_document_limit_error(field, actual, limit));
    }
    Ok(actual)
}

fn validate_document_animation_channel(
    document: &gltf::Document,
    animation: &gltf::Animation<'_>,
    channel: &gltf::animation::Channel<'_>,
    channel_index: usize,
) -> Result<(u64, u64), gltf::Error> {
    let raw_animation = document
        .as_json()
        .animations
        .get(animation.index())
        .ok_or_else(|| {
            invalid_decode_source_error(format!(
                "glTF animation {} has an invalid document index",
                animation.index()
            ))
        })?;
    let raw_channel = raw_animation.channels.get(channel_index).ok_or_else(|| {
        invalid_decode_source_error(format!(
            "glTF animation {} channel {channel_index} has an invalid document index",
            animation.index()
        ))
    })?;
    let sampler_index = raw_channel.sampler.value();
    let raw_sampler = raw_animation.samplers.get(sampler_index).ok_or_else(|| {
        invalid_decode_source_error(format!(
            "glTF animation {} channel {channel_index} has an invalid sampler reference",
            animation.index()
        ))
    })?;
    if raw_sampler.input.value() >= document.as_json().accessors.len()
        || raw_sampler.output.value() >= document.as_json().accessors.len()
    {
        return Err(invalid_decode_source_error(format!(
            "glTF animation {} channel {channel_index} has an invalid accessor reference",
            animation.index()
        )));
    }
    if raw_channel.target.node.value() >= document.as_json().nodes.len() {
        return Err(invalid_decode_source_error(format!(
            "glTF animation {} channel {channel_index} has an invalid target node reference",
            animation.index()
        )));
    }
    let sampler = channel.sampler();
    let input_count = sampler.input().count();
    let output_count = sampler.output().count();
    let target = channel.target();
    let components_per_key = match target.property() {
        gltf::animation::Property::Translation
        | gltf::animation::Property::Rotation
        | gltf::animation::Property::Scale => 1,
        gltf::animation::Property::MorphTargetWeights => target
            .node()
            .mesh()
            .and_then(|mesh| {
                mesh.primitives()
                    .map(|primitive| primitive.morph_targets().len())
                    .max()
            })
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                invalid_decode_source_error(format!(
                    "glTF animation {} channel {channel_index} targets a node without morph targets",
                    animation.index()
                ))
            })?,
    };
    validate_animation_output_cardinality(
        input_count,
        matches!(
            sampler.interpolation(),
            gltf::animation::Interpolation::CubicSpline
        ),
        output_count,
        components_per_key,
    )
    .map_err(|error| {
        invalid_decode_source_error(format!(
            "glTF animation {} channel {channel_index} has invalid output cardinality: {error}",
            animation.index()
        ))
    })?;
    Ok((
        u64::try_from(input_count)
            .map_err(|_| invalid_document_limit_overflow("animation input key values", u64::MAX))?,
        u64::try_from(output_count).map_err(|_| {
            invalid_document_limit_overflow("animation output key values", u64::MAX)
        })?,
    ))
}

fn invalid_document_limit_error(field: &str, actual: u64, limit: u64) -> gltf::Error {
    invalid_decode_source_error(format!("glTF {field} is {actual}; limit is {limit}"))
}

fn invalid_document_limit_overflow(field: &str, limit: u64) -> gltf::Error {
    invalid_decode_source_error(format!(
        "glTF {field} aggregate overflowed before a representable actual value; limit is {limit}"
    ))
}

fn validate_source_byte_limit(
    source_bytes: u64,
    limits: &GltfDecodeLimits,
) -> Result<(), gltf::Error> {
    if source_bytes > limits.max_source_bytes {
        return Err(invalid_decode_source_error(format!(
            "glTF source is {source_bytes} bytes; limit is {} bytes",
            limits.max_source_bytes
        )));
    }
    Ok(())
}

fn validate_json_byte_limit(json_bytes: u64, limits: &GltfDecodeLimits) -> Result<(), gltf::Error> {
    if json_bytes > limits.max_json_bytes {
        return Err(invalid_decode_source_error(format!(
            "glTF JSON document is {json_bytes} bytes; limit is {} bytes",
            limits.max_json_bytes
        )));
    }
    Ok(())
}

fn validate_glb_header_limits(
    source_bytes: u64,
    header_tail: &[u8; 8],
    limits: &GltfDecodeLimits,
) -> Result<u64, gltf::Error> {
    let version = u32::from_le_bytes(header_tail[..4].try_into().unwrap());
    if version != 2 {
        return Err(invalid_decode_source_error(
            "glTF GLB source requires version 2",
        ));
    }
    let declared_length = u64::from(u32::from_le_bytes(header_tail[4..].try_into().unwrap()));
    if declared_length > limits.max_source_bytes {
        return Err(invalid_decode_source_error(format!(
            "glTF GLB declares {declared_length} bytes; limit is {} bytes",
            limits.max_source_bytes
        )));
    }
    if declared_length != source_bytes {
        return Err(invalid_decode_source_error(format!(
            "glTF GLB declares {declared_length} bytes but source has {source_bytes} bytes"
        )));
    }
    Ok(declared_length)
}

fn validate_glb_json_chunk_limits(
    declared_length: u64,
    chunk_header: &[u8; 8],
    limits: &GltfDecodeLimits,
) -> Result<(), gltf::Error> {
    if &chunk_header[4..] != b"JSON" {
        return Err(invalid_decode_source_error(
            "glTF GLB source requires a JSON first chunk",
        ));
    }
    let json_length = u64::from(u32::from_le_bytes(chunk_header[..4].try_into().unwrap()));
    if json_length > limits.max_json_bytes {
        return Err(invalid_decode_source_error(format!(
            "glTF GLB JSON chunk is {json_length} bytes; limit is {} bytes",
            limits.max_json_bytes
        )));
    }
    let json_end = 20u64
        .checked_add(json_length)
        .ok_or_else(|| invalid_decode_source_error("glTF GLB JSON chunk length overflowed"))?;
    if json_end > declared_length {
        return Err(invalid_decode_source_error(format!(
            "glTF GLB JSON chunk ends at {json_end} bytes beyond declared source length {declared_length}"
        )));
    }
    Ok(())
}

fn import_buffers_with_budget(
    document: &gltf::Document,
    base: Option<&Path>,
    mut blob: Option<Vec<u8>>,
    resource_budget: &mut EncodedResourceBudget,
) -> Result<Vec<gltf::buffer::Data>, gltf::Error> {
    let mut buffers = Vec::new();
    for buffer in document.buffers() {
        let mut data = match buffer.source() {
            gltf::buffer::Source::Uri(uri) => read_external_uri(uri, base, resource_budget)?,
            gltf::buffer::Source::Bin => blob.take().ok_or(gltf::Error::MissingBlob)?,
        };
        while data.len() % 4 != 0 {
            data.push(0);
        }
        if data.len() < buffer.length() {
            return Err(gltf::Error::BufferLength {
                buffer: buffer.index(),
                expected: buffer.length(),
                actual: data.len(),
            });
        }
        buffers.push(gltf::buffer::Data(data));
    }
    Ok(buffers)
}

fn read_external_uri(
    uri: &str,
    base: Option<&Path>,
    resource_budget: &mut EncodedResourceBudget,
) -> Result<Vec<u8>, gltf::Error> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let mut parts = rest.split(";base64,");
        let payload = match (parts.next(), parts.next()) {
            (_, Some(payload)) => payload,
            (Some(payload), None) => payload,
            _ => return Err(gltf::Error::UnsupportedScheme),
        };
        return read_data_uri(payload, resource_budget);
    }

    if let Some(path) = uri.strip_prefix("file://") {
        if base.is_none() {
            return Err(gltf::Error::ExternalReferenceInSliceImport);
        }
        return read_bounded_file(Path::new(path), resource_budget);
    }
    if let Some(path) = uri.strip_prefix("file:") {
        if base.is_none() {
            return Err(gltf::Error::ExternalReferenceInSliceImport);
        }
        return read_bounded_file(Path::new(path), resource_budget);
    }
    if uri.contains(':') {
        return Err(gltf::Error::UnsupportedScheme);
    }

    let path = urlencoding::decode(uri).map_err(|error| {
        invalid_decode_source_error(format!("glTF URI decoding failed: {error}"))
    })?;
    let base = base.ok_or(gltf::Error::ExternalReferenceInSliceImport)?;
    read_bounded_file(&base.join(path.as_ref()), resource_budget)
}

fn read_bounded_file(
    path: &Path,
    resource_budget: &mut EncodedResourceBudget,
) -> Result<Vec<u8>, gltf::Error> {
    let file = File::open(path).map_err(gltf::Error::Io)?;
    let read_limit = resource_budget.remaining().saturating_add(1);
    let mut data = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut data)
        .map_err(gltf::Error::Io)?;
    resource_budget.reserve(data.len())?;
    Ok(data)
}

fn read_data_uri(
    payload: &str,
    resource_budget: &mut EncodedResourceBudget,
) -> Result<Vec<u8>, gltf::Error> {
    let estimated_decoded_bytes = base64_decoded_length_bound(payload)?;
    resource_budget.ensure_bound(estimated_decoded_bytes)?;
    let decoded = base64::decode(payload).map_err(gltf::Error::Base64)?;
    resource_budget.reserve(decoded.len())?;
    Ok(decoded)
}

fn base64_decoded_length_bound(payload: &str) -> Result<u64, gltf::Error> {
    let bytes = payload.as_bytes();
    let payload_bytes = u64::try_from(bytes.len())
        .map_err(|_| invalid_decode_source_error("glTF data URI length does not fit u64"))?;
    let remainder = bytes.len() % 4;
    if remainder == 1 {
        return Err(gltf::Error::Base64(base64::DecodeError::InvalidLength));
    }

    let mut padding = 0usize;
    for (index, byte) in bytes.iter().enumerate().rev() {
        if *byte != b'=' {
            break;
        }
        padding += 1;
        if padding > 2 {
            return Err(gltf::Error::Base64(base64::DecodeError::InvalidByte(
                index, *byte,
            )));
        }
    }
    if padding > 0 && remainder != 0 {
        return Err(gltf::Error::Base64(base64::DecodeError::InvalidByte(
            bytes.len() - padding,
            b'=',
        )));
    }

    let content_len = bytes.len().saturating_sub(padding);
    for (index, byte) in bytes[..content_len].iter().copied().enumerate() {
        if !byte.is_ascii_alphanumeric() && byte != b'+' && byte != b'/' {
            return Err(gltf::Error::Base64(base64::DecodeError::InvalidByte(
                index, byte,
            )));
        }
    }
    if bytes[..content_len].contains(&b'=') {
        let index = bytes[..content_len]
            .iter()
            .position(|byte| *byte == b'=')
            .unwrap();
        return Err(gltf::Error::Base64(base64::DecodeError::InvalidByte(
            index, b'=',
        )));
    }

    let groups = payload_bytes / 4;
    let full_group_bytes = groups
        .checked_mul(3)
        .ok_or_else(|| invalid_decode_source_error("glTF data URI decoded length overflowed"))?;
    if padding > 0 {
        return full_group_bytes
            .checked_sub(u64::try_from(padding).unwrap())
            .ok_or_else(|| {
                invalid_decode_source_error("glTF data URI decoded length underflowed")
            });
    }
    let tail_bytes = match remainder {
        0 => 0,
        2 => 1,
        3 => 2,
        _ => unreachable!(),
    };
    full_group_bytes
        .checked_add(tail_bytes)
        .ok_or_else(|| invalid_decode_source_error("glTF data URI decoded length overflowed"))
}

fn import_images_with_budget(
    document: &gltf::Document,
    base: Option<&Path>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<DynamicImage>, gltf::Error> {
    let mut budget = DecodedImageBudget::default();
    document
        .images()
        .map(|image| decode_image_source(image.source(), base, buffers, &mut budget, None))
        .collect()
}

fn import_images_with_encoded_budget(
    document: &gltf::Document,
    base: Option<&Path>,
    buffers: &[gltf::buffer::Data],
    resource_budget: &mut EncodedResourceBudget,
) -> Result<Vec<DynamicImage>, gltf::Error> {
    let mut budget = DecodedImageBudget::default();
    let mut images = Vec::new();
    for image in document.images() {
        images.push(decode_image_source(
            image.source(),
            base,
            buffers,
            &mut budget,
            Some(resource_budget),
        )?);
    }
    Ok(images)
}

fn decode_image_source(
    source: gltf::image::Source<'_>,
    base: Option<&Path>,
    buffers: &[gltf::buffer::Data],
    budget: &mut DecodedImageBudget,
    resource_budget: Option<&mut EncodedResourceBudget>,
) -> Result<DynamicImage, gltf::Error> {
    let (encoded, uri, mime_type) = match source {
        gltf::image::Source::View { view, mime_type } => {
            let buffer_index = view.buffer().index();
            let buffer = buffers.get(buffer_index).ok_or(gltf::Error::BufferLength {
                buffer: buffer_index,
                expected: view.offset().saturating_add(view.length()),
                actual: 0,
            })?;
            let begin = view.offset();
            let end = begin
                .checked_add(view.length())
                .ok_or(gltf::Error::BufferLength {
                    buffer: buffer_index,
                    expected: usize::MAX,
                    actual: buffer.len(),
                })?;
            let encoded = buffer.get(begin..end).ok_or(gltf::Error::BufferLength {
                buffer: buffer_index,
                expected: end,
                actual: buffer.len(),
            })?;
            (encoded.to_vec(), None, Some(mime_type))
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let encoded = match resource_budget {
                Some(resource_budget) => read_external_uri(uri, base, resource_budget)?,
                None => gltf::buffer::Data::from_source(gltf::buffer::Source::Uri(uri), base)?.0,
            };
            (encoded, Some(uri), mime_type)
        }
    };
    let format = image_format(uri, mime_type)?;
    decode_image_bytes(&encoded, format, budget)
}

fn image_format(uri: Option<&str>, mime_type: Option<&str>) -> Result<ImageFormat, gltf::Error> {
    let format = if let Some(mime_type) = mime_type {
        mime_type
    } else if let Some(uri) = uri.filter(|uri| uri.starts_with("data:")) {
        uri.strip_prefix("data:")
            .and_then(|value| value.split(";base64,").next())
            .unwrap_or("")
    } else {
        uri.and_then(|uri| uri.rsplit('.').next()).unwrap_or("")
    };
    match format {
        "image/png" | "png" => Ok(ImageFormat::Png),
        "image/jpeg" | "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        _ => Err(gltf::Error::UnsupportedImageEncoding),
    }
}

fn decode_image_bytes(
    encoded: &[u8],
    format: ImageFormat,
    budget: &mut DecodedImageBudget,
) -> Result<DynamicImage, gltf::Error> {
    let mut reader = ImageReader::with_format(Cursor::new(encoded), format);
    reader.limits(budget.limits()?);
    let image = reader.decode().map_err(gltf::Error::Image)?;
    budget.reserve(image.as_bytes().len())?;
    Ok(image)
}

/// Builds a [`GltfModel`] from already-imported glTF data. `prefix` namespaces the
/// mesh/texture cache keys so separately loaded files don't collide.
fn build_model(
    doc: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<DynamicImage>,
    prefix: &str,
) -> Result<GltfModel, gltf::Error> {
    validate_node_hierarchy(&doc)?;
    let prefix = prefix.to_owned();

    // Images are decoded once; textures are then created on demand per
    // (image, color-space) pair.
    let mut tex_cache: HashMap<(usize, bool), Arc<Texture>> = HashMap::new();

    // Build GPU meshes for every primitive of every mesh, keyed by mesh index.
    let mut meshes: Vec<Vec<PrimInfo>> = Vec::with_capacity(doc.meshes().count());
    for mesh in doc.meshes() {
        let mut prims = Vec::new();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "gltf: skipping non-triangle primitive (mode {:?}) in mesh {:?}",
                    prim.mode(),
                    bounded_warning_text(mesh.name().unwrap_or(""))
                );
                continue;
            }
            if let Some(gpu) = build_primitive(&prim, &buffers)? {
                prims.push(PrimInfo {
                    mesh: Rc::new(RefCell::new(gpu)),
                    material_index: prim.material().index(),
                });
            }
        }
        meshes.push(prims);
    }

    // Create one scene node per glTF node (no parenting yet), with its local TRS.
    let mut node_map: Vec<SceneNode3d> = doc
        .nodes()
        .map(|node| {
            let (t, r, s) = node.transform().decomposed();
            let pose = Pose3::from_parts(Vec3::from_array(t), Quat::from_array(r));
            SceneNode3d::new(Vec3::from_array(s), pose, None)
        })
        .collect();

    // Resolve every fallible skin and animation decode before registering any
    // material texture in the global manager. A malformed inverse-bind accessor
    // or animation channel must not leave a failed load's textures cached.
    let mut skin_data_by_node = HashMap::new();
    for node in doc.nodes() {
        if node.mesh().is_none() {
            continue;
        }
        if let Some(skin) = node.skin() {
            skin_data_by_node.insert(node.index(), build_skin_parts(&skin, &buffers, &node_map)?);
        }
    }
    let clips = doc
        .animations()
        .map(|anim| build_clip(&anim, &buffers, &node_map))
        .collect::<Result<Vec<_>, _>>()?;

    // Resolve every material texture source before registering anything in the
    // global texture manager.  The gltf crate's convenience accessors unwrap
    // malformed texture/image indices; keep that panic at this fallible,
    // side-effect-free boundary so a rejected asset cannot leave cached GPU
    // textures behind.
    validate_material_texture_sources(&doc, images.len())?;

    let material = MaterialManager3d::get_global_manager(|mm| mm.get_default());
    let default_tex = TextureManager::get_global_manager(|tm| tm.get_default());
    // `Document::materials()` is a forward iterator; calling `nth` for every
    // primitive makes material binding O(primitives * materials). Keep the
    // borrowed handles once so lookup is constant-time during graph assembly.
    let gltf_materials: Vec<_> = doc.materials().collect();

    // Attach mesh primitives (as child object nodes) and skins to each node.
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };

        for prim in &meshes[mesh.index()] {
            let mut object = Object3d::new(
                prim.mesh.clone(),
                Color::new(1.0, 1.0, 1.0, 1.0),
                default_tex.clone(),
                material.clone(),
            );
            // A primitive without a material uses glTF's default material,
            // whose alpha mode is OPAQUE (kiss3d defaults to Blend).
            let gltf_mat = prim
                .material_index
                .and_then(|mat_index| gltf_materials.get(mat_index));
            object.set_alpha_mode(gltf_alpha_mode_or_default(gltf_mat));
            if let Some(gltf_mat) = gltf_mat {
                apply_material(&mut object, gltf_mat, &images, &mut tex_cache, &prefix);
            }
            // A skinned mesh only deforms if it actually carries per-vertex
            // joints/weights; otherwise it stays a rigid child of the node.
            if prim.mesh.borrow().has_skin_vertices() {
                if let Some((joints, inverse_bind)) = skin_data_by_node.get(&node.index()) {
                    object.set_skin(Skin3d::new(joints.clone(), inverse_bind.clone()));
                }
            }
            // Seed the morph weights (one per target) from the node's or mesh's glTF
            // default weights, padded/truncated to the primitive's target count.
            let num_targets = prim.mesh.borrow().morph_target_count();
            if num_targets > 0 {
                let mut weights = vec![0.0f32; num_targets];
                if let Some(src) = node.weights().or_else(|| mesh.weights()) {
                    let n = src.len().min(num_targets);
                    weights[..n].copy_from_slice(&src[..n]);
                }
                object.data_mut().set_morph_weights(&weights);
            }
            node_map[node.index()].add_object(Vec3::ONE, Pose3::IDENTITY, object);
        }
    }

    // Link the node hierarchy.
    let mut parent_by_node = vec![None; node_map.len()];
    for node in doc.nodes() {
        let mut parent = node_map[node.index()].clone();
        for child in node.children() {
            parent_by_node[child.index()] = Some(node.index());
            parent.add_child(node_map[child.index()].clone());
        }
    }

    let joint_nodes: HashSet<usize> = doc
        .skins()
        .flat_map(|skin| skin.joints().map(|joint| joint.index()))
        .collect();
    let skeleton_edges = joint_edges(&parent_by_node, &joint_nodes)
        .into_iter()
        .map(|(parent_index, child_index)| {
            (
                node_map[parent_index].clone(),
                node_map[child_index].clone(),
            )
        })
        .collect();
    let skeleton_nodes = doc
        .nodes()
        .filter_map(|node| {
            let index = node.index();
            if !joint_nodes.contains(&index) {
                return None;
            }
            let parent_index =
                parent_by_node[index].filter(|parent_index| joint_nodes.contains(parent_index));
            let name = node
                .name()
                .filter(|name| !name.trim().is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("Joint {}", index + 1));
            Some(SkeletonNode3d {
                index,
                parent_index,
                name,
                node: node_map[index].clone(),
            })
        })
        .collect();

    // Assemble the model root from the default (or first) scene's root nodes.
    let mut root = SceneNode3d::empty();
    let scene = selected_scene(&doc);
    if let Some(scene) = scene {
        for node in scene.nodes() {
            root.add_child(node_map[node.index()].clone());
        }
    }

    Ok(GltfModel {
        root,
        player: AnimationPlayer::new(clips),
        skeleton_edges,
        skeleton_nodes,
    })
}

/// Returns the scene rendered by the loader: the authored default scene, or
/// the first scene when no default is declared.
fn selected_scene<'a>(doc: &'a gltf::Document) -> Option<gltf::Scene<'a>> {
    doc.default_scene().or_else(|| doc.scenes().next())
}

/// Extracts the static scene/geometry subset without creating any GPU or scene
/// graph values.  The production `load` path deliberately does not call this
/// yet; keeping it separate makes the future CPU decode boundary testable.
fn extract_static_template(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<GltfAssetTemplate, gltf::Error> {
    let node_count = doc.nodes().count();
    let mut parents = vec![None; node_count];
    let mut children = vec![Vec::new(); node_count];
    for node in doc.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
            children[node.index()].push(child.index());
        }
    }

    // The parent/child tables are only staging data. Move each child list into
    // the owned DTO instead of cloning it once per node.
    let mut nodes = Vec::with_capacity(node_count);
    for node in doc.nodes() {
        let (translation, rotation, scale) = node.transform().decomposed();
        let index = node.index();
        nodes.push(CpuNode {
            parent: parents[index],
            children: std::mem::take(&mut children[index]),
            mesh: node.mesh().map(|mesh| mesh.index()),
            skin: node.skin().map(|skin| skin.index()),
            weights: node
                .weights()
                .or_else(|| node.mesh().and_then(|mesh| mesh.weights()))
                .map(|weights| weights.to_vec())
                .unwrap_or_default(),
            translation,
            rotation,
            scale,
        });
    }

    let meshes = doc
        .meshes()
        .map(|mesh| {
            let primitives = mesh
                .primitives()
                .map(|primitive| extract_static_primitive(&primitive, buffers))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            Ok::<_, gltf::Error>(CpuMesh {
                primitives,
                weights: mesh
                    .weights()
                    .map(|weights| weights.to_vec())
                    .unwrap_or_default(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let default_scene_roots = selected_scene(doc)
        .map(|scene| scene.nodes().map(|node| node.index()).collect())
        .unwrap_or_default();

    Ok(GltfAssetTemplate {
        nodes,
        meshes,
        default_scene_roots,
        images: Vec::new(),
        materials: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
    })
}

/// Extracts triangle geometry into plain CPU-owned arrays.  Unsupported
/// primitive modes and primitives without positions are skipped just as the
/// existing GPU builder does.
fn extract_static_primitive(
    prim: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<CpuPrimitive>, gltf::Error> {
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Ok(None);
    }
    let reader = prim.reader(|b| Some(&buffers[b.index()]));
    let Some(positions) = reader.read_positions() else {
        return Ok(None);
    };
    let positions: Vec<[f32; 3]> = positions.collect();
    let num_vertices = positions.len();
    let indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<u32>>(),
        None => (0..num_vertices)
            .map(|index| {
                u32::try_from(index).map_err(|_| {
                    invalid_cpu_template("glTF primitive vertex count exceeds u32 indexing")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let faces = triangle_faces_from_indices(indices)?;
    let normals = reader.read_normals().map(|values| {
        values
            .map(|normal| [normal[0], normal[1], normal[2]])
            .collect()
    });
    let uvs = reader
        .read_tex_coords(0)
        .map(|values| values.into_f32().map(|uv| [uv[0], uv[1]]).collect());

    let (skin_joints, skin_weights) = match (reader.read_joints(0), reader.read_weights(0)) {
        (Some(joints), Some(weights)) => (
            Some(
                joints
                    .into_u16()
                    .map(|joint| {
                        [
                            joint[0] as u32,
                            joint[1] as u32,
                            joint[2] as u32,
                            joint[3] as u32,
                        ]
                    })
                    .collect(),
            ),
            Some(weights.into_f32().collect()),
        ),
        _ => (None, None),
    };

    let mut morph_positions = Vec::new();
    let mut morph_normals = Vec::new();
    let mut num_targets = 0usize;
    let mut any_normals = false;
    for (target_positions, normals, _tangents) in reader.read_morph_targets() {
        if num_targets >= crate::builtin::deform::MAX_MORPH_TARGETS {
            break;
        }
        num_targets += 1;
        match target_positions {
            Some(values) => {
                morph_positions.extend(values.map(|value| [value[0], value[1], value[2], 0.0]))
            }
            None => morph_positions.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
        match normals {
            Some(values) => {
                any_normals = true;
                morph_normals.extend(values.map(|value| [value[0], value[1], value[2], 0.0]));
            }
            None => morph_normals.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
    }
    let morph = (num_targets > 0).then(|| CpuMorphTargets {
        num_targets,
        positions: morph_positions,
        normals: any_normals.then_some(morph_normals),
    });

    Ok(Some(CpuPrimitive {
        positions,
        faces,
        normals,
        uvs,
        material_index: prim.material().index(),
        skin_joints,
        skin_weights,
        morph,
    }))
}

fn triangle_faces_from_indices(indices: Vec<u32>) -> Result<Vec<[u32; 3]>, gltf::Error> {
    if !indices.len().is_multiple_of(3) {
        return Err(invalid_cpu_template(
            "glTF triangle primitive index count is not a multiple of three",
        ));
    }
    Ok(indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

/// Materializes the first CPU-owned template seam shared by the future path
/// and in-memory decode entry points.  The production loader intentionally
/// remains on `build_model` until skin/animation metadata is part of this DTO.
fn decode_cpu_internal(
    doc: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<DynamicImage>,
) -> Result<GltfAssetTemplate, gltf::Error> {
    validate_node_hierarchy(&doc)?;
    let mut template = extract_static_template(&doc, &buffers)?;
    template.images = images
        .into_iter()
        .map(|decoded| CpuImage { decoded })
        .collect();
    template.materials = doc
        .materials()
        .map(|material| {
            let pbr = material.pbr_metallic_roughness();
            let factors = material_factors(&material);
            let alpha_mode = gltf_alpha_mode_or_default(Some(&material));
            let alpha_cutoff = match alpha_mode {
                AlphaMode::Mask(cutoff) => Some(cutoff),
                AlphaMode::Opaque | AlphaMode::Blend | AlphaMode::Premultiplied => None,
            };
            let base_color_texture = pbr
                .base_color_texture()
                .map(|info| info.texture().source().index());
            let metallic_roughness_texture = pbr
                .metallic_roughness_texture()
                .map(|info| info.texture().source().index());
            let normal_texture = material
                .normal_texture()
                .map(|info| info.texture().source().index());
            let occlusion_texture = material
                .occlusion_texture()
                .map(|info| info.texture().source().index());
            let emissive_texture = material
                .emissive_texture()
                .map(|info| info.texture().source().index());
            CpuMaterial {
                base_color: factors.base_color,
                metallic: factors.metallic,
                roughness: factors.roughness,
                emissive: factors.emissive,
                extensions: factors.extensions,
                alpha_mode,
                alpha_cutoff,
                base_color_image: base_color_texture,
                metallic_roughness_image: metallic_roughness_texture,
                normal_image: normal_texture,
                occlusion_image: occlusion_texture,
                emissive_image: emissive_texture,
            }
        })
        .collect();
    template.skins = extract_cpu_skins(&doc, &buffers);
    template.animations = extract_cpu_animations(&doc, &buffers)?;
    validate_cpu_template(&template)?;
    Ok(template)
}

fn extract_cpu_skins(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<CpuSkin> {
    doc.skins()
        .map(|skin| {
            let joints: Vec<usize> = skin.joints().map(|joint| joint.index()).collect();
            let reader = skin.reader(|b| Some(&buffers[b.index()]));
            let identity = Mat4::IDENTITY.to_cols_array_2d();
            let inverse_bind = reader
                .read_inverse_bind_matrices()
                .map(|values| values.collect())
                .unwrap_or_else(|| vec![identity; joints.len()]);
            CpuSkin {
                joints,
                inverse_bind,
            }
        })
        .collect()
}

fn extract_cpu_animations(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<CpuAnimationClip>, gltf::Error> {
    doc.animations()
        .map(|animation| {
            let channels = animation
                .channels()
                .map(|channel| extract_cpu_animation_channel(&channel, buffers))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            Ok(CpuAnimationClip {
                name: animation.name().unwrap_or("").to_string(),
                channels,
            })
        })
        .collect()
}

fn extract_cpu_animation_channel(
    channel: &gltf::animation::Channel<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<CpuAnimationChannel>, gltf::Error> {
    let interpolation = match channel.sampler().interpolation() {
        gltf::animation::Interpolation::Linear => Interpolation::Linear,
        gltf::animation::Interpolation::Step => Interpolation::Step,
        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
    };
    let reader = channel.reader(|b| Some(&buffers[b.index()]));
    let times: Vec<f32> = reader
        .read_inputs()
        .ok_or_else(|| invalid_cpu_template("glTF animation channel has no input accessor"))?
        .collect();
    let output = match reader
        .read_outputs()
        .ok_or_else(|| invalid_cpu_template("glTF animation channel has no output accessor"))?
    {
        gltf::animation::util::ReadOutputs::Translations(values) => {
            let values = values.collect::<Vec<_>>();
            validate_animation_output_cardinality(
                times.len(),
                interpolation == Interpolation::CubicSpline,
                values.len(),
                1,
            )
            .map_err(invalid_cpu_template)?;
            CpuAnimationOutput::Translation(values)
        }
        gltf::animation::util::ReadOutputs::Scales(values) => {
            let values = values.collect::<Vec<_>>();
            validate_animation_output_cardinality(
                times.len(),
                interpolation == Interpolation::CubicSpline,
                values.len(),
                1,
            )
            .map_err(invalid_cpu_template)?;
            CpuAnimationOutput::Scale(values)
        }
        gltf::animation::util::ReadOutputs::Rotations(values) => {
            let values = values.into_f32().collect::<Vec<_>>();
            validate_animation_output_cardinality(
                times.len(),
                interpolation == Interpolation::CubicSpline,
                values.len(),
                1,
            )
            .map_err(invalid_cpu_template)?;
            CpuAnimationOutput::Rotation(values)
        }
        gltf::animation::util::ReadOutputs::MorphTargetWeights(values) => {
            let values: Vec<f32> = values.into_f32().collect();
            let num_targets = morph_target_count(&channel.target().node()).ok_or_else(|| {
                invalid_cpu_template("glTF morph animation target has no morph targets")
            })?;
            validate_animation_output_cardinality(
                times.len(),
                interpolation == Interpolation::CubicSpline,
                values.len(),
                num_targets,
            )
            .map_err(invalid_cpu_template)?;
            CpuAnimationOutput::MorphWeights {
                values,
                num_targets,
            }
        }
    };
    Ok(Some(CpuAnimationChannel {
        target_node: channel.target().node().index(),
        times,
        interpolation,
        output,
    }))
}

fn invalid_cpu_template(message: impl Into<String>) -> gltf::Error {
    gltf::Error::Io(IoError::new(ErrorKind::InvalidData, message.into()))
}

fn validate_node_hierarchy(doc: &gltf::Document) -> Result<(), gltf::Error> {
    let node_count = doc.nodes().count();
    let mut parent_by_node = vec![None; node_count];
    for node in doc.nodes() {
        let (translation, rotation, scale) = node.transform().decomposed();
        validate_node_transform_values(&translation, &rotation, &scale)?;
        for child in node.children() {
            let Some(parent) = parent_by_node.get_mut(child.index()) else {
                return Err(invalid_cpu_template(
                    "glTF node child index is out of range",
                ));
            };
            if parent.replace(node.index()).is_some() {
                return Err(invalid_cpu_template(
                    "glTF node has more than one parent or appears twice",
                ));
            }
        }
    }
    validate_parent_graph(&parent_by_node).map_err(invalid_cpu_template)?;

    let scene = selected_scene(doc);
    if let Some(scene) = scene {
        let mut roots = HashSet::new();
        for node in scene.nodes() {
            if !roots.insert(node.index()) {
                return Err(invalid_cpu_template(
                    "glTF scene lists the same root node more than once",
                ));
            }
            if parent_by_node[node.index()].is_some() {
                return Err(invalid_cpu_template(
                    "glTF scene root node also has a parent",
                ));
            }
        }
    }
    Ok(())
}

fn validate_parent_graph(parent_by_node: &[Option<usize>]) -> Result<(), &'static str> {
    for &parent in parent_by_node.iter().flatten() {
        if parent >= parent_by_node.len() {
            return Err("glTF node parent index is out of range");
        }
    }
    let mut state = vec![0_u8; parent_by_node.len()];
    for start in 0..parent_by_node.len() {
        if state[start] == 2 {
            continue;
        }
        let mut current = start;
        let mut path = Vec::new();
        let mut reached_root = false;
        while state[current] == 0 {
            state[current] = 1;
            path.push(current);
            let Some(parent) = parent_by_node[current] else {
                reached_root = true;
                break;
            };
            current = parent;
        }
        if !reached_root && state[current] == 1 {
            return Err("glTF node hierarchy contains a cycle");
        }
        for node in path {
            state[node] = 2;
        }
    }
    Ok(())
}

fn validate_animation_times(times: &[f32]) -> Result<(), &'static str> {
    if times.iter().any(|time| !time.is_finite()) {
        return Err("glTF animation key times must be finite");
    }
    if times.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err("glTF animation key times must be non-decreasing");
    }
    Ok(())
}

fn validate_animation_vec3(values: &[Vec3]) -> Result<(), &'static str> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("glTF animation vector output must be finite");
    }
    Ok(())
}

fn validate_animation_vec3_arrays(values: &[[f32; 3]]) -> Result<(), &'static str> {
    if !finite_array_values(values) {
        return Err("glTF animation vector output must be finite");
    }
    Ok(())
}

fn validate_animation_quat(values: &[Quat], cubic_spline: bool) -> Result<(), &'static str> {
    if values.iter().enumerate().any(|(index, value)| {
        let length_squared = value.length_squared();
        !value.is_finite()
            || !length_squared.is_finite()
            || (length_squared <= f32::EPSILON && (!cubic_spline || index % 3 == 1))
    }) {
        return Err("glTF animation rotation output must be finite and non-zero");
    }
    Ok(())
}

fn validate_animation_quat_arrays(
    values: &[[f32; 4]],
    cubic_spline: bool,
) -> Result<(), &'static str> {
    if values.iter().enumerate().any(|(index, value)| {
        let length_squared = value
            .iter()
            .map(|component| component * component)
            .sum::<f32>();
        !value.iter().all(|component| component.is_finite())
            || !length_squared.is_finite()
            || (length_squared <= f32::EPSILON && (!cubic_spline || index % 3 == 1))
    }) {
        return Err("glTF animation rotation output must be finite and non-zero");
    }
    Ok(())
}

fn validate_animation_scalars(values: &[f32]) -> Result<(), &'static str> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("glTF animation scalar output must be finite");
    }
    Ok(())
}

fn validate_cpu_animation_output(
    output: &CpuAnimationOutput,
    interpolation: Interpolation,
) -> Result<(), gltf::Error> {
    match output {
        CpuAnimationOutput::Translation(values) | CpuAnimationOutput::Scale(values) => {
            validate_animation_vec3_arrays(values).map_err(invalid_cpu_template)?;
        }
        CpuAnimationOutput::Rotation(values) => {
            validate_animation_quat_arrays(values, interpolation == Interpolation::CubicSpline)
                .map_err(invalid_cpu_template)?;
        }
        CpuAnimationOutput::MorphWeights { values, .. } => {
            validate_animation_scalars(values).map_err(invalid_cpu_template)?;
        }
    }
    Ok(())
}

fn validate_finite_array<const N: usize>(
    values: &[[f32; N]],
    message: &'static str,
) -> Result<(), gltf::Error> {
    if !finite_array_values(values) {
        return Err(invalid_cpu_template(message));
    }
    Ok(())
}

fn finite_array_values<const N: usize>(values: &[[f32; N]]) -> bool {
    values
        .iter()
        .all(|value| value.iter().all(|component| component.is_finite()))
}

fn validate_node_transform_values(
    translation: &[f32; 3],
    rotation: &[f32; 4],
    scale: &[f32; 3],
) -> Result<(), gltf::Error> {
    validate_finite_array(
        std::slice::from_ref(translation),
        "glTF node translation must be finite",
    )?;
    validate_finite_array(
        std::slice::from_ref(rotation),
        "glTF node rotation must be finite",
    )?;
    validate_finite_array(
        std::slice::from_ref(scale),
        "glTF node scale must be finite",
    )?;
    let rotation = Quat::from_array(*rotation);
    if !rotation.length_squared().is_finite() || rotation.length_squared() <= f32::EPSILON {
        return Err(invalid_cpu_template("glTF node rotation must be non-zero"));
    }
    Ok(())
}

fn validate_cpu_primitive_values(primitive: &CpuPrimitive) -> Result<(), gltf::Error> {
    validate_finite_array(
        &primitive.positions,
        "glTF primitive positions must be finite",
    )?;
    if let Some(normals) = &primitive.normals {
        validate_finite_array(normals, "glTF primitive normals must be finite")?;
    }
    if let Some(uvs) = &primitive.uvs {
        validate_finite_array(uvs, "glTF primitive UVs must be finite")?;
    }
    if let Some(weights) = &primitive.skin_weights {
        validate_finite_array(weights, "glTF primitive skin weights must be finite")?;
    }
    if let Some(morph) = &primitive.morph {
        validate_finite_array(
            &morph.positions,
            "glTF primitive morph values must be finite",
        )?;
        if let Some(normals) = &morph.normals {
            validate_finite_array(normals, "glTF primitive morph values must be finite")?;
        }
    }
    Ok(())
}

fn validate_cpu_template(template: &GltfAssetTemplate) -> Result<(), gltf::Error> {
    for node in &template.nodes {
        validate_node_transform_values(&node.translation, &node.rotation, &node.scale)?;
        if let Some(mesh) = node.mesh {
            if mesh >= template.meshes.len() {
                return Err(invalid_cpu_template("glTF node mesh index is out of range"));
            }
        }
        if let Some(skin) = node.skin {
            if skin >= template.skins.len() {
                return Err(invalid_cpu_template("glTF node skin index is out of range"));
            }
        }
    }
    for mesh in &template.meshes {
        for primitive in &mesh.primitives {
            validate_cpu_primitive_values(primitive)?;
            validate_primitive_geometry(
                primitive.positions.len(),
                &primitive.faces,
                primitive.normals.as_ref().map(Vec::len),
                primitive.uvs.as_ref().map(Vec::len),
                primitive.skin_joints.as_ref().map(Vec::len),
                primitive.skin_weights.as_ref().map(Vec::len),
                primitive.morph.as_ref().map(|morph| {
                    (
                        morph.num_targets,
                        morph.positions.len(),
                        morph.normals.as_ref().map(Vec::len),
                    )
                }),
            )
            .map_err(invalid_cpu_template)?;
            if let Some(material) = primitive.material_index {
                if material >= template.materials.len() {
                    return Err(invalid_cpu_template(
                        "glTF primitive material index is out of range",
                    ));
                }
            }
        }
    }
    for skin in &template.skins {
        if skin.inverse_bind.len() != skin.joints.len()
            || skin
                .joints
                .iter()
                .any(|joint| *joint >= template.nodes.len())
            || skin
                .inverse_bind
                .iter()
                .flat_map(|matrix| matrix.iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err(invalid_cpu_template(
                "glTF skin joints or inverse-bind matrices are invalid",
            ));
        }
    }
    for clip in &template.animations {
        for channel in &clip.channels {
            if channel.target_node >= template.nodes.len() {
                return Err(invalid_cpu_template(
                    "glTF animation channel target is invalid",
                ));
            }
            validate_animation_times(&channel.times).map_err(invalid_cpu_template)?;
            validate_cpu_animation_output(&channel.output, channel.interpolation)?;
            let key_count = channel.times.len();
            let factor = usize::from(channel.interpolation == Interpolation::CubicSpline) * 2 + 1;
            match &channel.output {
                CpuAnimationOutput::Translation(values) | CpuAnimationOutput::Scale(values) => {
                    let expected = key_count.checked_mul(factor).ok_or_else(|| {
                        invalid_cpu_template("glTF animation output count overflowed")
                    })?;
                    if values.len() != expected {
                        return Err(invalid_cpu_template(
                            "glTF animation vector output count is invalid",
                        ));
                    }
                }
                CpuAnimationOutput::Rotation(values) => {
                    let expected = key_count.checked_mul(factor).ok_or_else(|| {
                        invalid_cpu_template("glTF animation output count overflowed")
                    })?;
                    if values.len() != expected {
                        return Err(invalid_cpu_template(
                            "glTF animation rotation output count is invalid",
                        ));
                    }
                }
                CpuAnimationOutput::MorphWeights {
                    values,
                    num_targets,
                } => {
                    if *num_targets == 0 {
                        return Err(invalid_cpu_template(
                            "glTF animation morph target count is zero",
                        ));
                    }
                    let expected = key_count
                        .checked_mul(*num_targets)
                        .and_then(|count| count.checked_mul(factor))
                        .ok_or_else(|| {
                            invalid_cpu_template("glTF animation morph output count overflowed")
                        })?;
                    if values.len() != expected {
                        return Err(invalid_cpu_template(
                            "glTF animation morph output count is invalid",
                        ));
                    }
                }
            }
        }
    }
    for material in &template.materials {
        if !material_values_are_finite(material) {
            return Err(invalid_cpu_template("glTF material factors must be finite"));
        }
        if let AlphaMode::Mask(cutoff) = material.alpha_mode {
            if !cutoff.is_finite()
                || material.alpha_cutoff != Some(cutoff)
                || material.alpha_cutoff.is_none_or(|value| !value.is_finite())
            {
                return Err(invalid_cpu_template(
                    "glTF material alpha cutoff must be finite and consistent",
                ));
            }
        } else if material.alpha_cutoff.is_some() {
            return Err(invalid_cpu_template(
                "glTF material alpha cutoff is only valid for MASK mode",
            ));
        }
        let image_indices = [
            material.base_color_image,
            material.metallic_roughness_image,
            material.normal_image,
            material.occlusion_image,
            material.emissive_image,
        ];
        if image_indices
            .iter()
            .flatten()
            .any(|image| *image >= template.images.len())
        {
            return Err(invalid_cpu_template(
                "glTF material image index is out of range",
            ));
        }
    }
    Ok(())
}

fn validate_primitive_geometry(
    vertex_count: usize,
    faces: &[[u32; 3]],
    normals_len: Option<usize>,
    uvs_len: Option<usize>,
    skin_joints_len: Option<usize>,
    skin_weights_len: Option<usize>,
    morph: Option<(usize, usize, Option<usize>)>,
) -> Result<(), &'static str> {
    if faces
        .iter()
        .flatten()
        .any(|index| usize::try_from(*index).map_or(true, |index| index >= vertex_count))
    {
        return Err("glTF primitive face index is out of range");
    }
    if normals_len.is_some_and(|length| length != vertex_count) {
        return Err("glTF primitive normals do not match vertex count");
    }
    if uvs_len.is_some_and(|length| length != vertex_count) {
        return Err("glTF primitive UVs do not match vertex count");
    }
    match (skin_joints_len, skin_weights_len) {
        (Some(joints), Some(weights)) if joints == vertex_count && weights == vertex_count => {}
        (None, None) => {}
        _ => return Err("glTF primitive skin attributes do not match vertex count"),
    }
    if let Some((num_targets, positions_len, normals_len)) = morph {
        let expected = num_targets
            .checked_mul(vertex_count)
            .ok_or("glTF morph target count overflowed")?;
        if positions_len != expected || normals_len.is_some_and(|length| length != expected) {
            return Err("glTF morph target data does not match vertex count");
        }
    }
    Ok(())
}

fn joint_edges(
    parent_by_node: &[Option<usize>],
    joint_nodes: &HashSet<usize>,
) -> Vec<(usize, usize)> {
    parent_by_node
        .iter()
        .enumerate()
        .filter_map(|(child_index, parent_index)| {
            let parent_index = (*parent_index)?;
            (joint_nodes.contains(&parent_index) && joint_nodes.contains(&child_index))
                .then_some((parent_index, child_index))
        })
        .collect()
}

/// Reads one triangle primitive into a [`GpuMesh3d`], including optional skinning
/// attributes. Returns `Ok(None)` if the primitive has no positions and an
/// explicit error for malformed geometry.
fn build_primitive(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<GpuMesh3d>, gltf::Error> {
    let reader = prim.reader(|b| Some(&buffers[b.index()]));

    let Some(positions) = reader.read_positions() else {
        return Ok(None);
    };
    let positions: Vec<Vec3> = positions.map(Vec3::from_array).collect();
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(invalid_cpu_template(
            "glTF primitive positions must be finite",
        ));
    }
    let num_vertices = positions.len();

    let indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<u32>>(),
        None => (0..num_vertices)
            .map(|index| {
                u32::try_from(index).map_err(|_| {
                    invalid_cpu_template("glTF primitive vertex count exceeds u32 indexing")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let faces = triangle_faces_from_indices(indices)?;

    let normals: Option<Vec<Vec3>> = reader
        .read_normals()
        .map(|it| it.map(Vec3::from_array).collect());
    if normals
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(invalid_cpu_template(
            "glTF primitive normals must be finite",
        ));
    }
    let uvs: Option<Vec<Vec2>> = reader
        .read_tex_coords(0)
        .map(|it| it.into_f32().map(Vec2::from_array).collect());
    if uvs
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(invalid_cpu_template("glTF primitive UVs must be finite"));
    }

    // Skinning attributes: present together on skinned primitives. JOINTS_0 is
    // widened from u8/u16 to u32 so a single vertex format covers every mesh.
    let joints = reader.read_joints(0).map(|joints| {
        joints
            .into_u16()
            .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
            .collect::<Vec<_>>()
    });
    let weights = reader
        .read_weights(0)
        .map(|weights| weights.into_f32().collect::<Vec<_>>());
    if weights.as_ref().is_some_and(|values| {
        values
            .iter()
            .any(|value| !value.iter().all(|component| component.is_finite()))
    }) {
        return Err(invalid_cpu_template(
            "glTF primitive skin weights must be finite",
        ));
    }

    // Morph targets: per-target position (and optional normal) deltas, flattened to
    // `[target * num_vertices + vertex]`. Read from `read_morph_targets`, which
    // yields each target's optional position/normal/tangent iterators (tangents are
    // unused — kiss3d derives tangents from screen-space derivatives).
    let mut morph_positions: Vec<[f32; 4]> = Vec::new();
    let mut morph_normals: Vec<[f32; 4]> = Vec::new();
    let mut num_targets = 0usize;
    let mut any_normals = false;
    for (pos_iter, nrm_iter, _tan_iter) in reader.read_morph_targets() {
        if num_targets >= crate::builtin::deform::MAX_MORPH_TARGETS {
            log::warn!(
                "gltf: mesh has more than {} morph targets; extra targets ignored",
                crate::builtin::deform::MAX_MORPH_TARGETS
            );
            break;
        }
        num_targets += 1;
        match pos_iter {
            Some(it) => morph_positions.extend(it.map(|p| [p[0], p[1], p[2], 0.0])),
            None => morph_positions.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
        match nrm_iter {
            Some(it) => {
                any_normals = true;
                morph_normals.extend(it.map(|n| [n[0], n[1], n[2], 0.0]));
            }
            None => morph_normals.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
    }
    if morph_positions
        .iter()
        .chain(morph_normals.iter())
        .any(|value| !value.iter().all(|component| component.is_finite()))
    {
        return Err(invalid_cpu_template(
            "glTF primitive morph values must be finite",
        ));
    }
    let morph = (num_targets > 0).then(|| {
        (
            num_targets,
            morph_positions,
            any_normals.then_some(morph_normals),
        )
    });

    validate_primitive_geometry(
        num_vertices,
        &faces,
        normals.as_ref().map(Vec::len),
        uvs.as_ref().map(Vec::len),
        joints.as_ref().map(Vec::len),
        weights.as_ref().map(Vec::len),
        morph.as_ref().map(|(targets, positions, normals)| {
            (*targets, positions.len(), normals.as_ref().map(Vec::len))
        }),
    )
    .map_err(invalid_cpu_template)?;

    let mut mesh = GpuMesh3d::new(positions, faces, normals, uvs, false);
    if let (Some(joints), Some(weights)) = (joints, weights) {
        mesh.set_skin_vertices(SkinVertexData::new(joints, weights));
    }
    if let Some((num_targets, morph_positions, normals)) = morph {
        mesh.set_morph_targets(MorphTargets::new(
            num_targets,
            num_vertices,
            morph_positions,
            normals,
        ));
    }

    Ok(Some(mesh))
}

/// Builds the (joint node weak handles, inverse bind matrices) for a glTF skin.
fn build_skin_parts(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
    node_map: &[SceneNode3d],
) -> Result<
    (
        Vec<std::rc::Weak<RefCell<crate::scene::SceneNodeData3d>>>,
        Vec<Mat4>,
    ),
    gltf::Error,
> {
    let joints: Vec<_> = skin
        .joints()
        .map(|j| node_map[j.index()].downgrade())
        .collect();

    let reader = skin.reader(|b| Some(&buffers[b.index()]));
    let inverse_bind: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(it) => it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; joints.len()],
    };

    if inverse_bind.iter().any(|matrix| !matrix.is_finite()) {
        return Err(invalid_cpu_template(
            "glTF skin inverse-bind matrix is not finite",
        ));
    }

    Ok((joints, inverse_bind))
}

fn validate_material_texture_sources(
    doc: &gltf::Document,
    image_count: usize,
) -> Result<(), gltf::Error> {
    for (material_index, material) in doc.materials().enumerate() {
        let sources_valid = catch_unwind(AssertUnwindSafe(|| {
            [
                material
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .map(|info| info.texture().source().index()),
                material
                    .pbr_metallic_roughness()
                    .metallic_roughness_texture()
                    .map(|info| info.texture().source().index()),
                material
                    .normal_texture()
                    .map(|info| info.texture().source().index()),
                material
                    .occlusion_texture()
                    .map(|info| info.texture().source().index()),
                material
                    .emissive_texture()
                    .map(|info| info.texture().source().index()),
                material
                    .transmission()
                    .and_then(|transmission| transmission.transmission_texture())
                    .map(|info| info.texture().source().index()),
            ]
            .iter()
            .flatten()
            .all(|source| *source < image_count)
        }))
        .map_err(|_| {
            invalid_cpu_template(format!(
                "glTF material {material_index} has an invalid texture source"
            ))
        })?;

        if !sources_valid {
            return Err(invalid_cpu_template(format!(
                "glTF material {material_index} texture source is out of range"
            )));
        }
    }
    Ok(())
}

/// Maps a glTF material's metallic-roughness parameters and textures onto an
/// [`Object3d`].
fn apply_material(
    object: &mut Object3d,
    material: &gltf::Material,
    images: &[DynamicImage],
    cache: &mut HashMap<(usize, bool), Arc<Texture>>,
    prefix: &str,
) {
    let pbr = material.pbr_metallic_roughness();
    let factors = material_factors(material);
    object.set_color(Color::new(
        factors.base_color[0],
        factors.base_color[1],
        factors.base_color[2],
        factors.base_color[3],
    ));
    object.set_metallic(factors.metallic);
    object.set_roughness(factors.roughness);
    object.set_alpha_mode(gltf_alpha_mode(material));
    let extensions = factors.extensions;
    let ef = factors.emissive;
    object.set_emissive(Color::new(
        finite_material_product_or(ef[0], extensions.emissive_strength, 0.0),
        finite_material_product_or(ef[1], extensions.emissive_strength, 0.0),
        finite_material_product_or(ef[2], extensions.emissive_strength, 0.0),
        1.0,
    ));
    if let Some(ior) = extensions.ior {
        object.set_ior(ior);
    }
    if let Some(transmission) = extensions.transmission {
        object.set_transmission(transmission);
    }
    if extensions.has_transmission_texture {
        log::warn!(
            "gltf: KHR_materials_transmission texture is not supported; using transmissionFactor only"
        );
    }

    // Color/emissive textures are sRGB; data textures (normal/MR/AO) are linear.
    if let Some(info) = pbr.base_color_texture() {
        object.set_texture(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            true,
        ));
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        object.set_metallic_roughness_map(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            false,
        ));
    }
    if let Some(nt) = material.normal_texture() {
        object.set_normal_map(get_texture(
            images,
            cache,
            prefix,
            nt.texture().source().index(),
            false,
        ));
    }
    if let Some(ot) = material.occlusion_texture() {
        object.set_ao_map(get_texture(
            images,
            cache,
            prefix,
            ot.texture().source().index(),
            false,
        ));
    }
    if let Some(info) = material.emissive_texture() {
        object.set_emissive_map(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            true,
        ));
    }
}

fn gltf_alpha_mode(material: &gltf::Material<'_>) -> AlphaMode {
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(
            material
                .alpha_cutoff()
                .filter(|cutoff| cutoff.is_finite())
                .unwrap_or(0.5),
        ),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    }
}

fn gltf_alpha_mode_or_default(material: Option<&gltf::Material<'_>>) -> AlphaMode {
    material.map(gltf_alpha_mode).unwrap_or(AlphaMode::Opaque)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaterialExtensionFactors {
    emissive_strength: f32,
    ior: Option<f32>,
    transmission: Option<f32>,
    has_transmission_texture: bool,
}

#[derive(Clone, Copy, Debug)]
struct MaterialFactors {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    extensions: MaterialExtensionFactors,
}

fn material_factors(material: &gltf::Material<'_>) -> MaterialFactors {
    let pbr = material.pbr_metallic_roughness();
    MaterialFactors {
        base_color: finite_material_array_or(pbr.base_color_factor(), [1.0, 1.0, 1.0, 1.0]),
        metallic: finite_material_or(pbr.metallic_factor(), 1.0),
        roughness: finite_material_or(pbr.roughness_factor(), 1.0),
        emissive: finite_material_array_or(material.emissive_factor(), [0.0, 0.0, 0.0]),
        extensions: material_extension_factors(material),
    }
}

fn material_extension_factors(material: &gltf::Material<'_>) -> MaterialExtensionFactors {
    let transmission = material.transmission();
    MaterialExtensionFactors {
        emissive_strength: finite_material_or(material.emissive_strength().unwrap_or(1.0), 1.0),
        ior: material.ior().filter(|value| value.is_finite()),
        transmission: transmission
            .as_ref()
            .map(|value| finite_material_or(value.transmission_factor(), 0.0)),
        has_transmission_texture: transmission
            .and_then(|value| value.transmission_texture())
            .is_some(),
    }
}

fn finite_material_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_material_array_or<const N: usize>(values: [f32; N], fallback: [f32; N]) -> [f32; N] {
    std::array::from_fn(|index| finite_material_or(values[index], fallback[index]))
}

fn finite_material_product_or(value: f32, factor: f32, fallback: f32) -> f32 {
    let product = value * factor;
    if product.is_finite() {
        product
    } else {
        fallback
    }
}

fn material_values_are_finite(material: &CpuMaterial) -> bool {
    material.base_color.iter().all(|value| value.is_finite())
        && material.metallic.is_finite()
        && material.roughness.is_finite()
        && material.emissive.iter().all(|value| value.is_finite())
        && material.extensions.emissive_strength.is_finite()
        && material
            .extensions
            .ior
            .is_none_or(|value| value.is_finite())
        && material
            .extensions
            .transmission
            .is_none_or(|value| value.is_finite())
}

/// Returns the kiss3d texture for a glTF image source in the requested color
/// space, registering and caching it on first use.
fn get_texture(
    images: &[DynamicImage],
    cache: &mut HashMap<(usize, bool), Arc<Texture>>,
    prefix: &str,
    source: usize,
    srgb: bool,
) -> Arc<Texture> {
    if let Some(t) = cache.get(&(source, srgb)) {
        return t.clone();
    }
    // `get_global_manager` takes an `FnMut`, so the `DynamicImage` is handed off
    // through an `Option::take` rather than moved directly out of the closure.
    let mut image = Some(images[source].clone());
    let name = format!("{prefix}::img{source}::srgb{}", srgb as u8);
    let tex = TextureManager::get_global_manager(move |tm| {
        tm.add_image_with_color_space(image.take().unwrap(), &name, srgb)
    });
    cache.insert((source, srgb), tex.clone());
    tex
}

/// Builds an [`AnimationClip`] from a glTF animation, binding each channel to its
/// target node in `node_map`.
fn build_clip(
    anim: &gltf::Animation,
    buffers: &[gltf::buffer::Data],
    node_map: &[SceneNode3d],
) -> Result<AnimationClip, gltf::Error> {
    let name = anim.name().unwrap_or("").to_string();
    let mut channels = Vec::new();

    for channel in anim.channels() {
        let target = node_map[channel.target().node().index()].clone();
        let interp = match channel.sampler().interpolation() {
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        };

        let reader = channel.reader(|b| Some(&buffers[b.index()]));
        let times = reader
            .read_inputs()
            .ok_or_else(|| invalid_cpu_template("glTF animation channel has no input accessor"))?;
        let times: Vec<f32> = times.collect();
        validate_animation_times(&times).map_err(invalid_cpu_template)?;

        let outputs = reader
            .read_outputs()
            .ok_or_else(|| invalid_cpu_template("glTF animation channel has no output accessor"))?;
        let ch = match outputs {
            gltf::animation::util::ReadOutputs::Translations(it) => {
                let values: Vec<Vec3> = it.map(Vec3::from_array).collect();
                validate_animation_vec3(&values).map_err(invalid_cpu_template)?;
                validate_animation_output_cardinality(
                    times.len(),
                    interp == Interpolation::CubicSpline,
                    values.len(),
                    1,
                )
                .map_err(invalid_cpu_template)?;
                AnimationChannel::translation(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::Scales(it) => {
                let values: Vec<Vec3> = it.map(Vec3::from_array).collect();
                validate_animation_vec3(&values).map_err(invalid_cpu_template)?;
                validate_animation_output_cardinality(
                    times.len(),
                    interp == Interpolation::CubicSpline,
                    values.len(),
                    1,
                )
                .map_err(invalid_cpu_template)?;
                AnimationChannel::scale(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::Rotations(rot) => {
                let values: Vec<Quat> = rot.into_f32().map(Quat::from_array).collect();
                validate_animation_quat(&values, interp == Interpolation::CubicSpline)
                    .map_err(invalid_cpu_template)?;
                validate_animation_output_cardinality(
                    times.len(),
                    interp == Interpolation::CubicSpline,
                    values.len(),
                    1,
                )
                .map_err(invalid_cpu_template)?;
                AnimationChannel::rotation(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::MorphTargetWeights(weights) => {
                let values: Vec<f32> = weights.into_f32().collect();
                validate_animation_scalars(&values).map_err(invalid_cpu_template)?;
                let num_keys = times.len();
                let num_targets =
                    morph_target_count(&channel.target().node()).ok_or_else(|| {
                        invalid_cpu_template("glTF morph animation target has no morph targets")
                    })?;
                validate_animation_output_cardinality(
                    num_keys,
                    interp == Interpolation::CubicSpline,
                    values.len(),
                    num_targets,
                )
                .map_err(invalid_cpu_template)?;
                AnimationChannel::morph_weights(target, times, values, num_targets, interp)
            }
        };
        channels.push(ch);
    }

    Ok(AnimationClip::new(name, channels))
}

fn morph_target_count(node: &gltf::Node<'_>) -> Option<usize> {
    let mesh = node.mesh()?;
    let count = mesh
        .primitives()
        .map(|primitive| primitive.morph_targets().len())
        .max()?;
    (count > 0).then_some(count)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_cpu_from_slice, decode_cpu_from_slice_with_limits, decode_cpu_with_limits,
        decode_image_bytes, extract_static_template, finite_material_product_or, gltf_alpha_mode,
        gltf_alpha_mode_or_default, image_format, import_images_with_budget, joint_edges,
        load_from_slice, material_extension_factors, read_data_uri, triangle_faces_from_indices,
        validate_animation_output_cardinality, validate_animation_quat, validate_animation_times,
        validate_animation_vec3, validate_cpu_template, validate_document_decode_limits,
        validate_material_texture_sources, validate_parent_graph, CpuAnimationOutput, CpuImage,
        CpuMaterial, CpuMesh, CpuMorphTargets, CpuNode, CpuPrimitive, CpuSkin, DecodedImageBudget,
        EncodedResourceBudget, GltfAssetTemplate, GltfDecodeLimits, Interpolation,
        MaterialExtensionFactors, MAX_GLTF_IMAGE_DIMENSION,
    };
    use crate::scene::AlphaMode;
    use crate::scene::SceneNode3d;
    use crate::window::{CanvasSetup, Window};
    use glamx::{Quat, Vec3};
    use image::{DynamicImage, GenericImageView, ImageFormat};
    use std::collections::HashSet;
    use std::convert::TryInto;
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn first_material(json: &str) -> gltf::Gltf {
        gltf::Gltf::from_slice(json.as_bytes()).expect("test glTF should parse")
    }

    fn assert_send<T: Send>() {}

    #[test]
    fn static_template_types_are_send() {
        assert_send::<CpuPrimitive>();
        assert_send::<CpuMesh>();
        assert_send::<CpuNode>();
        assert_send::<CpuImage>();
        assert_send::<CpuMaterial>();
        assert_send::<CpuMorphTargets>();
        assert_send::<CpuSkin>();
        assert_send::<CpuAnimationOutput>();
        assert_send::<GltfAssetTemplate>();
    }

    #[test]
    fn animation_output_cardinality_matches_interpolation_contract() {
        assert!(validate_animation_output_cardinality(2, false, 2, 1).is_ok());
        assert!(validate_animation_output_cardinality(2, true, 6, 1).is_ok());
        assert!(validate_animation_output_cardinality(2, false, 6, 3).is_ok());
        assert!(validate_animation_output_cardinality(2, false, 1, 1).is_err());
        assert!(validate_animation_output_cardinality(2, true, 2, 1).is_err());
        assert!(validate_animation_output_cardinality(2, false, 2, 2).is_err());
    }

    fn fixture_template(bytes: &[u8]) -> GltfAssetTemplate {
        decode_cpu_from_slice(bytes).expect("template should validate")
    }

    fn source_limits(max_source_bytes: u64, max_json_bytes: u64) -> GltfDecodeLimits {
        GltfDecodeLimits {
            max_source_bytes,
            max_json_bytes,
            ..GltfDecodeLimits::default()
        }
    }

    const ANIMATION_FIXTURE: &str = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 0 }],
        "bufferViews": [{ "buffer": 0, "byteLength": 0 }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR" },
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" }
        ],
        "nodes": [{}],
        "animations": [{
            "samplers": [{ "input": 0, "output": 1 }],
            "channels": [{
                "sampler": 0,
                "target": { "node": 0, "path": "translation" }
            }]
        }]
    }"#;

    const CUBIC_ANIMATION_FIXTURE: &str = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 0 }],
        "bufferViews": [{ "buffer": 0, "byteLength": 0 }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR" },
            { "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3" }
        ],
        "nodes": [{}],
        "animations": [{
            "samplers": [{
                "input": 0,
                "output": 1,
                "interpolation": "CUBICSPLINE"
            }],
            "channels": [{
                "sampler": 0,
                "target": { "node": 0, "path": "translation" }
            }]
        }]
    }"#;

    fn assert_invalid_data(error: gltf::Error) {
        assert!(matches!(
            error,
            gltf::Error::Io(error) if error.kind() == std::io::ErrorKind::InvalidData
        ));
    }

    fn assert_invalid_data_message(error: gltf::Error, expected: &[&str]) {
        match error {
            gltf::Error::Io(error) => {
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                let message = error.to_string();
                for part in expected {
                    assert!(message.contains(part), "{:?} missing {:?}", message, part);
                }
            }
            other => panic!("expected InvalidData, got {:?}", other),
        }
    }

    #[test]
    fn gltf_decode_limits_default_matches_renderer_source_contract() {
        let limits = GltfDecodeLimits::default();

        assert_eq!(limits.max_source_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.max_json_bytes, 64 * 1024 * 1024);
        assert_eq!(limits.max_declared_buffer_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.max_images, 256);
        assert_eq!(limits.max_accessor_elements, 100_000_000);
        assert_eq!(limits.max_accessor_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.max_external_resource_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.max_nodes, 262_144);
        assert_eq!(limits.max_skins, 65_536);
        assert_eq!(limits.max_skin_joint_references, 262_144);
        assert_eq!(limits.max_meshes, 262_144);
        assert_eq!(limits.max_primitives, 262_144);
        assert_eq!(limits.max_materials, 262_144);
        assert_eq!(limits.max_animations, 64);
        assert_eq!(limits.max_animation_channels, 65_536);
        assert_eq!(limits.max_animation_key_values, 1_000_000);
    }

    #[test]
    fn decode_cpu_limits_reject_template_collection_counts_before_import() {
        let primitive_fixture = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 0 }],
            "bufferViews": [{ "buffer": 0, "byteLength": 0 }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 0,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [0, 0, 0]
            }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }]
        }"#;
        let cases: &[(&str, &str, fn(&mut GltfDecodeLimits))] = &[
            (
                r#"{ "asset": { "version": "2.0" }, "nodes": [{}] }"#,
                "node count",
                |limits| limits.max_nodes = 0,
            ),
            (
                r#"{
                    "asset": { "version": "2.0" },
                    "nodes": [{}],
                    "skins": [{ "joints": [0] }]
                }"#,
                "skin count",
                |limits| limits.max_skins = 0,
            ),
            (
                r#"{
                    "asset": { "version": "2.0" },
                    "nodes": [{}],
                    "skins": [{ "joints": [0] }]
                }"#,
                "skin joint references",
                |limits| limits.max_skin_joint_references = 0,
            ),
            (
                r#"{
                    "asset": { "version": "2.0" },
                    "meshes": [{ "primitives": [] }]
                }"#,
                "mesh count",
                |limits| limits.max_meshes = 0,
            ),
            (primitive_fixture, "primitive count", |limits| {
                limits.max_primitives = 0
            }),
            (
                r#"{
                    "asset": { "version": "2.0" },
                    "materials": [{}]
                }"#,
                "material count",
                |limits| limits.max_materials = 0,
            ),
        ];

        for (json, field, configure) in cases {
            let mut limits = source_limits(json.len() as u64, json.len() as u64);
            configure(&mut limits);
            let error = decode_cpu_from_slice_with_limits(json.as_bytes(), &limits)
                .expect_err("template collection limit must reject before import");
            assert_invalid_data_message(error, &[field, "limit"]);
        }
    }

    #[test]
    fn decode_cpu_limits_reject_animation_collections_and_key_values() {
        let mut limits = source_limits(
            ANIMATION_FIXTURE.len() as u64,
            ANIMATION_FIXTURE.len() as u64,
        );
        limits.max_animations = 0;
        let error = decode_cpu_from_slice_with_limits(ANIMATION_FIXTURE.as_bytes(), &limits)
            .expect_err("animation count limit must reject before import");
        assert_invalid_data_message(error, &["animation count", "1", "0"]);

        let mut limits = source_limits(
            ANIMATION_FIXTURE.len() as u64,
            ANIMATION_FIXTURE.len() as u64,
        );
        limits.max_animation_channels = 0;
        let error = decode_cpu_from_slice_with_limits(ANIMATION_FIXTURE.as_bytes(), &limits)
            .expect_err("animation channel limit must reject before import");
        assert_invalid_data_message(error, &["animation channel count", "1", "0"]);

        let mut limits = source_limits(
            ANIMATION_FIXTURE.len() as u64,
            ANIMATION_FIXTURE.len() as u64,
        );
        limits.max_animation_key_values = 1;
        let error = decode_cpu_from_slice_with_limits(ANIMATION_FIXTURE.as_bytes(), &limits)
            .expect_err("animation input value limit must reject before import");
        assert_invalid_data_message(error, &["animation input key values", "2", "1"]);

        let mut limits = source_limits(
            CUBIC_ANIMATION_FIXTURE.len() as u64,
            CUBIC_ANIMATION_FIXTURE.len() as u64,
        );
        limits.max_animation_key_values = 2;
        let error = decode_cpu_from_slice_with_limits(CUBIC_ANIMATION_FIXTURE.as_bytes(), &limits)
            .expect_err("cubic animation output value limit must reject before import");
        assert_invalid_data_message(error, &["animation output key values", "6", "2"]);
    }

    #[test]
    fn document_animation_reference_panics_become_invalid_data() {
        let malformed_sampler = r#"{
            "asset": { "version": "2.0" },
            "nodes": [{}],
            "animations": [{
                "samplers": [],
                "channels": [{
                    "sampler": 0,
                    "target": { "node": 0, "path": "translation" }
                }]
            }]
        }"#;
        let document = gltf::Gltf::from_slice_without_validation(malformed_sampler.as_bytes())
            .expect("malformed sampler fixture should deserialize")
            .document;
        let error = validate_document_decode_limits(&document, &GltfDecodeLimits::default())
            .expect_err("invalid sampler reference must reject without panicking");
        assert_invalid_data_message(error, &["invalid sampler reference"]);

        let malformed_target = ANIMATION_FIXTURE.replace("\"node\": 0", "\"node\": 9");
        let limits = source_limits(malformed_target.len() as u64, malformed_target.len() as u64);
        let error = decode_cpu_from_slice_with_limits(malformed_target.as_bytes(), &limits)
            .expect_err("invalid target node reference must reject without panicking");
        assert_invalid_data_message(error, &["invalid target node reference"]);
    }

    #[test]
    fn document_skin_joint_reference_out_of_range_is_invalid_data() {
        let malformed_skin = r#"{
            "asset": { "version": "2.0" },
            "nodes": [{}],
            "skins": [{ "joints": [3] }]
        }"#;
        let document = gltf::Gltf::from_slice_without_validation(malformed_skin.as_bytes())
            .expect("malformed skin fixture should deserialize")
            .document;
        let error = validate_document_decode_limits(&document, &GltfDecodeLimits::default())
            .expect_err("out-of-range skin joint must reject before import");
        assert_invalid_data_message(error, &["skin 0", "joint 0", "node 3"]);
    }

    #[test]
    fn decode_cpu_limits_reject_declared_buffer_bytes_before_import() {
        let json = br#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 17 }]
        }"#;
        let source_bytes = json.len() as u64;
        let mut limits = source_limits(source_bytes, source_bytes);
        limits.max_declared_buffer_bytes = 16;

        let error = decode_cpu_from_slice_with_limits(json, &limits)
            .expect_err("declared buffer byte limit must reject before import");
        assert_invalid_data_message(error, &["declared buffer bytes", "17", "16"]);
    }

    #[test]
    fn decode_cpu_limits_reject_external_data_uri_aggregate_before_decode() {
        let image = tiny_png();
        let payload = base64::encode(&image);
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "uri": "data:application/octet-stream;base64,{payload}", "byteLength": {} }}],
                "images": [{{ "uri": "data:image/png;base64,{payload}" }}]
            }}"#,
            image.len()
        );
        let source_bytes = json.len() as u64;
        let mut limits = source_limits(source_bytes, source_bytes);
        limits.max_external_resource_bytes = image.len() as u64 + 1;

        let error = decode_cpu_from_slice_with_limits(json.as_bytes(), &limits)
            .expect_err("aggregate data URI budget must reject before image decode");
        assert_invalid_data_message(error, &["external encoded resource", "limit"]);
    }

    #[test]
    fn decode_cpu_data_uri_budget_accepts_exact_decoded_boundary() {
        let data = tiny_png();
        let payload = base64::encode(&data);
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "uri": "data:application/octet-stream;base64,{payload}", "byteLength": {} }}]
            }}"#,
            data.len()
        );
        let source_bytes = json.len() as u64;
        let mut exact_limits = source_limits(source_bytes, source_bytes);
        exact_limits.max_external_resource_bytes = data.len() as u64;
        decode_cpu_from_slice_with_limits(json.as_bytes(), &exact_limits)
            .expect("exact decoded data URI budget must pass");

        let mut short_limits = exact_limits;
        short_limits.max_external_resource_bytes -= 1;
        let error = decode_cpu_from_slice_with_limits(json.as_bytes(), &short_limits)
            .expect_err("one byte below decoded data URI budget must reject");
        assert_invalid_data_message(error, &["external encoded resource", "limit"]);
    }

    #[test]
    fn data_uri_invalid_base64_shape_keeps_base64_error_contract() {
        let error = read_data_uri("A=", &mut EncodedResourceBudget::new(0))
            .expect_err("invalid base64 padding must fail as a base64 error");
        assert!(matches!(error, gltf::Error::Base64(_)));
    }

    #[test]
    fn decode_cpu_limits_reject_relative_buffer_and_image_aggregate() {
        let root = temp_image_path().with_extension("resources");
        fs::create_dir_all(&root).expect("create resource directory");
        let buffer = [1_u8, 2, 3];
        let image = tiny_png();
        fs::write(root.join("buffer.bin"), buffer).expect("write external buffer");
        fs::write(root.join("image.png"), &image).expect("write external image");
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "uri": "buffer.bin", "byteLength": {} }}],
                "images": [{{ "uri": "image.png" }}]
            }}"#,
            buffer.len()
        );
        let path = root.join("fixture.gltf");
        fs::write(&path, json.as_bytes()).expect("write external URI fixture");
        let mut limits = source_limits(json.len() as u64, json.len() as u64);
        limits.max_external_resource_bytes = (buffer.len() + image.len() - 1) as u64;

        let error = decode_cpu_with_limits(&path, &limits)
            .expect_err("relative buffer and image aggregate must reject");
        assert_invalid_data_message(error, &["external encoded resource", "limit"]);

        fs::remove_dir_all(root).expect("remove resource directory");
    }

    #[test]
    fn decode_cpu_slice_limits_reject_external_filesystem_reference() {
        let json = br#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "uri": "buffer.bin", "byteLength": 1 }]
        }"#;
        let source_bytes = json.len() as u64;
        let limits = source_limits(source_bytes, source_bytes);

        let error = decode_cpu_from_slice_with_limits(json, &limits)
            .expect_err("slice decode must reject external filesystem references");
        assert!(matches!(error, gltf::Error::ExternalReferenceInSliceImport));
    }

    #[test]
    fn decode_cpu_path_preserves_relative_file_and_data_uri_sources() {
        let root = temp_image_path().with_extension("parity");
        fs::create_dir_all(&root).expect("create resource directory");
        let buffer = [1_u8, 2, 3];
        let image = tiny_png();
        fs::write(root.join("buffer.bin"), buffer).expect("write external buffer");
        fs::write(root.join("relative.png"), &image).expect("write relative image");
        let file_uri = format!(
            "file://{}",
            root.join("file.png").to_string_lossy().replace('\\', "/")
        );
        fs::write(root.join("file.png"), &image).expect("write file URI image");
        let payload = base64::encode(&image);
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "uri": "buffer.bin", "byteLength": {} }}],
                "images": [
                    {{ "uri": "relative.png" }},
                    {{ "uri": "{}" }},
                    {{ "uri": "data:image/png;base64,{payload}" }}
                ]
            }}"#,
            buffer.len(),
            file_uri
        );
        let path = root.join("fixture.gltf");
        fs::write(&path, json.as_bytes()).expect("write URI parity fixture");
        let limits = source_limits(json.len() as u64, json.len() as u64);

        decode_cpu_with_limits(&path, &limits).expect("relative/file/data URI sources decode");

        fs::remove_dir_all(root).expect("remove resource directory");
    }

    #[test]
    fn decode_cpu_limits_reject_image_count_before_image_import() {
        let images = (0..257)
            .map(|_| r#"{"uri":"data:,"}"#)
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"asset":{{"version":"2.0"}},"images":[{images}]}}"#);
        let source_bytes = json.len() as u64;
        let mut limits = source_limits(source_bytes, source_bytes);
        limits.max_images = 256;

        let error = decode_cpu_from_slice_with_limits(json.as_bytes(), &limits)
            .expect_err("image count limit must reject before image import");
        assert_invalid_data_message(error, &["image count", "257", "256"]);
    }

    #[test]
    fn decode_cpu_limits_reject_accessor_element_and_byte_totals_before_import() {
        let json = br#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 0 }],
            "bufferViews": [{ "buffer": 0, "byteLength": 0 }],
            "accessors": [{ "bufferView": 0, "count": 2, "componentType": 5126, "type": "VEC4" }]
        }"#;
        let source_bytes = json.len() as u64;

        let mut element_limits = source_limits(source_bytes, source_bytes);
        element_limits.max_accessor_elements = 1;
        let element_error = decode_cpu_from_slice_with_limits(json, &element_limits)
            .expect_err("accessor element limit must reject before import");
        assert_invalid_data_message(element_error, &["accessor elements", "2", "1"]);

        let mut byte_limits = source_limits(source_bytes, source_bytes);
        byte_limits.max_accessor_elements = 2;
        byte_limits.max_accessor_bytes = 31;
        let byte_error = decode_cpu_from_slice_with_limits(json, &byte_limits)
            .expect_err("accessor byte limit must reject before import");
        assert_invalid_data_message(byte_error, &["accessor bytes", "32", "31"]);
    }

    #[test]
    fn decode_cpu_limits_reject_oversized_json_path_before_parser() {
        let path = temp_image_path().with_extension("gltf");
        let json = br#"{"asset":{"version":"2.0"}}"#;
        fs::write(&path, json).expect("write JSON fixture");

        let source_bytes = json.len() as u64;
        assert!(decode_cpu_with_limits(&path, &source_limits(source_bytes, source_bytes)).is_ok());

        let source_error =
            decode_cpu_with_limits(&path, &source_limits(source_bytes - 1, source_bytes))
                .expect_err("source limit must reject before parsing");
        assert_invalid_data(source_error);
        let json_error =
            decode_cpu_with_limits(&path, &source_limits(source_bytes, source_bytes - 1))
                .expect_err("JSON limit must reject before parsing");
        assert_invalid_data(json_error);

        fs::remove_file(path).expect("remove JSON fixture");
    }

    #[test]
    fn decode_cpu_slice_limits_reject_oversized_json_before_parser() {
        let json = br#"{"asset":{"version":"2.0"}}"#;
        let source_bytes = json.len() as u64;

        assert!(decode_cpu_from_slice_with_limits(
            json,
            &source_limits(source_bytes, source_bytes),
        )
        .is_ok());
        let source_error =
            decode_cpu_from_slice_with_limits(json, &source_limits(source_bytes - 1, source_bytes))
                .expect_err("source limit must reject before parsing");
        assert_invalid_data(source_error);
        let json_error =
            decode_cpu_from_slice_with_limits(json, &source_limits(source_bytes, source_bytes - 1))
                .expect_err("JSON limit must reject before parsing");
        assert_invalid_data(json_error);
    }

    #[test]
    fn decode_cpu_slice_limits_reject_oversized_declared_glb_length() {
        let mut glb = json_glb(r#"{"asset":{"version":"2.0"}}"#);
        let source_bytes = glb.len() as u64;
        let declared_length = (source_bytes + 1) as u32;
        glb[8..12].copy_from_slice(&declared_length.to_le_bytes());

        let error =
            decode_cpu_from_slice_with_limits(&glb, &source_limits(source_bytes, source_bytes))
                .expect_err("declared GLB length must reject before parsing");
        assert_invalid_data(error);
    }

    #[test]
    fn decode_cpu_slice_limits_reject_short_declared_glb_length() {
        let mut glb = json_glb(r#"{"asset":{"version":"2.0"}}"#);
        let source_bytes = glb.len() as u64;
        let declared_length = (source_bytes - 1) as u32;
        glb[8..12].copy_from_slice(&declared_length.to_le_bytes());

        let error =
            decode_cpu_from_slice_with_limits(&glb, &source_limits(source_bytes, source_bytes))
                .expect_err("short declared GLB length must reject trailing data");
        assert_invalid_data(error);
    }

    #[test]
    fn decode_cpu_limits_reject_oversized_glb_json_path_before_parser() {
        let path = temp_image_path().with_extension("glb");
        let glb = json_glb(r#"{"asset":{"version":"2.0"}}"#);
        let source_bytes = glb.len() as u64;
        let json_bytes = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as u64;
        fs::write(&path, &glb).expect("write GLB fixture");

        assert!(decode_cpu_with_limits(&path, &source_limits(source_bytes, json_bytes)).is_ok());
        let error = decode_cpu_with_limits(&path, &source_limits(source_bytes, json_bytes - 1))
            .expect_err("GLB JSON limit must reject before parsing");
        assert_invalid_data(error);

        fs::remove_file(path).expect("remove GLB fixture");
    }

    #[test]
    fn decode_cpu_slice_limits_reject_oversized_glb_json_chunk() {
        let mut glb = json_glb(r#"{"asset":{"version":"2.0"}}"#);
        let source_bytes = glb.len() as u64;
        let json_length = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as u64;
        assert!(
            decode_cpu_from_slice_with_limits(&glb, &source_limits(source_bytes, json_length),)
                .is_ok()
        );

        let oversized_json_length = (json_length + 1) as u32;
        glb[12..16].copy_from_slice(&oversized_json_length.to_le_bytes());
        let error =
            decode_cpu_from_slice_with_limits(&glb, &source_limits(source_bytes, json_length))
                .expect_err("JSON chunk limit must reject before parsing");
        assert_invalid_data(error);
    }

    fn decode_single_material(alpha_fields: &str) -> CpuMaterial {
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "materials": [{{ {alpha_fields} }}]
            }}"#
        );
        let template = decode_cpu_from_slice(&json_glb(&json)).expect("material should decode");
        template
            .materials
            .into_iter()
            .next()
            .expect("fixture should contain one material")
    }

    #[test]
    fn cpu_material_alpha_opaque_maps_to_opaque() {
        let material = decode_single_material(r#""alphaMode": "OPAQUE""#);

        assert_eq!(material.alpha_mode, AlphaMode::Opaque);
        assert_eq!(material.alpha_cutoff, None);
    }

    #[test]
    fn cpu_material_alpha_mask_preserves_cutoff_and_default() {
        let explicit = decode_single_material(r#""alphaMode": "MASK", "alphaCutoff": 0.25"#);
        assert_eq!(explicit.alpha_mode, AlphaMode::Mask(0.25));
        assert_eq!(explicit.alpha_cutoff, Some(0.25));

        let default = decode_single_material(r#""alphaMode": "MASK""#);
        assert_eq!(default.alpha_mode, AlphaMode::Mask(0.5));
        assert_eq!(default.alpha_cutoff, Some(0.5));
    }

    #[test]
    fn cpu_material_alpha_blend_maps_to_blend() {
        let material = decode_single_material(r#""alphaMode": "BLEND""#);

        assert_eq!(material.alpha_mode, AlphaMode::Blend);
        assert_eq!(material.alpha_cutoff, None);
    }

    fn first_mesh_info(node: &SceneNode3d) -> Option<(usize, (u32, u32))> {
        let data = node.data();
        if let Some(object) = data.object() {
            return Some((
                object.mesh().borrow().num_pts(),
                object.data().texture().size,
            ));
        }
        data.children().iter().find_map(first_mesh_info)
    }

    #[test]
    fn headless_box_textured_runtime_load_releases_resources() {
        pollster::block_on(async {
            let window = Window::new_headless_with_setup(64, 64, CanvasSetup::default()).await;
            {
                let model = load_from_slice(include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/media/gltf/BoxTextured.glb"
                )))
                .expect("BoxTextured should load through the runtime loader");

                let scene = model.root.clone();

                let (num_points, (width, height)) = first_mesh_info(&model.root)
                    .expect("BoxTextured should produce a mesh object node");
                assert!(num_points > 0);
                assert!(
                    width > 1 && height > 1,
                    "runtime texture must not be the 1x1 placeholder: {width}x{height}",
                    width = width,
                    height = height
                );

                drop(model);
                drop(scene);
            }

            drop(window);
        });
    }

    #[test]
    fn cpu_decode_preserves_fox_skin_and_animation_metadata() {
        let template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/Fox.glb"
        )));

        assert_eq!(template.skins.len(), 1);
        assert_eq!(template.skins[0].joints.len(), 24);
        assert_eq!(template.skins[0].inverse_bind.len(), 24);
        assert_eq!(
            template
                .nodes
                .iter()
                .filter(|node| node.skin == Some(0))
                .count(),
            1
        );

        let primitive = &template.meshes[0].primitives[0];
        let joints = primitive
            .skin_joints
            .as_ref()
            .expect("Fox primitive should carry JOINTS_0");
        let weights = primitive
            .skin_weights
            .as_ref()
            .expect("Fox primitive should carry WEIGHTS_0");
        assert_eq!(joints.len(), primitive.positions.len());
        assert_eq!(weights.len(), primitive.positions.len());

        assert_eq!(template.animations.len(), 3);
        assert!(template
            .animations
            .iter()
            .all(|clip| clip.channels.len() == 21));
        for clip in &template.animations {
            assert!(!clip.name.is_empty());
            for channel in &clip.channels {
                assert!(channel.target_node < template.nodes.len());
                assert!(!channel.times.is_empty());
                assert!(channel.times.iter().all(|time| time.is_finite()));
                assert!(matches!(
                    channel.interpolation,
                    Interpolation::Linear | Interpolation::Step | Interpolation::CubicSpline
                ));
                match &channel.output {
                    CpuAnimationOutput::Translation(values) | CpuAnimationOutput::Scale(values) => {
                        assert!(!values.is_empty());
                    }
                    CpuAnimationOutput::Rotation(values) => assert!(!values.is_empty()),
                    CpuAnimationOutput::MorphWeights {
                        values,
                        num_targets,
                    } => {
                        assert!(*num_targets > 0);
                        assert!(!values.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_decode_preserves_morph_and_animation_cardinality() {
        let template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/AnimatedMorphCube.glb"
        )));

        let primitive = &template.meshes[0].primitives[0];
        let morph = primitive.morph.as_ref().expect("morph targets expected");
        assert_eq!(morph.num_targets, 2);
        assert_eq!(
            morph.positions.len(),
            morph.num_targets * primitive.positions.len()
        );
        if let Some(normals) = &morph.normals {
            assert_eq!(normals.len(), morph.num_targets * primitive.positions.len());
        }
        assert_eq!(template.animations.len(), 1);
        assert_eq!(template.animations[0].channels.len(), 1);
        let channel = &template.animations[0].channels[0];
        assert_eq!(channel.times.len(), 127);
        assert!(matches!(
            &channel.output,
            CpuAnimationOutput::MorphWeights { num_targets: 2, .. }
        ));
    }

    #[test]
    fn cpu_decode_preserves_box_animation_channel_count() {
        let template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/BoxAnimated.glb"
        )));
        assert_eq!(template.animations.len(), 1);
        assert_eq!(template.animations[0].channels.len(), 2);
        assert!(template.animations[0]
            .channels
            .iter()
            .all(|channel| !channel.times.is_empty()));
    }

    #[test]
    fn cpu_template_validation_rejects_out_of_range_material() {
        let mut template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/BoxTextured.glb"
        )));
        template.meshes[0].primitives[0].material_index = Some(template.materials.len());
        let error = validate_cpu_template(&template).expect_err("invalid material must reject");
        assert!(error.to_string().contains("material index"));
    }

    #[test]
    fn cpu_template_validation_rejects_non_finite_inverse_bind() {
        let mut template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/Fox.glb"
        )));
        template.skins[0].inverse_bind[0][0][0] = f32::NAN;

        let error = validate_cpu_template(&template)
            .expect_err("non-finite inverse-bind matrix must reject");
        assert!(error.to_string().contains("inverse-bind"));
    }

    #[test]
    fn cpu_template_validation_rejects_non_finite_primitive_values() {
        let mut template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/Fox.glb"
        )));
        template.meshes[0].primitives[0].positions[0][0] = f32::NAN;

        let error = validate_cpu_template(&template)
            .expect_err("non-finite primitive position must reject");
        assert!(error.to_string().contains("positions"));
    }

    #[test]
    fn material_numeric_values_fail_closed_before_gpu_upload() {
        assert_eq!(finite_material_product_or(0.5, 2.0, 0.0), 1.0);
        assert_eq!(finite_material_product_or(f32::MAX, f32::MAX, 0.0), 0.0);

        let mut template = fixture_template(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/BoxTextured.glb"
        )));
        template.materials[0].base_color[0] = f32::INFINITY;
        let error =
            validate_cpu_template(&template).expect_err("non-finite material factor must reject");
        assert!(error.to_string().contains("material factors"));
    }

    #[test]
    fn material_texture_source_validation_covers_all_slots_and_boundary() {
        let gltf = gltf::Gltf::from_slice(
            br#"{
                "asset": { "version": "2.0" },
                "extensionsUsed": ["KHR_materials_transmission"],
                "extensionsRequired": ["KHR_materials_transmission"],
                "images": [{ "uri": "unused.png" }],
                "textures": [{ "source": 0 }],
                "materials": [{
                    "pbrMetallicRoughness": {
                        "baseColorTexture": { "index": 0 },
                        "metallicRoughnessTexture": { "index": 0 }
                    },
                    "normalTexture": { "index": 0 },
                    "occlusionTexture": { "index": 0 },
                    "emissiveTexture": { "index": 0 },
                    "extensions": {
                        "KHR_materials_transmission": {
                            "transmissionTexture": { "index": 0 }
                        }
                    }
                }]
            }"#,
        )
        .expect("malformed source index should parse as a document");
        assert!(validate_material_texture_sources(&gltf.document, 1).is_ok());
        let error = validate_material_texture_sources(&gltf.document, 0)
            .expect_err("out-of-range texture source must reject before GPU registration");
        assert!(error.to_string().contains("texture source"));
    }

    #[test]
    fn cpu_decode_rejects_singular_matrix_node_transform() {
        let error = decode_cpu_from_slice(
            br#"{
                "asset": { "version": "2.0" },
                "nodes": [{
                    "matrix": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                }],
                "scenes": [{ "nodes": [0] }],
                "scene": 0
            }"#,
        )
        .expect_err("singular node matrix must reject");
        assert!(error.to_string().contains("node rotation"));
    }

    #[test]
    fn cpu_decode_materializes_box_textured_image_and_material_contract() {
        let template = decode_cpu_from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/BoxTextured.glb"
        )))
        .expect("template should validate");

        assert_eq!(template.images.len(), 1);
        let (width, height) = template.images[0].decoded.dimensions();
        assert!(width > 0 && height > 0);
        assert!(!template.images[0].decoded.as_bytes().is_empty());

        assert_eq!(template.materials.len(), 1);
        let material = &template.materials[0];
        assert_eq!(material.base_color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.roughness, 1.0);
        assert_eq!(material.emissive, [0.0, 0.0, 0.0]);
        assert_eq!(material.alpha_mode, AlphaMode::Opaque);
        assert_eq!(material.alpha_cutoff, None);
        assert_eq!(
            material.extensions,
            MaterialExtensionFactors {
                emissive_strength: 1.0,
                ior: None,
                transmission: None,
                has_transmission_texture: false,
            }
        );
        assert_eq!(material.base_color_image, Some(0));
        assert_eq!(material.metallic_roughness_image, None);
        assert_eq!(material.normal_image, None);
        assert_eq!(material.occlusion_image, None);
        assert_eq!(material.emissive_image, None);

        let primitive_materials: Vec<_> = template
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives
                    .iter()
                    .map(|primitive| primitive.material_index)
            })
            .collect();
        assert_eq!(primitive_materials, vec![Some(0)]);
    }

    #[test]
    fn static_template_captures_scene_trs_roots_and_geometry() {
        let gltf = gltf::Gltf::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/gltf/BoxTextured.glb"
        )))
        .expect("fixture glTF should parse");
        let gltf::Gltf { document, blob } = gltf;
        let buffers = gltf::import_buffers(&document, None, blob).expect("fixture buffers");
        let template = extract_static_template(&document, &buffers).expect("static template");

        assert_eq!(template.nodes.len(), document.nodes().count());
        assert_eq!(template.meshes.len(), document.meshes().count());
        let expected_roots: Vec<_> = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .map(|scene| scene.nodes().map(|node| node.index()).collect())
            .unwrap_or_default();
        assert_eq!(template.default_scene_roots, expected_roots);

        for node in document.nodes() {
            let (translation, rotation, scale) = node.transform().decomposed();
            let cpu = &template.nodes[node.index()];
            assert_eq!(cpu.parent, node_parent(&document, node.index()));
            assert_eq!(
                cpu.children,
                node.children()
                    .map(|child| child.index())
                    .collect::<Vec<_>>()
            );
            assert_eq!(cpu.mesh, node.mesh().map(|mesh| mesh.index()));
            let expected_weights = node
                .weights()
                .or_else(|| node.mesh().and_then(|mesh| mesh.weights()))
                .map(|weights| weights.to_vec())
                .unwrap_or_default();
            assert_eq!(cpu.weights, expected_weights);
            assert_eq!(cpu.translation, translation);
            assert_eq!(cpu.rotation, rotation);
            assert_eq!(cpu.scale, scale);
        }
        for mesh in document.meshes() {
            assert_eq!(
                template.meshes[mesh.index()].weights,
                mesh.weights().unwrap_or(&[])
            );
        }

        let mut primitive_count = 0;
        for mesh in &template.meshes {
            for primitive in &mesh.primitives {
                primitive_count += 1;
                assert!(!primitive.positions.is_empty());
                assert!(!primitive.faces.is_empty());
                assert!(primitive
                    .faces
                    .iter()
                    .flatten()
                    .all(|index| (*index as usize) < primitive.positions.len()));
                if let Some(normals) = &primitive.normals {
                    assert_eq!(normals.len(), primitive.positions.len());
                }
                if let Some(uvs) = &primitive.uvs {
                    assert_eq!(uvs.len(), primitive.positions.len());
                }
            }
        }
        assert!(primitive_count > 0);
    }

    fn node_parent(document: &gltf::Document, index: usize) -> Option<usize> {
        document
            .nodes()
            .find(|node| node.children().any(|child| child.index() == index))
            .map(|node| node.index())
    }

    #[test]
    fn material_extension_factors_default_to_core_gltf_values() {
        let gltf = first_material(
            r#"{
                "asset": { "version": "2.0" },
                "materials": [{}]
            }"#,
        );
        let material = gltf.materials().next().unwrap();

        assert_eq!(
            material_extension_factors(&material),
            MaterialExtensionFactors {
                emissive_strength: 1.0,
                ior: None,
                transmission: None,
                has_transmission_texture: false,
            }
        );
    }

    #[test]
    fn material_extension_factors_read_lightweight_khr_materials() {
        let gltf = first_material(
            r#"{
                "asset": { "version": "2.0" },
                "extensionsUsed": [
                    "KHR_materials_emissive_strength",
                    "KHR_materials_ior",
                    "KHR_materials_transmission"
                ],
                "extensionsRequired": [
                    "KHR_materials_emissive_strength",
                    "KHR_materials_ior",
                    "KHR_materials_transmission"
                ],
                "images": [{ "uri": "unused.png" }],
                "textures": [{ "source": 0 }],
                "materials": [{
                    "extensions": {
                        "KHR_materials_emissive_strength": { "emissiveStrength": 3.0 },
                        "KHR_materials_ior": { "ior": 1.33 },
                        "KHR_materials_transmission": {
                            "transmissionFactor": 0.75,
                            "transmissionTexture": { "index": 0 }
                        }
                    }
                }]
            }"#,
        );
        let material = gltf.materials().next().unwrap();

        assert_eq!(
            material_extension_factors(&material),
            MaterialExtensionFactors {
                emissive_strength: 3.0,
                ior: Some(1.33),
                transmission: Some(0.75),
                has_transmission_texture: true,
            }
        );
    }

    #[test]
    fn gltf_alpha_modes_map_to_render_contract() {
        assert_eq!(gltf_alpha_mode_or_default(None), AlphaMode::Opaque);

        let opaque = first_material(
            r#"{
                "asset": { "version": "2.0" },
                "materials": [{ "alphaMode": "OPAQUE" }]
            }"#,
        );
        assert_eq!(
            gltf_alpha_mode(&opaque.document.materials().next().unwrap()),
            AlphaMode::Opaque
        );

        let mask = first_material(
            r#"{
                "asset": { "version": "2.0" },
                "materials": [{ "alphaMode": "MASK", "alphaCutoff": 0.25 }]
            }"#,
        );
        assert_eq!(
            gltf_alpha_mode(&mask.document.materials().next().unwrap()),
            AlphaMode::Mask(0.25)
        );

        let blend = first_material(
            r#"{
                "asset": { "version": "2.0" },
                "materials": [{ "alphaMode": "BLEND" }]
            }"#,
        );
        assert_eq!(
            gltf_alpha_mode(&blend.document.materials().next().unwrap()),
            AlphaMode::Blend
        );
    }

    #[test]
    fn skeleton_edges_exclude_non_joint_parents_and_mesh_children() {
        let parents = vec![None, Some(0), Some(1), Some(1), Some(2)];
        let joints = HashSet::from([1, 2, 4]);
        assert_eq!(joint_edges(&parents, &joints), vec![(1, 2), (2, 4)]);
    }

    fn tiny_png() -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([255, 0, 0, 255]),
        ))
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("encode test PNG");
        bytes.into_inner()
    }

    fn triangle_glb(index: u16) -> Vec<u8> {
        let json = r#"{
            "asset":{"version":"2.0"},
            "buffers":[{"byteLength":44}],
            "bufferViews":[
                {"buffer":0,"byteOffset":0,"byteLength":36},
                {"buffer":0,"byteOffset":36,"byteLength":6}
            ],
            "accessors":[
                {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
                {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}
            ],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
            "nodes":[{"mesh":0}],
            "scenes":[{"nodes":[0]}],
            "scene":0
        }"#;
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }

        let mut bin = Vec::with_capacity(44);
        for vertex in [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for component in vertex {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        for value in [0_u16, 1, index] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let total_length = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    fn json_glb(json: &str) -> Vec<u8> {
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let total_length = 12 + 8 + json_bytes.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb
    }

    #[test]
    fn cyclic_node_graph_returns_error_without_panicking() {
        let bytes = json_glb(
            r#"{
                "asset":{"version":"2.0"},
                "nodes":[{"children":[1]},{"children":[0]}],
                "scenes":[{"nodes":[0]}],
                "scene":0
            }"#,
        );
        let result = std::panic::catch_unwind(|| load_from_slice(&bytes));
        let result = result.expect("malformed node references must not panic");
        assert!(result.is_err());
    }

    #[test]
    fn parent_graph_rejects_cycles_and_out_of_range_links() {
        assert!(validate_parent_graph(&[None, Some(0), Some(1)]).is_ok());
        assert!(validate_parent_graph(&[Some(1), Some(0)]).is_err());
        assert!(validate_parent_graph(&[Some(2)]).is_err());
    }

    #[test]
    fn animation_values_reject_non_finite_or_invalid_order() {
        assert!(validate_animation_times(&[0.0, 0.5, 0.5, 1.0]).is_ok());
        assert!(validate_animation_times(&[0.0, 1.0, 0.5]).is_err());
        assert!(validate_animation_times(&[0.0, f32::NAN]).is_err());
        assert!(validate_animation_vec3(&[Vec3::ZERO, Vec3::ONE]).is_ok());
        assert!(validate_animation_vec3(&[Vec3::new(f32::INFINITY, 0.0, 0.0)]).is_err());
        assert!(validate_animation_quat(&[Quat::IDENTITY], false).is_ok());
        assert!(validate_animation_quat(&[Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)], false).is_err());
        assert!(validate_animation_quat(
            &[
                Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
                Quat::IDENTITY,
                Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)
            ],
            true
        )
        .is_ok());
    }

    #[test]
    fn cpu_decode_rejects_out_of_range_triangle_index() {
        let error = decode_cpu_from_slice(&triangle_glb(3))
            .expect_err("an index outside the position accessor must reject the asset");
        assert!(error.to_string().contains("face index"), "{}", error);
    }

    #[test]
    fn triangle_index_collection_rejects_partial_triangle() {
        assert!(triangle_faces_from_indices(vec![0, 1, 2]).is_ok());
        let error = triangle_faces_from_indices(vec![0, 1, 2, 3])
            .expect_err("partial triangle index data must reject the asset");
        assert!(error.to_string().contains("multiple of three"));
    }

    fn temp_image_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "kiss3d-gltf-image-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn decoded_image_budget_tracks_decoded_pixels() {
        let mut budget = DecodedImageBudget::default();
        let image = decode_image_bytes(&tiny_png(), ImageFormat::Png, &mut budget).unwrap();

        assert_eq!((image.width(), image.height()), (1, 1));
        assert_eq!(budget.total_bytes, 4);
    }

    #[test]
    fn decoded_image_budget_rejects_aggregate_overflow() {
        let mut budget = DecodedImageBudget {
            max_total_bytes: 3,
            ..DecodedImageBudget::default()
        };
        let error = decode_image_bytes(&tiny_png(), ImageFormat::Png, &mut budget)
            .expect_err("aggregate decoded image budget must reject the output");

        assert!(matches!(
            error,
            gltf::Error::Image(image::ImageError::Limits(_))
        ));
        assert_eq!(budget.total_bytes, 0);
    }

    #[test]
    fn decoded_image_budget_rejects_dimensions_before_allocation() {
        let mut encoded = tiny_png();
        encoded[16..20].copy_from_slice(&(MAX_GLTF_IMAGE_DIMENSION + 1).to_be_bytes());
        let crc = crc32(&encoded[12..29]);
        encoded[29..33].copy_from_slice(&crc.to_be_bytes());

        let error = decode_image_bytes(
            &encoded,
            ImageFormat::Png,
            &mut DecodedImageBudget::default(),
        )
        .expect_err("image dimensions over the limit must be rejected");

        assert!(matches!(
            error,
            gltf::Error::Image(image::ImageError::Limits(_))
        ));
    }

    #[test]
    fn image_uri_and_buffer_view_sources_decode_without_renderer() {
        const PNG_BASE64: &str =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "uri": "data:application/octet-stream;base64,{PNG_BASE64}", "byteLength": 68 }}],
                "bufferViews": [{{ "buffer": 0, "byteOffset": 0, "byteLength": 68 }}],
                "images": [
                    {{ "uri": "data:image/png;base64,{PNG_BASE64}" }},
                    {{ "bufferView": 0, "mimeType": "image/png" }}
                ]
            }}"#
        );
        let gltf = gltf::Gltf::from_slice(json.as_bytes()).expect("test glTF should parse");
        let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob).unwrap();
        let images = import_images_with_budget(&gltf.document, None, &buffers).unwrap();

        assert_eq!(images.len(), 2);
        assert!(images
            .iter()
            .all(|image| image.width() == 1 && image.height() == 1));
    }

    #[test]
    fn image_relative_and_file_uris_preserve_loader_contract() {
        let image_path = temp_image_path();
        let image_bytes = tiny_png();
        fs::write(&image_path, &image_bytes).expect("write test PNG");
        let base = image_path.parent().expect("temp image parent");
        let file_uri = format!("file://{}", image_path.to_string_lossy().replace('\\', "/"));
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "images": [
                    {{ "uri": "{}" }},
                    {{ "uri": "{}" }}
                ]
            }}"#,
            image_path.file_name().unwrap().to_string_lossy(),
            file_uri
        );
        let gltf = gltf::Gltf::from_slice(json.as_bytes()).expect("test glTF should parse");
        let images = import_images_with_budget(&gltf.document, Some(base), &[]).unwrap();

        assert_eq!(images.len(), 2);
        assert!(images
            .iter()
            .all(|image| image.width() == 1 && image.height() == 1));
        fs::remove_file(image_path).expect("remove test PNG");
    }

    #[test]
    fn image_in_glb_bin_blob_decodes_without_external_files() {
        let image = tiny_png();
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "buffers": [{{ "byteLength": {} }}],
                "bufferViews": [{{ "buffer": 0, "byteLength": {} }}],
                "images": [{{ "bufferView": 0, "mimeType": "image/png" }}]
            }}"#,
            image.len(),
            image.len()
        );
        let mut json_bytes = json.into_bytes();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let mut bin_bytes = image;
        while !bin_bytes.len().is_multiple_of(4) {
            bin_bytes.push(0);
        }
        let total_length = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin_bytes);

        let gltf = gltf::Gltf::from_slice(&glb).expect("test GLB should parse");
        let gltf::Gltf { document, blob } = gltf;
        let buffers = gltf::import_buffers(&document, None, blob).unwrap();
        let images = import_images_with_budget(&document, None, &buffers).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!((images[0].width(), images[0].height()), (1, 1));
    }

    #[test]
    fn image_format_keeps_png_jpeg_contract() {
        assert!(matches!(
            image_format(Some("data:image/png;base64,abc"), None),
            Ok(ImageFormat::Png)
        ));
        assert!(matches!(
            image_format(Some("texture.jpg"), None),
            Ok(ImageFormat::Jpeg)
        ));
        assert!(matches!(
            image_format(Some("texture.webp"), None),
            Err(gltf::Error::UnsupportedImageEncoding)
        ));
    }

    fn crc32(bytes: &[u8]) -> u32 {
        bytes.iter().fold(0xffff_ffff, |mut crc, byte| {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
            crc
        }) ^ 0xffff_ffff
    }
}
