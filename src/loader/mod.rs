//! File loading.

#[cfg(not(target_arch = "wasm32"))]
pub mod fbx;
pub mod gltf;
pub mod mtl;
pub mod obj;
