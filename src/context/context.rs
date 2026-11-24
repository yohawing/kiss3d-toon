//! wgpu rendering context management.
//!
//! This module provides a global wgpu context that can be initialized and reset
//! across window recreations.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

// The global wgpu context singleton.
// We use RefCell<Option<>> instead of OnceLock to allow resetting the context
// when creating new windows (required for multi-window support).
thread_local! {
    static CONTEXT_SINGLETON: RefCell<Option<Context>> = const { RefCell::new(None) };
    // Track number of active windows to know when to reset the context
    static WINDOW_COUNT: Cell<usize> = const { Cell::new(0) };
    // Draw commands recorded on this thread since the current frame started.
    static DRAW_CALLS: Cell<u32> = const { Cell::new(0) };
}

/// The wgpu rendering context containing all GPU resources needed for rendering.
///
/// This struct is cloneable and thread-safe. It wraps wgpu resources in Arc
/// to allow sharing across the application.
#[derive(Clone)]
pub struct Context {
    /// The wgpu instance used for creating surfaces.
    pub instance: Arc<wgpu::Instance>,
    /// The wgpu device used for creating GPU resources.
    pub device: Arc<wgpu::Device>,
    /// The wgpu queue used for submitting commands.
    pub queue: Arc<wgpu::Queue>,
    /// The wgpu adapter information.
    pub adapter: Arc<wgpu::Adapter>,
    /// The preferred texture format for the surface.
    pub surface_format: wgpu::TextureFormat,
    // Device-owned: never reuse pipelines across window/context recreation.
    pub(crate) mip_pipelines:
        Arc<std::sync::Mutex<std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>>>,
    /// First uncaptured device error or device loss, recorded by the handlers
    /// installed with [`Context::capture_device_errors`].
    pub(crate) device_error: Arc<std::sync::Mutex<Option<String>>>,
}

fn record_device_error(errors: &std::sync::Mutex<Option<String>>, message: String) {
    log::error!("kiss3d: {message}");
    let mut slot = errors
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(message);
    }
}

impl Context {
    /// Initializes or reinitializes the global wgpu context.
    ///
    /// This function is called when creating a window. For multi-window support,
    /// this will replace the existing context with a new one.
    ///
    /// # Arguments
    /// * `instance` - The wgpu instance
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `adapter` - The wgpu adapter
    /// * `surface_format` - The preferred surface texture format
    pub fn init(
        instance: wgpu::Instance,
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
    ) {
        CONTEXT_SINGLETON.with(|cell| {
            *cell.borrow_mut() = Some(Context {
                instance: Arc::new(instance),
                device: Arc::new(device),
                queue: Arc::new(queue),
                adapter: Arc::new(adapter),
                surface_format,
                mip_pipelines: Default::default(),
                device_error: Default::default(),
            });
        });
    }

    /// Gets a clone of the global wgpu context.
    ///
    /// # Panics
    /// Panics if the context has not been initialized via `init()`.
    pub fn get() -> Context {
        CONTEXT_SINGLETON.with(|cell| {
            cell.borrow()
                .as_ref()
                .expect("wgpu context not initialized. Call Context::init() first.")
                .clone()
        })
    }

    /// Records uncaptured device errors and device loss instead of panicking.
    ///
    /// wgpu's default uncaptured-error handler panics. Embedding hosts need
    /// to recover instead, so the first error (or the loss reason) is kept
    /// and reported by [`Context::device_error`]. Later errors are logged.
    pub(crate) fn capture_device_errors(&self) {
        let errors = self.device_error.clone();
        self.device
            .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                record_device_error(&errors, format!("wgpu error: {error}"));
            }));
        let errors = self.device_error.clone();
        self.device
            .set_device_lost_callback(move |reason, message| {
                record_device_error(&errors, format!("device lost ({reason:?}): {message}"));
            });
    }

    /// The first uncaptured device error or device loss of the current
    /// context, if error capture is installed and something went wrong.
    ///
    /// Once set, the device should be treated as unusable: drop every
    /// window so the context resets, then create a new one.
    pub fn device_error() -> Option<String> {
        CONTEXT_SINGLETON.with(|cell| {
            cell.borrow().as_ref().and_then(|ctx| {
                ctx.device_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone()
            })
        })
    }

    /// Checks if the context has been initialized.
    pub fn is_initialized() -> bool {
        CONTEXT_SINGLETON.with(|cell| cell.borrow().is_some())
    }

    /// Resets the global wgpu context, dropping all GPU resources.
    ///
    /// This should be called before thread-local storage destruction begins
    /// to avoid TLS access order issues with wgpu internals.
    ///
    /// After calling this, `is_initialized()` will return `false` and
    /// `get()` will panic until `init()` is called again.
    pub fn reset() {
        CONTEXT_SINGLETON.with(|cell| {
            // Explicitly destroy the device before dropping the context.
            // This ensures WebGPU resources are released immediately rather than
            // waiting for garbage collection, which is important for browsers
            // that limit the number of concurrent WebGPU contexts.
            if let Some(ctx) = cell.borrow().as_ref() {
                ctx.device.destroy();
            }
            *cell.borrow_mut() = None;
        });
    }

    /// Increments the window reference count.
    ///
    /// Called when a new window is created to track how many windows
    /// are using the context.
    pub fn increment_window_count() {
        WINDOW_COUNT.with(|count| {
            count.set(count.get() + 1);
        });
    }

    /// Decrements the window reference count and returns true if this was the last window.
    ///
    /// Called when a window is dropped. Returns true if all windows have been closed
    /// and it's safe to reset the context.
    pub fn decrement_window_count() -> bool {
        WINDOW_COUNT.with(|count| {
            let current = count.get();
            if current > 0 {
                count.set(current - 1);
                current == 1 // Was this the last window?
            } else {
                false
            }
        })
    }

    /// Returns the current number of active windows.
    pub fn window_count() -> usize {
        WINDOW_COUNT.with(|count| count.get())
    }

    /// Counts one draw command (`draw`, `draw_indexed` or an indirect draw)
    /// recorded on this thread for the frame being rendered.
    ///
    /// Every Kiss3D pass calls this next to its draw; custom materials,
    /// renderers and post-processing effects should do the same so
    /// [`Window::last_frame_draw_calls`](crate::window::Window::last_frame_draw_calls)
    /// covers them.
    #[inline]
    pub fn count_draw_call() {
        Self::count_draw_calls(1);
    }

    /// Counts `count` draw commands recorded on this thread for the frame
    /// being rendered. See [`Context::count_draw_call`].
    #[inline]
    pub fn count_draw_calls(count: u32) {
        DRAW_CALLS.with(|calls| calls.set(calls.get().saturating_add(count)));
    }

    /// Returns the draw commands counted since the last call and restarts
    /// the count. Windows call it at frame start and frame end.
    pub(crate) fn take_draw_calls() -> u32 {
        DRAW_CALLS.with(|calls| calls.replace(0))
    }

    /// Creates a new buffer on the GPU using a descriptor.
    ///
    /// # Arguments
    /// * `desc` - Buffer descriptor
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(desc)
    }

    /// Creates a new buffer on the GPU with specified parameters.
    ///
    /// # Arguments
    /// * `label` - Debug label for the buffer
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - Buffer usage flags
    pub fn create_buffer_simple(
        &self,
        label: Option<&str>,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Creates a new buffer initialized with data.
    ///
    /// # Arguments
    /// * `label` - Debug label for the buffer
    /// * `contents` - The data to initialize the buffer with
    /// * `usage` - Buffer usage flags
    pub fn create_buffer_init(
        &self,
        label: Option<&str>,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents,
                usage,
            })
    }

    /// Writes data to a buffer.
    ///
    /// # Arguments
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer
    /// * `data` - The data to write
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Creates a new texture on the GPU.
    ///
    /// # Arguments
    /// * `desc` - Texture descriptor
    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> wgpu::Texture {
        self.device.create_texture(desc)
    }

    /// Creates a new sampler.
    ///
    /// # Arguments
    /// * `desc` - Sampler descriptor
    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> wgpu::Sampler {
        self.device.create_sampler(desc)
    }

    /// Creates a new bind group layout.
    ///
    /// # Arguments
    /// * `desc` - Bind group layout descriptor
    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor,
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(desc)
    }

    /// Creates a new bind group.
    ///
    /// # Arguments
    /// * `desc` - Bind group descriptor
    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(desc)
    }

    /// Creates a new pipeline layout.
    ///
    /// # Arguments
    /// * `desc` - Pipeline layout descriptor
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor,
    ) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(desc)
    }

    /// Creates a new render pipeline.
    ///
    /// # Arguments
    /// * `desc` - Render pipeline descriptor
    pub fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(desc)
    }

    /// Creates a new shader module from WGSL source.
    ///
    /// # Arguments
    /// * `label` - Debug label for the shader
    /// * `source` - WGSL shader source code
    pub fn create_shader_module(&self, label: Option<&str>, source: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    /// Creates a new command encoder.
    ///
    /// # Arguments
    /// * `label` - Debug label for the encoder
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Submits command buffers to the GPU queue.
    ///
    /// # Arguments
    /// * `command_buffers` - Iterator of command buffers to submit
    pub fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(&self, command_buffers: I) {
        self.queue.submit(command_buffers);
    }

    /// Submits command buffers to the GPU queue, returning the submission
    /// index so callers can wait for exactly this submission (rather than the
    /// whole device).
    pub fn submit_indexed<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> wgpu::SubmissionIndex {
        self.queue.submit(command_buffers)
    }

    /// Writes texture data to the GPU.
    ///
    /// # Arguments
    /// * `texture` - The texture to write to
    /// * `data` - The pixel data
    /// * `data_layout` - Layout of the pixel data
    /// * `size` - Size of the region to write
    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.queue.write_texture(texture, data, data_layout, size);
    }

    /// Gets the depth texture format used for depth attachments.
    pub fn depth_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }

    /// Whether this device can run clustered (forward+) lighting.
    ///
    /// Clustered lighting needs compute shaders (for light culling) and at least three
    /// read-only storage buffers visible to the fragment stage (the light list, the
    /// per-cluster light grid and the global light-index list). Native and WebGPU
    /// browsers satisfy this; WebGL2 reports no compute shaders and zero storage buffers,
    /// so it transparently falls back to the legacy fixed 8-light uniform path.
    ///
    /// This is a runtime check on purpose: WebGPU and WebGL2 both build as `wasm32`, so a
    /// compile-time `cfg(target_arch = "wasm32")` gate would wrongly disable clustering on
    /// WebGPU.
    pub fn supports_clustered_lighting(&self) -> bool {
        let downlevel = self.adapter.get_downlevel_capabilities().flags;
        downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            // The cluster light grid is an array<vec2<u32>> — an 8-byte stride.
            // Downlevel GL (Android GLES; WebGL2 fails the compute check first)
            // requires buffer bindings sized in multiples of 16 bytes, so a
            // device without this flag cannot bind the grid at all.
            && downlevel.contains(wgpu::DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED)
            && self.device.limits().max_storage_buffers_per_shader_stage >= 3
    }

    /// Whether this device can run GPU skinning and morph targets.
    ///
    /// The deform bind group is five read-only storage buffers in the vertex
    /// stage (palette, joints, weights, morph positions, morph normals).
    /// WebGL2 has no vertex storage buffers at all and Android's GLES tops out
    /// at four, so on those targets the deform pipelines are never built and
    /// skinned/morphed meshes draw in their rest pose. A runtime check for the
    /// same reason as clustered lighting: the capable and incapable backends
    /// share compile targets.
    pub fn supports_deform(&self) -> bool {
        self.device.limits().max_storage_buffers_per_shader_stage >= 5
    }

    /// The internal floating-point color format the rasterizer renders into.
    ///
    /// The rasterized scene (3D + 2D + points/polylines) is drawn into an HDR
    /// `Rgba16Float` target so emissive values and bright highlights survive
    /// `> 1.0`. A final tonemap pass (see [`HdrPipeline`](crate::post_processing::HdrPipeline))
    /// converts this to the LDR [`surface_format`](Self::surface_format) for
    /// presentation. Material and renderer color pipelines must use this format,
    /// while the swapchain, post-processing, text and egui keep using the LDR
    /// surface format.
    pub fn render_format() -> wgpu::TextureFormat {
        crate::post_processing::HDR_FORMAT
    }
}

#[cfg(test)]
mod tests {
    use super::{record_device_error, Context};

    #[test]
    fn device_error_keeps_the_first_error() {
        let errors = std::sync::Mutex::new(None);
        record_device_error(&errors, "first".into());
        record_device_error(&errors, "second".into());
        assert_eq!(errors.lock().unwrap().as_deref(), Some("first"));
    }

    #[test]
    fn device_error_is_none_without_a_context() {
        if !Context::is_initialized() {
            assert_eq!(Context::device_error(), None);
        }
    }

    #[test]
    fn draw_call_count_accumulates_and_resets_on_take() {
        Context::take_draw_calls();
        Context::count_draw_call();
        Context::count_draw_calls(4);
        assert_eq!(Context::take_draw_calls(), 5);
        assert_eq!(Context::take_draw_calls(), 0);
        Context::count_draw_calls(u32::MAX);
        Context::count_draw_call();
        assert_eq!(Context::take_draw_calls(), u32::MAX);
    }
}
