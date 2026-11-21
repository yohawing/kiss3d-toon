//! Resource manager to allocate and switch between render targets.

use crate::context::Context;

/// The target to every rendering call.
pub enum RenderTarget {
    /// The screen (main surface).
    Screen,
    /// An off-screen buffer.
    Offscreen(Box<OffscreenBuffers>),
}

/// wgpu resources for an off-screen render target.
pub struct OffscreenBuffers {
    /// The color texture to render to.
    pub color_texture: wgpu::Texture,
    /// The color texture view.
    pub color_view: wgpu::TextureView,
    /// The depth texture (present only when created with `create_depth_texture = true`).
    pub depth_texture: Option<wgpu::Texture>,
    /// The depth texture view (present only when created with `create_depth_texture = true`).
    pub depth_view: Option<wgpu::TextureView>,
    /// The sampler for the color texture (for post-processing).
    pub sampler: wgpu::Sampler,
    /// Width of the render target.
    pub width: u32,
    /// Height of the render target.
    pub height: u32,
}

impl RenderTarget {
    /// Returns the color texture view for off-screen rendering.
    ///
    /// Returns `None` if this is the screen target.
    pub fn color_view(&self) -> Option<&wgpu::TextureView> {
        match self {
            RenderTarget::Screen => None,
            RenderTarget::Offscreen(o) => Some(&o.color_view),
        }
    }

    /// Returns the depth texture view for off-screen rendering.
    ///
    /// Returns `None` if this is the screen target or was created without a depth texture.
    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        match self {
            RenderTarget::Screen => None,
            RenderTarget::Offscreen(o) => o.depth_view.as_ref(),
        }
    }

    /// Returns the color texture for off-screen rendering.
    ///
    /// Returns `None` if this is the screen target.
    pub fn color_texture(&self) -> Option<&wgpu::Texture> {
        match self {
            RenderTarget::Screen => None,
            RenderTarget::Offscreen(o) => Some(&o.color_texture),
        }
    }

    /// Returns the sampler for the color texture.
    ///
    /// Returns `None` if this is the screen target.
    pub fn sampler(&self) -> Option<&wgpu::Sampler> {
        match self {
            RenderTarget::Screen => None,
            RenderTarget::Offscreen(o) => Some(&o.sampler),
        }
    }

    /// Resizes this render target, preserving whether it has a depth texture.
    pub fn resize(&mut self, width: u32, height: u32, surface_format: wgpu::TextureFormat) {
        match self {
            RenderTarget::Screen => {
                // Screen resizing is handled by the canvas/surface
            }
            RenderTarget::Offscreen(o) => {
                let width = width.max(1);
                let height = height.max(1);
                if o.width != width || o.height != height {
                    let has_depth = o.depth_texture.is_some();
                    // Recreate textures with new size
                    **o = OffscreenBuffers::new(width, height, surface_format, has_depth);
                }
            }
        }
    }

    /// Resizes this render target and ensures it has a depth texture when
    /// `needs_depth` is set.
    ///
    /// A size change recreates the target with a depth texture only if
    /// `needs_depth`. At an unchanged size, an existing depth texture is kept even
    /// when no longer needed, so the color view stays stable for callers holding
    /// it; a missing one is added (recreating the target) when needed.
    pub(crate) fn resize_with_depth(
        &mut self,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        needs_depth: bool,
    ) {
        if let RenderTarget::Offscreen(o) = self {
            let width = width.max(1);
            let height = height.max(1);
            if o.width != width || o.height != height || (needs_depth && o.depth_texture.is_none())
            {
                **o = OffscreenBuffers::new(width, height, surface_format, needs_depth);
            }
        }
    }
}

impl OffscreenBuffers {
    /// Creates new off-screen buffers with the specified dimensions.
    pub fn new(
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        create_depth_texture: bool,
    ) -> Self {
        let ctxt = Context::get();

        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);

        // Create color texture
        let color_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_color_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture only when requested
        let (depth_texture, depth_view) = if create_depth_texture {
            let depth_format = Context::depth_format();
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some("offscreen_depth_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        // Create sampler for the color texture
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("offscreen_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        OffscreenBuffers {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            sampler,
            width,
            height,
        }
    }
}

/// A framebuffer manager. It manages off-screen render targets for post-processing effects.
pub struct FramebufferManager {
    /// The surface format for creating compatible textures.
    surface_format: wgpu::TextureFormat,
}

impl Default for FramebufferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FramebufferManager {
    /// Creates a new framebuffer manager.
    pub fn new() -> FramebufferManager {
        let ctxt = Context::get();
        FramebufferManager {
            surface_format: ctxt.surface_format,
        }
    }

    /// Creates a new render target. A render target is the combination of a color buffer and a
    /// depth buffer.
    pub fn new_render_target(
        &self,
        width: u32,
        height: u32,
        create_depth_texture: bool,
    ) -> RenderTarget {
        RenderTarget::Offscreen(Box::new(OffscreenBuffers::new(
            width,
            height,
            self.surface_format,
            create_depth_texture,
        )))
    }

    /// Returns the render target associated with the screen.
    pub fn screen() -> RenderTarget {
        RenderTarget::Screen
    }

    /// Gets the surface format used by this manager.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }
}
