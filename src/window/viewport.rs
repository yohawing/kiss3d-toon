use super::Window;
use crate::context::Context;

/// A physical-pixel sub-rectangle of the surface used for embedded 3D output.
/// Coordinates use the wgpu/CSS convention: origin at the surface's top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl RenderViewport {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub(crate) fn clamp(self, surface_width: u32, surface_height: u32) -> Option<Self> {
        let x = self.x.min(surface_width);
        let y = self.y.min(surface_height);
        let width = self.width.min(surface_width.saturating_sub(x));
        let height = self.height.min(surface_height.saturating_sub(y));
        (width > 0 && height > 0).then_some(Self {
            x,
            y,
            width,
            height,
        })
    }
}

impl Window {
    /// Restricts embedded rendering to a physical-pixel surface rectangle.
    ///
    /// The renderer uses the rectangle's dimensions for camera projection and
    /// all intermediate effects, then composites the final image into the
    /// requested surface position. Passing `None` restores full-surface output.
    pub fn set_render_viewport(&mut self, viewport: Option<RenderViewport>) {
        self.render_viewport = viewport;
    }

    pub fn render_viewport(&self) -> Option<RenderViewport> {
        self.render_viewport
    }
}

pub(crate) struct ViewportCompositor {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

impl ViewportCompositor {
    pub(crate) fn new(format: wgpu::TextureFormat) -> Self {
        let ctxt = Context::get();
        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport_composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport_composite_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("viewport_composite_shader"),
            include_str!("../builtin/raytrace/blit.wgsl"),
        );
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport_composite_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport_composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            bind_group_layout,
            pipeline,
            sampler,
        }
    }

    pub(crate) fn composite(
        &self,
        source: &wgpu::TextureView,
        surface: &wgpu::TextureView,
        viewport: RenderViewport,
    ) {
        let ctxt = Context::get();
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport_composite_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut encoder = ctxt.create_command_encoder(Some("viewport_composite_encoder"));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(
                viewport.x as f32,
                viewport.y as f32,
                viewport.width as f32,
                viewport.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(viewport.x, viewport.y, viewport.width, viewport.height);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..1);
            Context::count_draw_call();
        }
        ctxt.submit(std::iter::once(encoder.finish()));
    }

    pub(crate) fn clear(&self, surface: &wgpu::TextureView) {
        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("viewport_clear_encoder"));
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        ctxt.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::RenderViewport;

    #[test]
    fn clamp_preserves_in_bounds_viewport() {
        let viewport = RenderViewport::new(10, 20, 300, 200);
        assert_eq!(viewport.clamp(800, 600), Some(viewport));
    }

    #[test]
    fn clamp_trims_right_and_bottom_edges() {
        assert_eq!(
            RenderViewport::new(700, 550, 200, 100).clamp(800, 600),
            Some(RenderViewport::new(700, 550, 100, 50))
        );
    }

    #[test]
    fn clamp_rejects_zero_or_fully_outside_viewports() {
        assert_eq!(RenderViewport::new(0, 0, 0, 20).clamp(800, 600), None);
        assert_eq!(RenderViewport::new(800, 0, 20, 20).clamp(800, 600), None);
        assert_eq!(RenderViewport::new(0, 600, 20, 20).clamp(800, 600), None);
    }
}
