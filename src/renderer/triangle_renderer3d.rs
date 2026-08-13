//! Batched, frame-local solid triangles for editor/debug overlays.

use bytemuck::{Pod, Zeroable};
use glamx::Vec3;

use crate::camera::Camera3d;
use crate::color::Color;
use crate::context::Context;
use crate::renderer::Renderer3d;
use crate::resource::{multisample_state, PipelineCache, RenderContext};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TriangleVertex {
    position: [f32; 3],
    depth_bias: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

pub struct TriangleRenderer3d {
    pipeline: PipelineCache,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_uniform_buffer: wgpu::Buffer,
    vertex_storage_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertices: Vec<TriangleVertex>,
}

impl Default for TriangleRenderer3d {
    fn default() -> Self {
        Self::new()
    }
}

impl TriangleRenderer3d {
    pub fn new() -> Self {
        let ctxt = Context::get();
        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("triangle_overlay_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle_overlay_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("triangle_overlay_shader"),
            include_str!("../builtin/triangles3d.wgsl"),
        );
        let pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("triangle_overlay_pipeline"),
                layout: Some(&pipeline_layout),
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
                        format: Context::render_format(),
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Context::depth_format(),
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: multisample_state(sample_count),
                multiview_mask: None,
                cache: None,
            })
        });
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_overlay_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_capacity = 256;
        let vertex_storage_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_overlay_vertex_storage_buffer"),
            size: (std::mem::size_of::<TriangleVertex>() * vertex_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_group_layout,
            frame_uniform_buffer,
            vertex_storage_buffer,
            vertex_capacity,
            vertices: Vec::new(),
        }
    }

    pub fn needs_rendering(&self) -> bool {
        !self.vertices.is_empty()
    }

    pub fn draw_triangle(&mut self, points: [Vec3; 3], color: Color, depth_bias: f32) {
        self.vertices.extend(points.map(|position| TriangleVertex {
            position: position.into(),
            depth_bias,
            color: [color.r, color.g, color.b, color.a],
        }));
    }

    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.vertex_capacity {
            return;
        }
        let ctxt = Context::get();
        self.vertex_capacity = needed.next_power_of_two();
        self.vertex_storage_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_overlay_vertex_storage_buffer"),
            size: (std::mem::size_of::<TriangleVertex>() * self.vertex_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
}

impl Renderer3d for TriangleRenderer3d {
    fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        if self.vertices.is_empty() {
            return;
        }
        let ctxt = Context::get();
        let (view, proj) = camera.view_transform_pair(pass);
        ctxt.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
            }),
        );
        self.ensure_capacity(self.vertices.len());
        ctxt.write_buffer(
            &self.vertex_storage_buffer,
            0,
            bytemuck::cast_slice(&self.vertices),
        );
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("triangle_overlay_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.vertex_storage_buffer.as_entire_binding(),
                },
            ],
        });
        render_pass.set_pipeline(&self.pipeline.get(context.sample_count));
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
        self.vertices.clear();
    }
}
