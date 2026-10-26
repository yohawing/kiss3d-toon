//! Image-based lighting support: a mip-chained equirectangular environment map.
//!
//! The rasterizer approximates IBL with the "mip-as-prefilter" technique: a
//! single equirectangular HDR texture is given a full mip chain (each level a
//! box-downsample of the previous), so the coarse mips stand in for
//! roughness-prefiltered reflections, and the coarsest mip stands in for the
//! diffuse irradiance. Combined with Karis' analytic environment BRDF (in the
//! shader) this needs no separate irradiance / prefilter / BRDF-LUT passes while
//! still giving plausible ambient reflections and fill light. Optionally
//! ([`DiffuseIbl`]) the diffuse term is instead evaluated along the normal from a
//! band-4 spherical-harmonic projection of the environment, computed on the CPU.

use crate::context::Context;

/// Number of spherical-harmonic coefficients kept for directional diffuse IBL
/// (bands `l = 0..=4`).
pub const IBL_SH_COEFFS: usize = 25;

/// How the rasterizer's diffuse image-based lighting is evaluated
/// ([`Window::set_diffuse_ibl`](crate::window::Window::set_diffuse_ibl)).
/// Reflection probes always use their coarsest mip.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DiffuseIbl {
    /// The coarsest environment mip (the 1x1 average) for every normal: no
    /// directional variation. The default (original behavior).
    #[default]
    CoarsestMip,
    /// Lambertian irradiance `E(n)/π`: the environment convolved with the clamped
    /// cosine, evaluated along the environment-rotated normal from spherical
    /// harmonics up to band 4.
    Irradiance,
    /// three.js r185's diffuse IBL, `textureCubeUV(envMap, n, 1.0)`: the most
    /// filtered PMREM level, whose lobe is the product of PMREM's incremental GGX
    /// VNDF filter stages for an equirect of this width (see
    /// [`three_pmrem_band_weights`]). Evaluated like `Irradiance` with those band
    /// weights instead of the cosine's.
    ThreePmrem,
}

/// A GPU-resident equirectangular environment map with a full mip chain, used as
/// the rasterizer's image-based-lighting source.
pub struct EnvironmentMap {
    // Kept alive alongside the view.
    _texture: wgpu::Texture,
    /// View over all mip levels (sampled by the lighting shader with an explicit LOD).
    pub view: wgpu::TextureView,
    /// Linear/trilinear sampler (repeat in U, clamp in V).
    pub sampler: wgpu::Sampler,
    /// Number of mip levels (the max sampleable LOD is `mip_count - 1`).
    pub mip_count: u32,
    /// Width of the source equirect in pixels.
    pub width: u32,
    /// Radiance projected onto the real orthonormal SH basis ([`sh_basis`],
    /// bands 0..=4), RGB per coefficient, in the unrotated environment frame.
    pub sh: [[f32; 3]; IBL_SH_COEFFS],
    // `three_pmrem_band_weights(width)`, computed once.
    three_pmrem_weights: [f32; 5],
}

impl EnvironmentMap {
    /// Builds a mip-chained environment map from an equirectangular image.
    pub fn from_image(img: &image::DynamicImage) -> EnvironmentMap {
        use image::GenericImageView;
        let (w, h) = img.dimensions();
        let rgba = img.to_rgba32f();
        Self::from_rgba_f32(w, h, rgba.as_raw())
    }

    /// Builds a mip-chained environment map from RGBA-f32 pixels.
    pub fn from_rgba_f32(width: u32, height: u32, rgba: &[f32]) -> EnvironmentMap {
        let ctxt = Context::get();
        let width = width.max(1);
        let height = height.max(1);
        let mip_count = (32 - width.max(height).leading_zeros()).max(1);

        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_environment"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Upload mip 0 (f32 -> f16).
        let halves: Vec<u16> = rgba.iter().map(|&v| f32_to_f16(v)).collect();
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl_environment_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        if mip_count > 1 {
            Self::generate_mips(&texture, &sampler, mip_count);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        EnvironmentMap {
            _texture: texture,
            view,
            sampler,
            mip_count,
            width,
            sh: project_sh(width, height, rgba),
            three_pmrem_weights: three_pmrem_band_weights(width),
        }
    }

    /// The diffuse SH coefficients for `mode` with the mode's per-band kernel
    /// weights applied (radiance units: a uniform environment `L` gives `L`), or
    /// `None` for [`DiffuseIbl::CoarsestMip`]. `.w` is unused.
    pub fn diffuse_sh(&self, mode: DiffuseIbl) -> Option<[[f32; 4]; IBL_SH_COEFFS]> {
        let weights = match mode {
            DiffuseIbl::CoarsestMip => return None,
            DiffuseIbl::Irradiance => COSINE_BAND_WEIGHTS,
            DiffuseIbl::ThreePmrem => self.three_pmrem_weights,
        };
        Some(weighted_sh(&self.sh, weights))
    }

    /// Renders each mip from the previous one with a box (bilinear) downsample.
    fn generate_mips(texture: &wgpu::Texture, sampler: &wgpu::Sampler, mip_count: u32) {
        let ctxt = Context::get();

        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl_downsample_layout"),
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
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("ibl_downsample"),
            &crate::builtin::compile_shader_with_common(
                "package::env_downsample",
                crate::builtin::ENV_DOWNSAMPLE_WESL,
            ),
        );
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_downsample_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let mut encoder = ctxt.create_command_encoder(Some("ibl_mipgen_encoder"));
        for mip in 1..mip_count {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl_mip_src"),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl_mip_dst"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl_downsample_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ibl_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        ctxt.submit(std::iter::once(encoder.finish()));
    }
}

/// Band weights of the normalized clamped-cosine kernel `max(n·l, 0)/π`
/// (Ramamoorthi & Hanrahan 2001, divided by π), i.e. `E(n)/π` per band.
pub const COSINE_BAND_WEIGHTS: [f32; 5] = [1.0, 2.0 / 3.0, 0.25, 0.0, -1.0 / 24.0];

/// SH band `l` of coefficient index `i` (`i = l² + l + m`).
fn sh_band(i: usize) -> usize {
    match i {
        0 => 0,
        1..=3 => 1,
        4..=8 => 2,
        9..=15 => 3,
        _ => 4,
    }
}

/// `sh` with band weight `weights[l]` applied to every coefficient (`.w` = 0).
fn weighted_sh(sh: &[[f32; 3]; IBL_SH_COEFFS], weights: [f32; 5]) -> [[f32; 4]; IBL_SH_COEFFS] {
    let mut out = [[0.0; 4]; IBL_SH_COEFFS];
    for (i, (o, c)) in out.iter_mut().zip(sh).enumerate() {
        let k = weights[sh_band(i)];
        *o = [c[0] * k, c[1] * k, c[2] * k, 0.0];
    }
    out
}

/// Real orthonormal SH basis, bands 0..=4, of a unit direction `(x, y, z)`
/// (the same functions as `ibl_sh_basis` in `default.wgsl`).
pub fn sh_basis(x: f32, y: f32, z: f32) -> [f32; IBL_SH_COEFFS] {
    let (x2, y2, z2) = (x * x, y * y, z * z);
    [
        0.282_094_8,
        0.488_602_5 * y,
        0.488_602_5 * z,
        0.488_602_5 * x,
        1.092_548_4 * x * y,
        1.092_548_4 * y * z,
        0.315_391_57 * (3.0 * z2 - 1.0),
        1.092_548_4 * x * z,
        0.546_274_2 * (x2 - y2),
        0.590_043_6 * y * (3.0 * x2 - y2),
        2.890_611_4 * x * y * z,
        0.457_045_8 * y * (5.0 * z2 - 1.0),
        0.373_176_33 * z * (5.0 * z2 - 3.0),
        0.457_045_8 * x * (5.0 * z2 - 1.0),
        1.445_305_7 * z * (x2 - y2),
        0.590_043_6 * x * (x2 - 3.0 * y2),
        2.503_343 * x * y * (x2 - y2),
        1.770_130_8 * y * z * (3.0 * x2 - y2),
        0.946_174_7 * x * y * (7.0 * z2 - 1.0),
        0.669_046_5 * y * z * (7.0 * z2 - 3.0),
        0.105_785_55 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
        0.669_046_5 * x * z * (7.0 * z2 - 3.0),
        0.473_087_35 * (x2 - y2) * (7.0 * z2 - 1.0),
        1.770_130_8 * x * z * (x2 - 3.0 * y2),
        0.625_835_7 * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2),
    ]
}

/// Direction at equirect pixel coordinate `(i, j)` (texel centers at `+0.5`),
/// the inverse of `equirect_dir_to_uv` (`u = atan2(z, x)/2π + 0.5`,
/// `v = acos(y)/π`).
fn equirect_texel_dir(i: f64, j: f64, width: u32, height: u32) -> [f64; 3] {
    let phi = (i / width as f64 - 0.5) * 2.0 * std::f64::consts::PI;
    let theta = j / height as f64 * std::f64::consts::PI;
    let (st, ct) = theta.sin_cos();
    [st * phi.cos(), ct, st * phi.sin()]
}

/// Projects equirect RGBA-f32 radiance onto [`sh_basis`]. Texels are first summed
/// (solid-angle weighted) into cells of at most 128x64, so large maps stay cheap;
/// band 4 varies over ~45°, far wider than a cell.
fn project_sh(width: u32, height: u32, rgba: &[f32]) -> [[f32; 3]; IBL_SH_COEFFS] {
    let mut sh = [[0.0f32; 3]; IBL_SH_COEFFS];
    if rgba.len() < (width as usize) * (height as usize) * 4 {
        return sh;
    }
    let (sx, sy) = (width.div_ceil(128), height.div_ceil(64));
    let (cw, ch) = (width.div_ceil(sx), height.div_ceil(sy));
    let pi = std::f64::consts::PI;
    // Per cell: rgb·dΩ and dΩ.
    let mut cells = vec![[0.0f64; 4]; (cw * ch) as usize];
    for j in 0..height {
        let theta = (j as f64 + 0.5) / height as f64 * pi;
        let d_omega = (2.0 * pi / width as f64) * (pi / height as f64) * theta.sin();
        let row = ((j / sy) * cw) as usize;
        for i in 0..width {
            let p = ((j * width + i) * 4) as usize;
            let cell = &mut cells[row + (i / sx) as usize];
            for c in 0..3 {
                let v = rgba[p + c];
                if v.is_finite() {
                    cell[c] += v as f64 * d_omega;
                }
            }
            cell[3] += d_omega;
        }
    }
    let mut acc = [[0.0f64; 3]; IBL_SH_COEFFS];
    for cj in 0..ch {
        let (j0, j1) = (cj * sy, ((cj + 1) * sy).min(height));
        for ci in 0..cw {
            let cell = cells[(cj * cw + ci) as usize];
            if cell[3] <= 0.0 {
                continue;
            }
            let (i0, i1) = (ci * sx, ((ci + 1) * sx).min(width));
            let d = equirect_texel_dir(
                (i0 + i1) as f64 * 0.5,
                (j0 + j1) as f64 * 0.5,
                width,
                height,
            );
            let y = sh_basis(d[0] as f32, d[1] as f32, d[2] as f32);
            for (a, b) in acc.iter_mut().zip(y) {
                for c in 0..3 {
                    a[c] += cell[c] * b as f64;
                }
            }
        }
    }
    for (o, a) in sh.iter_mut().zip(acc) {
        *o = [a[0] as f32, a[1] as f32, a[2] as f32];
    }
    sh
}

/// Legendre polynomial `P_l(x)`.
fn legendre(l: usize, x: f64) -> f64 {
    let (mut p0, mut p1) = (1.0, x);
    if l == 0 {
        return p0;
    }
    for n in 2..=l {
        let p2 = ((2 * n - 1) as f64 * x * p1 - (n - 1) as f64 * p0) / n as f64;
        p0 = p1;
        p1 = p2;
    }
    p1
}

/// Normalized band weights (`k_l = ∫K·P_l / ∫K`) of one PMREM GGX stage. three's
/// `PMREMGGXConvolution` sets `V = N` and samples `H` from the GGX VNDF (pdf
/// `D·NoH` at normal incidence), so `L = reflect(-V, H)` has density `D(NoH)/4`
/// and is weighted by `NoL`: `K(θ) ∝ D_GGX(cos(θ/2))·cos θ` on the hemisphere.
fn ggx_stage_weights(alpha: f64) -> [f64; 5] {
    let a2 = alpha * alpha;
    let pi = std::f64::consts::PI;
    // Quadratic spacing in θ resolves the narrow early stages.
    let n = 4096;
    let mut k = [0.0f64; 5];
    let mut norm = 0.0;
    for s in 0..n {
        let (t0, t1) = (s as f64 / n as f64, (s + 1) as f64 / n as f64);
        let (th0, th1) = (t0 * t0 * pi / 2.0, t1 * t1 * pi / 2.0);
        let th = 0.5 * (th0 + th1);
        let noh = (th / 2.0).cos();
        let den = noh * noh * (a2 - 1.0) + 1.0;
        let w = a2 / (pi * den * den) * th.cos() * th.sin() * (th1 - th0);
        norm += w;
        for (l, kl) in k.iter_mut().enumerate() {
            *kl += w * legendre(l, th.cos());
        }
    }
    k.map(|v| v / norm)
}

/// Band weights of three.js r185's roughness-1 PMREM level for an equirect of
/// `width` pixels (`PMREMGenerator.fromEquirectangular`: cube size `width/4`,
/// `lodMax = floor(log2(size))`, `lodMax - 4 + 1 + 6` levels). `_applyPMREM`
/// filters each level from the previous one with GGX roughness
/// `sqrt(t² - s²)·1.25·t` (`t`, `s` the target/source level roughness, alpha =
/// roughness²), so the final lobe's band weights are the product over stages
/// (zonal kernels compose by multiplying their Legendre coefficients). Texel
/// resolution, bilinear sampling and the 256-sample quadrature are not modeled.
pub fn three_pmrem_band_weights(width: u32) -> [f32; 5] {
    let size = (width / 4).max(1) as f64;
    let lod_max = (size.log2().floor() as i32).max(4);
    let levels = lod_max - 4 + 1 + 6;
    let mut total = [1.0f64; 5];
    for i in 1..levels {
        let t = i as f64 / (levels - 1) as f64;
        let s = (i - 1) as f64 / (levels - 1) as f64;
        let roughness = (t * t - s * s).sqrt() * 1.25 * t;
        if roughness < 0.001 {
            continue; // three copies the level unfiltered
        }
        let stage = ggx_stage_weights(roughness * roughness);
        for (a, b) in total.iter_mut().zip(stage) {
            *a *= b;
        }
    }
    total.map(|v| v as f32)
}

/// Converts an `f32` to IEEE-754 half-precision bits (truncating mantissa).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// Midpoint quadrature over a `w`x`h` equirect: `f(direction, dΩ)`.
    fn equirect_quadrature(w: u32, h: u32, mut f: impl FnMut([f64; 3], f64)) {
        for j in 0..h {
            let theta = (j as f64 + 0.5) / h as f64 * PI;
            let d_omega = (2.0 * PI / w as f64) * (PI / h as f64) * theta.sin();
            for i in 0..w {
                f(
                    equirect_texel_dir(i as f64 + 0.5, j as f64 + 0.5, w, h),
                    d_omega,
                );
            }
        }
    }

    #[test]
    fn sh_basis_is_orthonormal() {
        let mut gram = [[0.0f64; IBL_SH_COEFFS]; IBL_SH_COEFFS];
        equirect_quadrature(256, 128, |d, w| {
            let y = sh_basis(d[0] as f32, d[1] as f32, d[2] as f32);
            for a in 0..IBL_SH_COEFFS {
                for b in 0..IBL_SH_COEFFS {
                    gram[a][b] += y[a] as f64 * y[b] as f64 * w;
                }
            }
        });
        for (a, row) in gram.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((v - expected).abs() < 2e-3, "<Y{}, Y{}> = {}", a, b, v);
            }
        }
    }

    #[test]
    fn cosine_sh_matches_direct_irradiance() {
        // 64x32 equirect: 0.1 everywhere, 4.0 within 35° of +X.
        let (w, h) = (64u32, 32u32);
        let radiance = |d: [f64; 3]| {
            if d[0] > 35f64.to_radians().cos() {
                4.0
            } else {
                0.1
            }
        };
        let mut rgba = vec![1.0f32; (w * h * 4) as usize];
        for j in 0..h {
            for i in 0..w {
                let d = equirect_texel_dir(i as f64 + 0.5, j as f64 + 0.5, w, h);
                let p = ((j * w + i) * 4) as usize;
                rgba[p..p + 3].fill(radiance(d) as f32);
            }
        }
        let sh = weighted_sh(&project_sh(w, h, &rgba), COSINE_BAND_WEIGHTS);
        let mut peak = 0.0f64;
        for n in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.0, 0.8],
        ] {
            let mut e = 0.0;
            equirect_quadrature(w, h, |d, dw| {
                let cos = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).max(0.0);
                e += radiance(d) * cos * dw;
            });
            let expected = e / PI;
            peak = peak.max(expected);
            let y = sh_basis(n[0] as f32, n[1] as f32, n[2] as f32);
            let got: f32 = sh.iter().zip(y).map(|(c, b)| c[0] * b).sum();
            // Band-limiting the cosine to l <= 4 leaves a few percent of the
            // peak for this sharp-edged cap.
            assert!(
                (got as f64 - expected).abs() < 0.03 * peak.max(1.0),
                "{:?}: {} vs {}",
                n,
                got,
                expected
            );
        }
        // A uniform environment gives its radiance.
        let sh_u = project_sh(w, h, &vec![0.5f32; (w * h * 4) as usize]);
        assert!((sh_u[0][0] * 0.282_094_8 - 0.5).abs() < 1e-3);
    }

    #[test]
    fn three_pmrem_weights_follow_the_ggx_chain() {
        // 1k equirect (cube 256, lodMax 8, 11 levels); reference values from an
        // independent JS integration of the same stage chain.
        let k = three_pmrem_band_weights(1024);
        let expected = [1.0, 0.635, 0.2703, 0.0857, 0.0233];
        for (a, b) in k.iter().zip(expected) {
            assert!((a - b).abs() < 2e-3, "{:?}", k);
        }
        // A single roughness-1 stage (alpha 1) is the normalized clamped cosine.
        let cos = ggx_stage_weights(1.0);
        for (a, b) in cos.iter().zip(COSINE_BAND_WEIGHTS) {
            assert!((*a as f32 - b).abs() < 2e-3, "{:?}", cos);
        }
    }
}
