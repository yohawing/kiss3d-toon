use std::path::PathBuf;

fn main() {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: fbx_anim_inspect <file.fbx>");
    let path_text = path.to_str().expect("path must be valid UTF-8");
    let scene = ufbx::load_file(
        path_text,
        ufbx::LoadOpts {
            load_external_files: false,
            ignore_embedded: true,
            target_axes: ufbx::CoordinateAxes {
                right: ufbx::CoordinateAxis::PositiveX,
                up: ufbx::CoordinateAxis::PositiveY,
                front: ufbx::CoordinateAxis::PositiveZ,
            },
            target_unit_meters: 1.0,
            space_conversion: ufbx::SpaceConversion::ModifyGeometry,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!(
            "ufbx {:?}: {} ({})",
            error.type_,
            error.description,
            error.info()
        )
    });

    println!(
        "file={} fps={} nodes={} meshes={} skins={} stacks={}",
        path.display(),
        scene.settings.frames_per_second,
        scene.nodes.len(),
        scene.meshes.len(),
        scene.skin_deformers.len(),
        scene.anim_stacks.len()
    );
    for stack in scene.anim_stacks.iter() {
        let mut transform_props = 0usize;
        let mut animated_nodes = std::collections::HashSet::new();
        for layer in stack.layers.iter() {
            for prop in layer.anim_props.iter() {
                if matches!(
                    prop.prop_name.as_ref(),
                    "Lcl Translation" | "Lcl Rotation" | "Lcl Scaling"
                ) {
                    transform_props += 1;
                    animated_nodes.insert(prop.element.element_id);
                }
            }
        }
        println!(
            "stack={:?} begin={:.6} end={:.6} duration={:.6} layers={} transform_props={} animated_nodes={}",
            stack.element.name.as_ref(),
            stack.time_begin,
            stack.time_end,
            stack.time_end - stack.time_begin,
            stack.layers.len(),
            transform_props,
            animated_nodes.len()
        );
    }
    for (index, mesh) in scene.meshes.iter().enumerate() {
        println!(
            "mesh[{index}] name={:?} vertices={} triangles={} skin_deformers={} blend_deformers={}",
            mesh.element.name.as_ref(),
            mesh.num_vertices,
            mesh.num_triangles,
            mesh.skin_deformers.len(),
            mesh.blend_deformers.len()
        );
    }
}
