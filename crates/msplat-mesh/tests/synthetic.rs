//! End-to-end mesh extraction on a synthetic splat: write a standard 3DGS
//! PLY of gaussians coating a sphere, then run the full load → TSDF →
//! surface nets → export pipeline and sanity-check the result.

use std::io::Write;
use std::path::PathBuf;

use glam::Vec3;

const FIELDS: &[&str] = &[
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
];

/// Write gaussians distributed on a sphere of `radius`, flattened along the
/// surface normal (smallest scale = radial direction), in INRIA 3DGS layout
/// (logit opacity, log scales, scalar-first quaternion).
fn write_sphere_splat(path: &PathBuf, radius: f32, count: usize) {
    let mut file = std::fs::File::create(path).unwrap();
    write!(file, "ply\nformat binary_little_endian 1.0\nelement vertex {count}\n").unwrap();
    for f in FIELDS {
        writeln!(file, "property float {f}").unwrap();
    }
    writeln!(file, "end_header").unwrap();

    // Fibonacci sphere for even coverage.
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    for i in 0..count {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
        let r = (1.0 - y * y).sqrt();
        let theta = golden * i as f32;
        let n = Vec3::new(r * theta.cos(), y, r * theta.sin());
        let p = n * radius;

        // Rotation taking +x to the normal: gaussian is flattest along x.
        let quat = glam::Quat::from_rotation_arc(Vec3::X, n);

        let logit_opacity = 4.0_f32; // sigmoid(4) ≈ 0.982
        let scales = [
            (0.01_f32).ln(),
            (0.05_f32).ln(),
            (0.05_f32).ln(),
        ];
        let color_dc = [0.5_f32, -0.2, 0.1];

        let mut row: Vec<f32> = vec![p.x, p.y, p.z];
        row.extend_from_slice(&color_dc);
        row.push(logit_opacity);
        row.extend_from_slice(&scales);
        // Scalar-first storage: (w, x, y, z).
        row.extend_from_slice(&[quat.w, quat.x, quat.y, quat.z]);
        for v in row {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

#[test]
fn sphere_splat_to_mesh() {
    let dir = std::env::temp_dir().join("msplat-synthetic");
    std::fs::create_dir_all(&dir).unwrap();
    let splat_path = dir.join("sphere_splat.ply");
    let radius = 1.0;
    write_sphere_splat(&splat_path, radius, 20_000);

    let cloud = msplat_mesh::load_splat_ply(&splat_path).unwrap();
    assert_eq!(cloud.len(), 20_000);
    // Opacity activation applied?
    assert!((cloud.opacities[0] - 0.982).abs() < 0.01);
    // Scale activation applied?
    assert!(cloud.scales[0].min_element() < 0.02);

    // Cameras around the sphere orient normals outward.
    let cameras: Vec<Vec3> = (0..8)
        .map(|i| {
            let a = i as f32 / 8.0 * std::f32::consts::TAU;
            Vec3::new(a.cos() * 4.0, if i % 2 == 0 { 2.0 } else { -2.0 }, a.sin() * 4.0)
        })
        .collect();

    let params = msplat_mesh::MeshParams {
        grid_res: 96,
        ..Default::default()
    };
    let mesh = msplat_mesh::extract_mesh(&cloud, &cameras, &params, |_| {}).unwrap();

    assert!(mesh.num_vertices() > 1_000, "too few vertices: {}", mesh.num_vertices());
    assert!(mesh.num_triangles() > 1_000);

    // Vertices should lie near the sphere surface.
    let mut max_err = 0.0f32;
    for v in &mesh.positions {
        max_err = max_err.max((v.length() - radius).abs());
    }
    assert!(max_err < 0.15, "surface deviates {max_err} from the sphere");

    // Colors come from the splats: rgb = 0.5 + 0.2820948 * dc.
    let c = mesh.colors[0];
    assert!((c[0] - (0.5 + 0.2820948 * 0.5)).abs() < 0.05);

    // All exporters accept the mesh.
    msplat_mesh::write_ply(&mesh, &dir.join("mesh.ply")).unwrap();
    msplat_mesh::write_obj(&mesh, &dir.join("mesh.obj")).unwrap();
    msplat_mesh::write_glb(&mesh, &dir.join("mesh.glb")).unwrap();
    assert!(std::fs::metadata(dir.join("mesh.glb")).unwrap().len() > 1_000);
}
