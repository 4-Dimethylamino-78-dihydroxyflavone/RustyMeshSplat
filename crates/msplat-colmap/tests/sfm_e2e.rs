//! End-to-end SfM test against a real COLMAP binary.
//!
//! Ray-traces a richly textured synthetic scene (sphere on a ground plane)
//! from 16 orbiting cameras, then runs the full feature-extract → match →
//! map pipeline and checks that COLMAP registers the views.
//!
//! Skips (passes vacuously) when COLMAP is not installed, so CI without
//! COLMAP stays green while dev machines get real coverage.

use std::path::PathBuf;

use glam::Vec3;

/// Cheap deterministic 3D value noise: integer lattice hash + trilinear blend.
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x1656_67b1);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

fn value_noise(p: Vec3) -> f32 {
    let base = p.floor();
    let f = p - base;
    let (x, y, z) = (base.x as i32, base.y as i32, base.z as i32);
    let s = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let mut acc = 0.0;
    for (dx, dy, dz, w) in [
        (0, 0, 0, (1.0 - s.x) * (1.0 - s.y) * (1.0 - s.z)),
        (1, 0, 0, s.x * (1.0 - s.y) * (1.0 - s.z)),
        (0, 1, 0, (1.0 - s.x) * s.y * (1.0 - s.z)),
        (1, 1, 0, s.x * s.y * (1.0 - s.z)),
        (0, 0, 1, (1.0 - s.x) * (1.0 - s.y) * s.z),
        (1, 0, 1, s.x * (1.0 - s.y) * s.z),
        (0, 1, 1, (1.0 - s.x) * s.y * s.z),
        (1, 1, 1, s.x * s.y * s.z),
    ] {
        acc += w * hash3(x + dx, y + dy, z + dz);
    }
    acc
}

/// Multi-octave noise gives SIFT structure at several scales.
fn fbm(p: Vec3) -> f32 {
    0.55 * value_noise(p * 3.0) + 0.3 * value_noise(p * 9.0) + 0.15 * value_noise(p * 27.0)
}

fn texture(p: Vec3) -> [f32; 3] {
    let n = fbm(p);
    // High-contrast blotches with a color ramp.
    let t = (n * 4.0).fract();
    let warm = [0.85, 0.55, 0.25];
    let cool = [0.2, 0.4, 0.7];
    let mut c = [0.0f32; 3];
    for i in 0..3 {
        c[i] = warm[i] * t + cool[i] * (1.0 - t);
    }
    // Checker on top adds strong corners.
    let checker = ((p.x * 2.0).floor() + (p.z * 2.0).floor()) as i64 % 2 == 0;
    if checker {
        for v in &mut c {
            *v *= 0.6;
        }
    }
    c
}

struct Hit {
    point: Vec3,
    normal: Vec3,
}

fn trace(origin: Vec3, dir: Vec3) -> Option<Hit> {
    // Sphere r=1 at origin.
    let b = origin.dot(dir);
    let c = origin.length_squared() - 1.0;
    let disc = b * b - c;
    let sphere_t = (disc >= 0.0).then(|| -b - disc.sqrt()).filter(|t| *t > 1e-3);
    // Ground plane y = -1.
    let plane_t = (dir.y.abs() > 1e-6)
        .then(|| (-1.0 - origin.y) / dir.y)
        .filter(|t| *t > 1e-3);

    match (sphere_t, plane_t) {
        (Some(ts), Some(tp)) if ts < tp => Some(Hit {
            point: origin + dir * ts,
            normal: (origin + dir * ts).normalize(),
        }),
        (Some(ts), None) => Some(Hit {
            point: origin + dir * ts,
            normal: (origin + dir * ts).normalize(),
        }),
        (_, Some(tp)) => Some(Hit {
            point: origin + dir * tp,
            normal: Vec3::Y,
        }),
        _ => None,
    }
}

fn render(cam_pos: Vec3, look_at: Vec3, width: u32, height: u32) -> image::RgbImage {
    let forward = (look_at - cam_pos).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    let fov_scale = (55.0f32.to_radians() / 2.0).tan();
    let aspect = width as f32 / height as f32;
    let light1 = Vec3::new(0.5, 0.8, 0.3).normalize();
    let light2 = Vec3::new(-0.6, 0.4, -0.7).normalize();

    image::RgbImage::from_fn(width, height, |px, py| {
        let u = (2.0 * (px as f32 + 0.5) / width as f32 - 1.0) * fov_scale * aspect;
        let v = (1.0 - 2.0 * (py as f32 + 0.5) / height as f32) * fov_scale;
        let dir = (forward + right * u + up * v).normalize();
        let rgb = match trace(cam_pos, dir) {
            Some(hit) => {
                let lambert = 0.25
                    + 0.55 * hit.normal.dot(light1).max(0.0)
                    + 0.3 * hit.normal.dot(light2).max(0.0);
                let tex = texture(hit.point);
                [tex[0] * lambert, tex[1] * lambert, tex[2] * lambert]
            }
            // Sky gradient with noise so the background isn't featureless.
            None => {
                let g = 0.55 + 0.3 * dir.y + 0.1 * value_noise(dir * 14.0);
                [g * 0.7, g * 0.8, g]
            }
        };
        image::Rgb(rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8))
    })
}

fn generate_scene(dir: &PathBuf, views: usize) {
    std::fs::create_dir_all(dir).unwrap();
    for i in 0..views {
        let angle = i as f32 / views as f32 * std::f32::consts::TAU;
        // Slight elevation wobble avoids a degenerate single-ring orbit.
        let elev = 0.45 + 0.15 * (i as f32 * 1.7).sin();
        let cam = Vec3::new(angle.cos() * 3.6, elev * 3.6, angle.sin() * 3.6);
        let img = render(cam, Vec3::new(0.0, -0.2, 0.0), 800, 600);
        img.save(dir.join(format!("view_{i:02}.png"))).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn colmap_pipeline_on_synthetic_scene() {
    let Some(colmap) = msplat_colmap::locate_colmap(None) else {
        eprintln!("COLMAP not installed — skipping SfM e2e test");
        return;
    };
    eprintln!("Using {:?}", colmap.version());

    let root = std::env::temp_dir().join("msplat-sfm-e2e");
    let photos = root.join("photos");
    let work = root.join("work");
    let _ = std::fs::remove_dir_all(&root);
    let views = 16;
    generate_scene(&photos, views);

    let opts = msplat_colmap::SfmOptions {
        capture: msplat_colmap::CaptureMode::Unordered,
        single_camera: true,
        ..Default::default()
    };
    let output = msplat_colmap::run_sfm(&colmap, &photos, &work, &opts, |event| {
        if let msplat_colmap::SfmEvent::StageStarted { name, index, total } = event {
            eprintln!("[{index}/{total}] {name}");
        }
    })
    .await
    .expect("SfM pipeline failed");

    assert!(
        output.stats.registered_images >= views * 3 / 4,
        "only {}/{} images registered",
        output.stats.registered_images,
        views
    );
    assert!(
        output.stats.sparse_points > 500,
        "too few sparse points: {}",
        output.stats.sparse_points
    );
    // The assembled dataset must look like what Brush expects.
    assert!(output.dataset_dir.join("images").join("view_00.png").is_file());
    for f in ["cameras.bin", "images.bin", "points3D.bin"] {
        assert!(output.sparse_dir.join(f).is_file(), "missing {f}");
    }

    // Camera centers should sit near the orbit radius.
    let model = msplat_colmap::read_sparse_model(&output.sparse_dir)
        .await
        .unwrap();
    let centers = model.camera_centers();
    let mean_r = centers.iter().map(|c| c.length()).sum::<f32>() / centers.len() as f32;
    for c in &centers {
        let r = c.length();
        assert!(
            (r / mean_r - 1.0).abs() < 0.25,
            "camera at radius {r}, mean {mean_r} — orbit not recovered"
        );
    }
}
