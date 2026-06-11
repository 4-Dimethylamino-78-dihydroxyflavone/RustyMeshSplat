use std::path::Path;

use anyhow::{Context, Result};
use colmap_reader::{ColmapCamera, Image, Point3D};
use glam::Vec3;
use tokio::io::BufReader;

/// A parsed COLMAP sparse reconstruction.
pub struct SparseModel {
    pub cameras: Vec<ColmapCamera>,
    pub images: Vec<Image>,
    pub points: Vec<Point3D>,
}

impl SparseModel {
    /// World-space camera centers: C = -Rᵀ·t for COLMAP's world-to-cam pose.
    pub fn camera_centers(&self) -> Vec<Vec3> {
        self.images
            .iter()
            .map(|img| -(img.quat.conjugate() * img.tvec))
            .collect()
    }
}

/// Read a COLMAP sparse model dir, in binary (`.bin`) or text (`.txt`)
/// format — tools like VGGT/hloc export either.
pub async fn read_sparse_model(dir: &Path) -> Result<SparseModel> {
    let binary = dir.join("cameras.bin").is_file();
    let ext = if binary { "bin" } else { "txt" };
    let open = |stem: &str| {
        let path = dir.join(format!("{stem}.{ext}"));
        async move {
            let file = tokio::fs::File::open(&path)
                .await
                .with_context(|| format!("Opening {}", path.display()))?;
            anyhow::Ok(BufReader::new(file))
        }
    };

    let cameras = colmap_reader::read_cameras(open("cameras").await?, binary)
        .await
        .context("Parsing cameras")?;
    let images = colmap_reader::read_images(open("images").await?, binary, false)
        .await
        .context("Parsing images")?;
    let points = colmap_reader::read_points3d(open("points3D").await?, binary, false)
        .await
        .context("Parsing points3D")?;

    Ok(SparseModel {
        cameras,
        images,
        points,
    })
}
