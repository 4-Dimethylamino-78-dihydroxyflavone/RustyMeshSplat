use std::path::{Path, PathBuf};

use serde::Serialize;

/// Machine-readable run summary written next to the outputs.
#[derive(Serialize, Default)]
pub struct RunReport {
    pub gpu: Option<String>,
    pub sfm: Option<SfmReport>,
    pub training: Option<TrainReport>,
    pub mesh: Option<MeshReport>,
    pub outputs: Vec<PathBuf>,
}

#[derive(Serialize)]
pub struct SfmReport {
    /// "colmap", "mapanything" or "mast3r".
    pub backend: String,
    /// Model identifier for feed-forward backends; None for COLMAP.
    pub model: Option<String>,
    pub input_images: usize,
    pub registered_images: usize,
    pub sparse_points: usize,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct TrainReport {
    pub iterations: u32,
    pub final_splat_count: u32,
    pub last_psnr: Option<f32>,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct MeshReport {
    pub vertices: usize,
    pub triangles: usize,
    pub seconds: f64,
}

impl RunReport {
    pub fn save(&self, out_dir: &Path) -> anyhow::Result<PathBuf> {
        let path = out_dir.join("report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)?;
        Ok(path)
    }
}
