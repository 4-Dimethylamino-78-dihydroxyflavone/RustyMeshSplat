use std::path::{Path, PathBuf};

use serde::Serialize;

/// Machine-readable run summary written next to the outputs.
#[derive(Serialize, Default)]
pub struct RunReport {
    /// Binary version that produced this report — pins which build a result came
    /// from when comparing runs (no more "is this the new exe?" guessing).
    pub meshsplat_version: String,
    /// Human-readable pipeline descriptor (pose backend → trainer → mesher).
    pub method: String,
    /// Short token burned into this run's output filenames.
    pub tag: String,
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
    /// Write `report.<tag>.json` (a per-method archive, so several methods can
    /// be compared side-by-side in one output dir) plus a stable `report.json`
    /// (latest run) for tooling that reads a fixed path. Returns every file
    /// written.
    pub fn save(&self, out_dir: &Path, tag: &str) -> anyhow::Result<Vec<PathBuf>> {
        let bytes = serde_json::to_vec_pretty(self)?;
        let mut written = Vec::new();
        if !tag.is_empty() {
            let tagged = out_dir.join(format!("report.{tag}.json"));
            std::fs::write(&tagged, &bytes)?;
            written.push(tagged);
        }
        let canonical = out_dir.join("report.json");
        std::fs::write(&canonical, &bytes)?;
        written.push(canonical);
        Ok(written)
    }
}
