//! Camera pose estimation for uncalibrated photo directories.
//!
//! Wraps the COLMAP binary (BSD-3, no CUDA required): locates an existing
//! install, auto-downloads the official prebuilt on Windows, runs the
//! feature-extract / match / map pipeline with streamed progress, and
//! assembles a Brush-compatible dataset directory (images + sparse/0).

mod fetch;
mod locate;
mod pipeline;
mod sparse;

pub use fetch::{FetchEvent, ensure_colmap};
pub use locate::{ColmapBinary, locate as locate_colmap};
pub use pipeline::{
    CaptureMode, MapperKind, SfmEvent, SfmOptions, SfmOutput, SfmStats, run_sfm,
};
pub use sparse::{SparseModel, read_sparse_model};

/// Image extensions COLMAP can ingest.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "bmp"];

/// List the images directly inside `dir` (non-recursive), sorted by name.
pub fn list_images(dir: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut images: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    images.sort();
    Ok(images)
}
