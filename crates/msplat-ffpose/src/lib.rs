//! Feed-forward camera pose estimation via MapAnything.
//!
//! Classical SfM needs ~60–80% overlap between views; feed-forward pose
//! transformers reconstruct sparse, low-overlap and object-centric captures
//! where COLMAP registers almost nothing. They are PyTorch research code, so
//! this crate keeps them out of the meshsplat binary: a pinned PEP 723 Python
//! script (auto-provisioned by `uv`) runs MapAnything — or MASt3R through
//! MapAnything's wrapper — and exports a COLMAP-format model, which is then
//! assembled into the same Brush-compatible dataset layout `msplat-colmap`
//! produces. The rest of the pipeline never knows the difference.

mod locate;
mod pipeline;
mod script;

pub use locate::{PyRuntime, locate};
pub use pipeline::{FfBackend, FfposeOptions, ModelChoice, run_ffpose};
