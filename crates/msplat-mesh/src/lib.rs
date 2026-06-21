//! Mesh extraction from trained gaussian splats.
//!
//! Portable (CPU + rayon) pipeline: load a 3DGS-format PLY, fuse the
//! opacity-filtered, surface-oriented gaussians into a truncated signed
//! distance field, extract a surface with naive surface nets, transfer
//! splat colors to vertices, and export PLY / OBJ / GLB.

mod color;
mod export;
mod mesh;
mod point_cloud;
mod splat_ply;
mod surface_nets;
mod tsdf;

pub use export::{write_glb, write_obj, write_ply};
pub use mesh::TriMesh;
pub use point_cloud::{
    PointCloud, load_camera_centers, load_point_cloud, point_cloud_to_splats,
};
pub use splat_ply::{SplatCloud, load_splat_ply};
pub use tsdf::{ExtractProgress, MeshParams, extract_mesh};
