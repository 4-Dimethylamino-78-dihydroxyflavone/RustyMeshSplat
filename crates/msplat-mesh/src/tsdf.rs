use anyhow::{Result, bail};
use glam::{Mat3, Vec3};
use rayon::prelude::*;

use crate::color::transfer_colors;
use crate::mesh::TriMesh;
use crate::splat_ply::SplatCloud;
use crate::surface_nets::surface_nets;

#[derive(Debug, Clone)]
pub struct MeshParams {
    /// Voxels along the longest axis of the (robust) bounding box.
    pub grid_res: u32,
    /// Ignore gaussians more transparent than this.
    pub opacity_min: f32,
    /// TSDF truncation band, in voxels.
    pub trunc_voxels: f32,
    /// Robust bounding box: drop this fraction of outliers per axis end.
    pub bbox_percentile: f32,
    /// Voxels need this much accumulated splat weight to define a surface.
    pub min_weight: f32,
    /// Laplacian smoothing passes on the extracted mesh.
    pub smooth_iters: u32,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            grid_res: 256,
            opacity_min: 0.35,
            trunc_voxels: 2.5,
            bbox_percentile: 0.01,
            min_weight: 0.25,
            smooth_iters: 2,
        }
    }
}

pub enum ExtractProgress {
    Phase(&'static str),
    /// Fusion progress, 0..=1.
    Fusing(f32),
}

/// Voxel value: weighted-average signed distance + accumulated weight.
#[derive(Clone, Copy, Default)]
struct Voxel {
    sdf: f32,
    weight: f32,
}

struct Grid {
    dims: [usize; 3],
    origin: Vec3,
    voxel: f32,
}

impl Grid {
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    fn center(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin + Vec3::new(x as f32, y as f32, z as f32) * self.voxel
    }

    fn num_voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

/// Extract a triangle mesh from a trained splat cloud.
///
/// `camera_centers` (when available) orient the gaussian normals toward the
/// side the scene was photographed from; without them normals are oriented
/// away from the cloud centroid.
pub fn extract_mesh(
    cloud: &SplatCloud,
    camera_centers: &[Vec3],
    params: &MeshParams,
    mut on_progress: impl FnMut(ExtractProgress) + Send,
) -> Result<TriMesh> {
    on_progress(ExtractProgress::Phase("Filtering gaussians"));
    let kept: Vec<usize> = (0..cloud.len())
        .filter(|&i| cloud.opacities[i] >= params.opacity_min)
        .collect();
    if kept.len() < 100 {
        bail!(
            "Only {} gaussians pass the opacity threshold {} — splat too sparse to mesh. \
             Try lowering --opacity-min.",
            kept.len(),
            params.opacity_min
        );
    }
    log::info!(
        "Meshing from {}/{} gaussians (opacity ≥ {})",
        kept.len(),
        cloud.len(),
        params.opacity_min
    );

    let grid = build_grid(cloud, &kept, params)?;
    log::info!(
        "TSDF grid {}×{}×{} (voxel {:.4})",
        grid.dims[0],
        grid.dims[1],
        grid.dims[2],
        grid.voxel
    );

    on_progress(ExtractProgress::Phase("Fusing TSDF"));
    let voxels = fuse(cloud, &kept, camera_centers, &grid, params, &mut on_progress);

    on_progress(ExtractProgress::Phase("Extracting surface"));
    let trunc = params.trunc_voxels * grid.voxel;
    let sdf_at = |i: usize| -> Option<f32> {
        let v = voxels[i];
        (v.weight >= params.min_weight).then(|| (v.sdf / v.weight).clamp(-trunc, trunc))
    };
    let (positions, indices) = surface_nets(&grid.dims, |x, y, z| sdf_at(grid.index(x, y, z)));
    if positions.is_empty() {
        bail!(
            "No surface found. The splat may be too sparse or the opacity \
             threshold too strict (try --opacity-min 0.1 or a higher --grid-res)."
        );
    }

    // Surface nets emits vertices in voxel coordinates; map to world space.
    let positions: Vec<Vec3> = positions
        .into_iter()
        .map(|p| grid.origin + p * grid.voxel)
        .collect();

    let mut mesh = TriMesh {
        positions,
        normals: Vec::new(),
        colors: Vec::new(),
        indices,
    };

    on_progress(ExtractProgress::Phase("Smoothing"));
    mesh.laplacian_smooth(params.smooth_iters, 0.5);
    mesh.recompute_normals();

    on_progress(ExtractProgress::Phase("Transferring colors"));
    mesh.colors = transfer_colors(cloud, &kept, &mesh.positions, grid.voxel);

    log::info!(
        "Extracted mesh: {} vertices, {} triangles",
        mesh.num_vertices(),
        mesh.num_triangles()
    );
    Ok(mesh)
}

/// Robust bounding box from per-axis percentiles, expanded by a few voxels.
fn build_grid(cloud: &SplatCloud, kept: &[usize], params: &MeshParams) -> Result<Grid> {
    // Sampling keeps the percentile sort cheap on multi-million splat clouds.
    let stride = (kept.len() / 500_000).max(1);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for &i in kept.iter().step_by(stride) {
        let p = cloud.positions[i];
        xs.push(p.x);
        ys.push(p.y);
        zs.push(p.z);
    }
    let percentile_range = |vals: &mut Vec<f32>| {
        vals.sort_unstable_by(f32::total_cmp);
        let lo = ((vals.len() - 1) as f32 * params.bbox_percentile) as usize;
        let hi = ((vals.len() - 1) as f32 * (1.0 - params.bbox_percentile)) as usize;
        (vals[lo], vals[hi])
    };
    let (x0, x1) = percentile_range(&mut xs);
    let (y0, y1) = percentile_range(&mut ys);
    let (z0, z1) = percentile_range(&mut zs);

    let min = Vec3::new(x0, y0, z0);
    let max = Vec3::new(x1, y1, z1);
    let extent = max - min;
    let longest = extent.max_element();
    if !(longest.is_finite() && longest > 0.0) {
        bail!("Degenerate splat bounding box: {min} .. {max}");
    }

    let res = params.grid_res.max(16) as f32;
    let voxel = longest / res;
    // Pad by the truncation band so border surfaces close properly.
    let pad = (params.trunc_voxels + 2.0) * voxel;
    let min = min - pad;
    let max = max + pad;
    let dims = [
        ((max.x - min.x) / voxel).ceil() as usize + 1,
        ((max.y - min.y) / voxel).ceil() as usize + 1,
        ((max.z - min.z) / voxel).ceil() as usize + 1,
    ];

    let grid = Grid {
        dims,
        origin: min,
        voxel,
    };
    let bytes = grid.num_voxels() * std::mem::size_of::<Voxel>();
    if bytes > 6 * 1024 * 1024 * 1024 {
        bail!(
            "TSDF grid would need {} GB; lower --grid-res",
            bytes / (1024 * 1024 * 1024)
        );
    }
    if bytes > 1024 * 1024 * 1024 {
        log::warn!("Large TSDF grid: {} MB", bytes / (1024 * 1024));
    }
    Ok(grid)
}

/// Splat-to-TSDF fusion. Each gaussian contributes a signed-distance plane
/// sample along its shortest (normal) axis, weighted by opacity and a
/// tangential gaussian falloff. Parallelism: splats are bucketed into z-slabs
/// of the grid and each slab is fused independently (slabs own disjoint,
/// contiguous voxel ranges, so no locks are needed).
fn fuse(
    cloud: &SplatCloud,
    kept: &[usize],
    camera_centers: &[Vec3],
    grid: &Grid,
    params: &MeshParams,
    on_progress: &mut (impl FnMut(ExtractProgress) + Send),
) -> Vec<Voxel> {
    let [nx, ny, nz] = grid.dims;
    let trunc = params.trunc_voxels * grid.voxel;

    let centroid = kept
        .iter()
        .fold(Vec3::ZERO, |acc, &i| acc + cloud.positions[i])
        / kept.len().max(1) as f32;

    // Precompute per-splat fusion inputs: oriented normal, tangential sigma,
    // and the world-space half-extent of the influence box.
    struct Footprint {
        pos: Vec3,
        normal: Vec3,
        sigma_t: f32,
        half_extent: f32,
        weight: f32,
        z_min: usize,
        z_max: usize,
    }

    let footprints: Vec<Footprint> = kept
        .par_iter()
        .map(|&i| {
            let pos = cloud.positions[i];
            let scale = cloud.scales[i];
            let rot = Mat3::from_quat(cloud.rotations[i]);

            // The gaussian's flattest direction approximates the surface normal.
            let (axis, s_min) = if scale.x <= scale.y && scale.x <= scale.z {
                (rot.x_axis, scale.x)
            } else if scale.y <= scale.z {
                (rot.y_axis, scale.y)
            } else {
                (rot.z_axis, scale.z)
            };
            let _ = s_min;
            let s_max = scale.max_element();

            // Orient toward the photographed side.
            let toward = nearest_camera_dir(camera_centers, pos, centroid);
            let normal = if axis.dot(toward) < 0.0 { -axis } else { axis };

            let sigma_t = s_max.max(grid.voxel);
            // Cap the tangential footprint so huge background blobs don't
            // dominate runtime; surface coverage comes from splat density.
            let tangent_reach = (2.0 * sigma_t).min(4.0 * grid.voxel);
            let half_extent = tangent_reach.max(trunc);

            let z_lo = (pos.z - half_extent - grid.origin.z) / grid.voxel;
            let z_hi = (pos.z + half_extent - grid.origin.z) / grid.voxel;
            Footprint {
                pos,
                normal,
                sigma_t,
                half_extent,
                weight: cloud.opacities[i],
                z_min: (z_lo.floor().max(0.0) as usize).min(nz - 1),
                z_max: (z_hi.ceil().max(0.0) as usize).min(nz - 1),
            }
        })
        .collect();

    // Bucket splats into z-slabs.
    const SLAB: usize = 8;
    let num_slabs = nz.div_ceil(SLAB);
    let mut slab_splats: Vec<Vec<u32>> = vec![Vec::new(); num_slabs];
    for (fi, fp) in footprints.iter().enumerate() {
        for bucket in &mut slab_splats[(fp.z_min / SLAB)..=(fp.z_max / SLAB)] {
            bucket.push(fi as u32);
        }
    }

    let mut voxels = vec![Voxel::default(); grid.num_voxels()];
    let slab_len = nx * ny * SLAB;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let progress_cb = std::sync::Mutex::new(&mut *on_progress);

    voxels
        .par_chunks_mut(slab_len)
        .enumerate()
        .for_each(|(slab, slab_voxels)| {
            let z_base = slab * SLAB;
            for &fi in &slab_splats[slab] {
                let fp = &footprints[fi as usize];
                let h = fp.half_extent;
                let x0 = (((fp.pos.x - h - grid.origin.x) / grid.voxel).floor().max(0.0)) as usize;
                let x1 = ((((fp.pos.x + h - grid.origin.x) / grid.voxel).ceil()) as usize).min(nx - 1);
                let y0 = (((fp.pos.y - h - grid.origin.y) / grid.voxel).floor().max(0.0)) as usize;
                let y1 = ((((fp.pos.y + h - grid.origin.y) / grid.voxel).ceil()) as usize).min(ny - 1);
                let z0 = fp.z_min.max(z_base);
                let z1 = fp.z_max.min((z_base + SLAB - 1).min(nz - 1));
                if x0 > x1 || y0 > y1 || z0 > z1 {
                    continue;
                }

                let inv_2sig2 = 0.5 / (fp.sigma_t * fp.sigma_t);
                for z in z0..=z1 {
                    for y in y0..=y1 {
                        let row = (z - z_base) * ny * nx + y * nx;
                        for x in x0..=x1 {
                            let d = grid.center(x, y, z) - fp.pos;
                            let sdf = d.dot(fp.normal);
                            if sdf.abs() > trunc {
                                continue;
                            }
                            let dt2 = (d.length_squared() - sdf * sdf).max(0.0);
                            let w = fp.weight * (-dt2 * inv_2sig2).exp();
                            if w < 1e-3 {
                                continue;
                            }
                            let v = &mut slab_voxels[row + x];
                            v.sdf += w * sdf;
                            v.weight += w;
                        }
                    }
                }
            }
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if let Ok(mut cb) = progress_cb.lock() {
                cb(ExtractProgress::Fusing(n as f32 / num_slabs as f32));
            }
        });

    voxels
}

/// Direction from `pos` toward the nearest camera, or away from the cloud
/// centroid when no cameras are known.
fn nearest_camera_dir(camera_centers: &[Vec3], pos: Vec3, centroid: Vec3) -> Vec3 {
    let mut best = f32::INFINITY;
    let mut dir = pos - centroid;
    for &c in camera_centers {
        let d2 = (c - pos).length_squared();
        if d2 < best {
            best = d2;
            dir = c - pos;
        }
    }
    dir.normalize_or(Vec3::Z)
}
