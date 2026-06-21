//! Ingest external point clouds (and OBJ vertex sets) and adapt them to the
//! oriented-sample representation the TSDF mesher consumes.
//!
//! A plain colored point cloud carries no surface orientation or footprint,
//! so we estimate a per-point normal and local spacing (k-NN PCA over a
//! spatial hash) and wrap each point as a thin disc-shaped "gaussian": the
//! flattest axis is the normal, the tangential extent is the local spacing.
//! `extract_mesh` then fuses these exactly as it fuses trained splats — the
//! normal sign is resolved there toward the cameras (or the centroid).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::splat_ply::SplatCloud;

/// A colored point cloud, optionally with per-point normals.
pub struct PointCloud {
    pub positions: Vec<Vec3>,
    pub colors: Vec<[f32; 3]>,
    /// Present only when the source file supplied normals.
    pub normals: Option<Vec<Vec3>>,
}

impl PointCloud {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Load a `.ply` (ASCII or binary little-endian) or `.obj` point cloud.
pub fn load_point_cloud(path: &Path) -> Result<PointCloud> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let pc = match ext.as_deref() {
        Some("ply") => load_ply(path)?,
        Some("obj") => load_obj(path)?,
        other => bail!("Unsupported point-cloud format {other:?} (expected .ply or .obj)"),
    };
    if pc.is_empty() {
        bail!("No points found in {}", path.display());
    }
    log::info!(
        "Loaded {} points from {} ({} normals)",
        pc.len(),
        path.display(),
        if pc.normals.is_some() { "with" } else { "no" }
    );
    Ok(pc)
}

// ── PLY ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
}

impl Ty {
    fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 => 8,
        }
    }
    fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }
    fn read_le(self, b: &[u8], off: usize) -> f32 {
        let b = &b[off..off + self.size()];
        match self {
            Self::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            Self::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::I8 => b[0] as i8 as f32,
            Self::U8 => b[0] as f32,
            Self::I16 => i16::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::U16 => u16::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f32,
            Self::U32 => u32::from_le_bytes(b.try_into().unwrap()) as f32,
        }
    }
}

struct PlyProp {
    name: String,
    ty: Ty,
    offset: usize,
}

/// Find a property by any of the given alias names.
fn find<'a>(props: &'a [PlyProp], names: &[&str]) -> Option<&'a PlyProp> {
    names
        .iter()
        .find_map(|n| props.iter().find(|p| p.name.eq_ignore_ascii_case(n)))
}

fn load_ply(path: &Path) -> Result<PointCloud> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Opening {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        bail!("Not a PLY file (missing 'ply' magic)");
    }

    let mut ascii = false;
    let mut props: Vec<PlyProp> = Vec::new();
    let mut stride = 0usize;
    let mut count = None;
    let mut in_vertex = false;

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            bail!("PLY header ended unexpectedly");
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["format", "ascii", _] => ascii = true,
            ["format", "binary_little_endian", _] => ascii = false,
            ["format", other, _] => bail!("Unsupported PLY format '{other}'"),
            ["element", "vertex", n] => {
                in_vertex = true;
                count = Some(n.parse::<usize>().context("Bad vertex count")?);
            }
            ["element", ..] => in_vertex = false,
            ["property", "list", ..] => {} // face lists etc. — ignored
            ["property", ty, name] if in_vertex => {
                let ty = Ty::parse(ty).with_context(|| format!("Unknown PLY type '{ty}'"))?;
                props.push(PlyProp {
                    name: (*name).to_owned(),
                    ty,
                    offset: stride,
                });
                stride += ty.size();
            }
            ["end_header"] => break,
            _ => {}
        }
    }
    let count = count.context("PLY has no vertex element")?;

    let px = find(&props, &["x"]).context("PLY vertex has no 'x'")?;
    let py = find(&props, &["y"]).context("PLY vertex has no 'y'")?;
    let pz = find(&props, &["z"]).context("PLY vertex has no 'z'")?;
    let cr = find(&props, &["red", "r", "diffuse_red"]);
    let cg = find(&props, &["green", "g", "diffuse_green"]);
    let cb = find(&props, &["blue", "b", "diffuse_blue"]);
    let nx = find(&props, &["nx", "normal_x"]);
    let ny = find(&props, &["ny", "normal_y"]);
    let nz = find(&props, &["nz", "normal_z"]);
    let has_color = cr.is_some() && cg.is_some() && cb.is_some();
    let has_normals = nx.is_some() && ny.is_some() && nz.is_some();

    let mut positions = Vec::with_capacity(count);
    let mut raw_colors: Vec<[f32; 3]> = if has_color { Vec::with_capacity(count) } else { Vec::new() };
    let mut normals: Vec<Vec3> = if has_normals { Vec::with_capacity(count) } else { Vec::new() };
    // Integer colour channels are 0..255 (uchar) / 0..65535; floats are
    // usually already 0..1 but some tools write 0..255 — detect from range.
    let color_is_integer = cr.map(|p| p.ty.is_integer()).unwrap_or(false);

    if ascii {
        let idx_of = |p: &PlyProp| props.iter().position(|q| q.offset == p.offset).unwrap();
        let (ix, iy, iz) = (idx_of(px), idx_of(py), idx_of(pz));
        let ic = has_color.then(|| [idx_of(cr.unwrap()), idx_of(cg.unwrap()), idx_of(cb.unwrap())]);
        let in_ = has_normals.then(|| [idx_of(nx.unwrap()), idx_of(ny.unwrap()), idx_of(nz.unwrap())]);
        let mut read = 0;
        for l in reader.lines() {
            let l = l?;
            let f: Vec<f32> = l.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if f.len() < props.len() {
                continue; // blank or face line
            }
            positions.push(Vec3::new(f[ix], f[iy], f[iz]));
            if let Some([a, b, c]) = ic {
                raw_colors.push([f[a], f[b], f[c]]);
            }
            if let Some([a, b, c]) = in_ {
                normals.push(Vec3::new(f[a], f[b], f[c]));
            }
            read += 1;
            if read == count {
                break;
            }
        }
    } else {
        let mut data = vec![0u8; stride * count];
        reader.read_exact(&mut data).context("PLY ended early — truncated?")?;
        for row in data.chunks_exact(stride) {
            positions.push(Vec3::new(
                px.ty.read_le(row, px.offset),
                py.ty.read_le(row, py.offset),
                pz.ty.read_le(row, pz.offset),
            ));
            if has_color {
                raw_colors.push([
                    cr.unwrap().ty.read_le(row, cr.unwrap().offset),
                    cg.unwrap().ty.read_le(row, cg.unwrap().offset),
                    cb.unwrap().ty.read_le(row, cb.unwrap().offset),
                ]);
            }
            if has_normals {
                normals.push(Vec3::new(
                    nx.unwrap().ty.read_le(row, nx.unwrap().offset),
                    ny.unwrap().ty.read_le(row, ny.unwrap().offset),
                    nz.unwrap().ty.read_le(row, nz.unwrap().offset),
                ));
            }
        }
    }

    Ok(PointCloud {
        colors: finalize_colors(&positions, raw_colors, color_is_integer),
        normals: has_normals.then_some(normals),
        positions,
    })
}

// ── OBJ ─────────────────────────────────────────────────────────────────────

fn load_obj(path: &Path) -> Result<PointCloud> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Opening {}", path.display()))?;
    let mut positions = Vec::new();
    let mut raw_colors = Vec::new();
    let mut any_color = false;
    for l in BufReader::new(file).lines() {
        let l = l?;
        let mut it = l.split_whitespace();
        if it.next() != Some("v") {
            continue; // only vertex positions; faces/normals ignored
        }
        let nums: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
        if nums.len() < 3 {
            continue;
        }
        positions.push(Vec3::new(nums[0], nums[1], nums[2]));
        // Some exporters append per-vertex RGB after xyz.
        if nums.len() >= 6 {
            any_color = true;
            raw_colors.push([nums[3], nums[4], nums[5]]);
        } else {
            raw_colors.push([0.7, 0.7, 0.7]);
        }
    }
    Ok(PointCloud {
        colors: if any_color {
            finalize_colors(&positions, raw_colors, false)
        } else {
            vec![[0.7, 0.7, 0.7]; positions.len()]
        },
        normals: None,
        positions,
    })
}

/// Normalize colors to [0,1], inferring the 0..255 vs 0..1 convention.
fn finalize_colors(positions: &[Vec3], raw: Vec<[f32; 3]>, integer: bool) -> Vec<[f32; 3]> {
    if raw.is_empty() {
        return vec![[0.7, 0.7, 0.7]; positions.len()];
    }
    let max = raw
        .iter()
        .take(4096)
        .flat_map(|c| c.iter())
        .copied()
        .fold(0.0f32, f32::max);
    let scale = if integer || max > 1.5 { 1.0 / 255.0 } else { 1.0 };
    raw.into_iter()
        .map(|c| {
            [
                (c[0] * scale).clamp(0.0, 1.0),
                (c[1] * scale).clamp(0.0, 1.0),
                (c[2] * scale).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

// ── Point cloud → oriented gaussians ────────────────────────────────────────

/// Wrap a point cloud as thin oriented gaussians for the TSDF mesher.
/// Normals come from the file when present, otherwise from k-NN PCA; the
/// local point spacing sets each disc's tangential footprint.
pub fn point_cloud_to_splats(pc: &PointCloud) -> SplatCloud {
    let (est_normals, spacing) = estimate_normals_and_spacing(&pc.positions);
    let normals = pc.normals.as_ref().unwrap_or(&est_normals);

    let n = pc.len();
    let mut cloud = SplatCloud {
        positions: pc.positions.clone(),
        scales: Vec::with_capacity(n),
        rotations: Vec::with_capacity(n),
        opacities: vec![1.0; n],
        colors: pc.colors.clone(),
    };
    for i in 0..n {
        let s = spacing[i].max(1e-6);
        // Min-scale axis is the normal (the mesher reads it back as such);
        // tangential extent ~ local spacing so neighbouring discs overlap.
        cloud.scales.push(Vec3::new(0.25 * s, s, s));
        cloud.rotations.push(basis_from_normal(normals[i]));
    }
    cloud
}

/// A right-handed rotation whose first (x) column is `normal`.
fn basis_from_normal(normal: Vec3) -> Quat {
    let n = normal.normalize_or(Vec3::Z);
    let seed = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = (seed - n * n.dot(seed)).normalize_or(Vec3::Y);
    let t2 = n.cross(t1);
    Quat::from_mat3(&Mat3::from_cols(n, t1, t2))
}

/// Per-point normal (undirected; sign resolved later by the mesher) and mean
/// spacing to nearby points, via PCA over a spatial-hash neighbourhood.
fn estimate_normals_and_spacing(points: &[Vec3]) -> (Vec<Vec3>, Vec<f32>) {
    let n = points.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut min = points[0];
    let mut max = points[0];
    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }
    let diag = (max - min).length().max(1e-6);
    // ~one point per cell on average; a 3³ neighbourhood then yields enough.
    let cell = (diag / (n as f32).cbrt()).max(1e-6);
    let key = |p: Vec3| {
        (
            ((p.x - min.x) / cell).floor() as i32,
            ((p.y - min.y) / cell).floor() as i32,
            ((p.z - min.z) / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::with_capacity(n);
    for (i, &p) in points.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }

    const MAX_NEIGHBORS: usize = 24;
    let global_spacing = cell;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let p = points[i];
            let (kx, ky, kz) = key(p);
            // Gather candidates from the 3³ neighbourhood, keep the nearest few.
            let mut cand: Vec<(f32, Vec3)> = Vec::with_capacity(32);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(ids) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &j in ids {
                                if j as usize == i {
                                    continue;
                                }
                                let q = points[j as usize];
                                cand.push(((q - p).length_squared(), q));
                            }
                        }
                    }
                }
            }
            if cand.len() < 3 {
                return (Vec3::Z, global_spacing);
            }
            cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            cand.truncate(MAX_NEIGHBORS);

            let mut mean = Vec3::ZERO;
            for &(_, q) in &cand {
                mean += q;
            }
            mean /= cand.len() as f32;
            // 3×3 covariance (upper triangle).
            let (mut cxx, mut cyy, mut czz) = (0.0f32, 0.0, 0.0);
            let (mut cxy, mut cxz, mut cyz) = (0.0f32, 0.0, 0.0);
            for &(_, q) in &cand {
                let d = q - mean;
                cxx += d.x * d.x;
                cyy += d.y * d.y;
                czz += d.z * d.z;
                cxy += d.x * d.y;
                cxz += d.x * d.z;
                cyz += d.y * d.z;
            }
            let cov = [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]];
            let normal = smallest_eigenvector(cov);
            let spacing = (cand.iter().map(|&(d2, _)| d2.sqrt()).sum::<f32>()
                / cand.len() as f32)
                .max(1e-6);
            (normal, spacing)
        })
        .unzip()
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3×3 matrix
/// (cyclic Jacobi rotations — robust for any symmetric input).
fn smallest_eigenvector(mut a: [[f32; 3]; 3]) -> Vec3 {
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        // Largest off-diagonal element.
        let (mut p, mut q, mut best) = (0, 1, a[0][1].abs());
        for (i, j) in [(0, 2), (1, 2)] {
            if a[i][j].abs() > best {
                best = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if best < 1e-9 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // Apply rotation J^T A J and accumulate V J.
        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let eig = [a[0][0], a[1][1], a[2][2]];
    let mut min_i = 0;
    for i in 1..3 {
        if eig[i] < eig[min_i] {
            min_i = i;
        }
    }
    Vec3::new(v[0][min_i], v[1][min_i], v[2][min_i]).normalize_or(Vec3::Z)
}

// ── Camera centers (for normal orientation) ─────────────────────────────────

/// Best-effort camera positions from a text file: one camera per line, the
/// position taken as the first three numeric columns (handles COLMAP-style and
/// Metashape "Export Cameras" rows, which lead with a label/id then x y z).
pub fn load_camera_centers(path: &Path) -> Result<Vec<Vec3>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Opening cameras file {}", path.display()))?;
    let mut centers = Vec::new();
    for l in BufReader::new(file).lines() {
        let l = l?;
        let trimmed = l.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let nums: Vec<f32> = l
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter_map(|s| s.parse::<f32>().ok())
            .filter(|f| f.is_finite())
            .collect();
        if nums.len() >= 3 {
            centers.push(Vec3::new(nums[0], nums[1], nums[2]));
        }
    }
    if centers.is_empty() {
        bail!(
            "No camera positions parsed from {} (expected one camera per line \
             with x y z in the first three numeric columns)",
            path.display()
        );
    }
    log::info!("Loaded {} camera positions from {}", centers.len(), path.display());
    Ok(centers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_of_a_plane_points_along_z() {
        // Points on the z=0 plane: the smallest-variance axis is ±z.
        let mut pts = Vec::new();
        for x in -3..=3 {
            for y in -3..=3 {
                pts.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let (normals, spacing) = estimate_normals_and_spacing(&pts);
        let center = pts.iter().position(|p| *p == Vec3::ZERO).unwrap();
        assert!(normals[center].normalize().z.abs() > 0.9, "{:?}", normals[center]);
        assert!(spacing[center] > 0.5 && spacing[center] < 2.5, "{}", spacing[center]);
    }

    #[test]
    fn basis_first_axis_is_the_normal() {
        let n = Vec3::new(0.3, -0.6, 0.74).normalize();
        let q = basis_from_normal(n);
        assert!((Mat3::from_quat(q).x_axis - n).length() < 1e-5);
    }

    #[test]
    fn ascii_ply_with_colors_round_trips() {
        let dir = std::env::temp_dir().join(format!("pc-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.ply");
        std::fs::write(
            &p,
            "ply\nformat ascii 1.0\nelement vertex 2\n\
             property float x\nproperty float y\nproperty float z\n\
             property uchar red\nproperty uchar green\nproperty uchar blue\n\
             end_header\n0 0 0 255 0 0\n1 2 3 0 128 255\n",
        )
        .unwrap();
        let pc = load_point_cloud(&p).unwrap();
        assert_eq!(pc.len(), 2);
        assert!((pc.colors[0][0] - 1.0).abs() < 1e-6);
        assert!((pc.positions[1] - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert!(pc.normals.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn camera_centers_skip_labels_and_comments() {
        let dir = std::env::temp_dir().join(format!("pc-cam-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("cams.txt");
        std::fs::write(&p, "# label x y z ...\nIMG_001 1.0 2.0 3.0 0.1 0.2\nIMG_002, 4, 5, 6\n").unwrap();
        let c = load_camera_centers(&p).unwrap();
        assert_eq!(c.len(), 2);
        assert!((c[0] - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert!((c[1] - Vec3::new(4.0, 5.0, 6.0)).length() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
