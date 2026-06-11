use glam::Vec3;

/// Naive surface nets over a sampled SDF.
///
/// `sample(x, y, z)` returns the signed distance at a grid corner, or `None`
/// where the field is unknown (no splat coverage) — cells touching unknown
/// corners produce no geometry, which suppresses the phantom "back shell"
/// that plain TSDF extraction creates behind unobserved surfaces.
///
/// Returns vertex positions in grid (voxel) coordinates plus triangle indices.
pub fn surface_nets(
    dims: &[usize; 3],
    sample: impl Fn(usize, usize, usize) -> Option<f32> + Sync,
) -> (Vec<Vec3>, Vec<u32>) {
    let [nx, ny, nz] = *dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return (Vec::new(), Vec::new());
    }
    // Cells are indexed by their lowest corner; cell grid is one smaller.
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let cell_index = |x: usize, y: usize, z: usize| (z * cy + y) * cx + x;

    // Corner offsets in (x, y, z), bit i of the corner id selects axis i.
    const CORNERS: [(usize, usize, usize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ];
    // The 12 cell edges as corner-id pairs.
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut cell_vertex = vec![u32::MAX; cx * cy * cz];
    let mut positions: Vec<Vec3> = Vec::new();

    // Pass 1: one vertex per sign-changing cell, at the mean of edge crossings.
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                let mut values = [0.0f32; 8];
                let mut ok = true;
                for (ci, (dx, dy, dz)) in CORNERS.iter().enumerate() {
                    match sample(x + dx, y + dy, z + dz) {
                        Some(v) => values[ci] = v,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }

                let mut crossing_sum = Vec3::ZERO;
                let mut crossings = 0u32;
                for &(a, b) in &EDGES {
                    let (va, vb) = (values[a], values[b]);
                    if (va < 0.0) == (vb < 0.0) {
                        continue;
                    }
                    let t = va / (va - vb);
                    let pa = CORNERS[a];
                    let pb = CORNERS[b];
                    let pa = Vec3::new(pa.0 as f32, pa.1 as f32, pa.2 as f32);
                    let pb = Vec3::new(pb.0 as f32, pb.1 as f32, pb.2 as f32);
                    crossing_sum += pa.lerp(pb, t);
                    crossings += 1;
                }
                if crossings == 0 {
                    continue;
                }

                cell_vertex[cell_index(x, y, z)] = positions.len() as u32;
                positions.push(
                    Vec3::new(x as f32, y as f32, z as f32) + crossing_sum / crossings as f32,
                );
            }
        }
    }

    // Pass 2: for every grid edge with a sign change, connect the 4 cells
    // sharing that edge into a quad (two triangles). We visit, per cell, the
    // three edges leaving its (x+1, y+1, z+1)-most... — concretely: the edges
    // along +x, +y, +z from corner (x, y, z), handled so each grid edge is
    // visited exactly once by the cell that owns its lowest corner.
    let mut indices: Vec<u32> = Vec::new();
    let value_at = |x: usize, y: usize, z: usize| sample(x, y, z);

    for z in 1..cz {
        for y in 1..cy {
            for x in 1..cx {
                let Some(v0) = value_at(x, y, z) else {
                    continue;
                };
                let inside = v0 < 0.0;

                // Axis d: edge from (x,y,z) to (x,y,z)+e_d. The four cells
                // around that edge differ in the two other axes by -1/0.
                for (d, (ex, ey, ez)) in
                    [(0usize, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))]
                {
                    let Some(v1) = value_at(x + ex, y + ey, z + ez) else {
                        continue;
                    };
                    if (v1 < 0.0) == inside {
                        continue;
                    }

                    let quad = match d {
                        0 => [
                            cell_vertex[cell_index(x, y, z)],
                            cell_vertex[cell_index(x, y - 1, z)],
                            cell_vertex[cell_index(x, y - 1, z - 1)],
                            cell_vertex[cell_index(x, y, z - 1)],
                        ],
                        1 => [
                            cell_vertex[cell_index(x, y, z)],
                            cell_vertex[cell_index(x, y, z - 1)],
                            cell_vertex[cell_index(x - 1, y, z - 1)],
                            cell_vertex[cell_index(x - 1, y, z)],
                        ],
                        _ => [
                            cell_vertex[cell_index(x, y, z)],
                            cell_vertex[cell_index(x - 1, y, z)],
                            cell_vertex[cell_index(x - 1, y - 1, z)],
                            cell_vertex[cell_index(x, y - 1, z)],
                        ],
                    };
                    if quad.contains(&u32::MAX) {
                        continue;
                    }

                    // Wind so triangles face out of the negative (inside) region.
                    let [a, b, c, e] = quad;
                    if inside {
                        indices.extend([a, b, c, a, c, e]);
                    } else {
                        indices.extend([a, c, b, a, e, c]);
                    }
                }
            }
        }
    }

    (positions, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere SDF must produce a closed, watertight-ish mesh with sane bounds.
    #[test]
    fn sphere_mesh() {
        let dims = [32, 32, 32];
        let center = Vec3::splat(15.5);
        let radius = 10.0;
        let (verts, tris) = surface_nets(&dims, |x, y, z| {
            Some((Vec3::new(x as f32, y as f32, z as f32) - center).length() - radius)
        });

        assert!(!verts.is_empty(), "sphere produced no vertices");
        assert_eq!(tris.len() % 3, 0);
        assert!(!tris.is_empty(), "sphere produced no triangles");

        for v in &verts {
            let r = (*v - center).length();
            assert!(
                (r - radius).abs() < 1.0,
                "vertex at radius {r}, expected ~{radius}"
            );
        }
        // Euler characteristic of a sphere: V - E + F = 2 (E = 3F/2 for tris).
        let f = (tris.len() / 3) as i64;
        let mut edges = std::collections::HashSet::new();
        for t in tris.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.insert((a.min(b), a.max(b)));
            }
        }
        let euler = verts.len() as i64 - edges.len() as i64 + f;
        assert_eq!(euler, 2, "sphere mesh is not closed");

        // Signed volume (divergence theorem) checks outward winding and size.
        let vol: f64 = tris
            .chunks_exact(3)
            .map(|t| {
                let a = verts[t[0] as usize] - center;
                let b = verts[t[1] as usize] - center;
                let c = verts[t[2] as usize] - center;
                a.cross(b).dot(c) as f64 / 6.0
            })
            .sum();
        let expected = 4.0 / 3.0 * std::f64::consts::PI * (radius as f64).powi(3);
        assert!(
            vol > 0.85 * expected && vol < 1.15 * expected,
            "sphere volume {vol:.1}, expected ~{expected:.1} (negative → winding flipped)"
        );
    }

    /// Unknown samples must suppress geometry rather than crash or emit junk.
    #[test]
    fn unknown_regions_are_skipped() {
        let dims = [16, 16, 16];
        let (verts, tris) = surface_nets(&dims, |x, _, _| {
            if x > 8 {
                None
            } else {
                Some(x as f32 - 4.5)
            }
        });
        assert!(!verts.is_empty());
        // All vertices sit on the known x≈4.5 plane.
        for v in &verts {
            assert!((v.x - 4.5).abs() < 0.6, "vertex off the plane: {v}");
        }
        assert_eq!(tris.len() % 3, 0);
    }
}
