use std::collections::HashMap;

use glam::Vec3;
use rayon::prelude::*;

use crate::splat_ply::SplatCloud;

/// Per-vertex colors from the nearest opaque gaussian, found via a uniform
/// hash grid over splat centers (searched in expanding shells).
pub fn transfer_colors(
    cloud: &SplatCloud,
    kept: &[usize],
    vertices: &[Vec3],
    cell_size: f32,
) -> Vec<[f32; 3]> {
    let cell_of = |p: Vec3| -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
    };

    let mut buckets: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for &i in kept {
        buckets
            .entry(cell_of(cloud.positions[i]))
            .or_default()
            .push(i as u32);
    }

    vertices
        .par_iter()
        .map(|&v| {
            let (cx, cy, cz) = cell_of(v);
            let mut best_d2 = f32::INFINITY;
            let mut best: Option<usize> = None;
            // Shell 1 almost always hits; shell 3 is the give-up radius.
            for shell in 1..=3i32 {
                for dz in -shell..=shell {
                    for dy in -shell..=shell {
                        for dx in -shell..=shell {
                            // Only the new outer shell on later passes.
                            if shell > 1
                                && dx.abs() < shell
                                && dy.abs() < shell
                                && dz.abs() < shell
                            {
                                continue;
                            }
                            let Some(ids) = buckets.get(&(cx + dx, cy + dy, cz + dz)) else {
                                continue;
                            };
                            for &i in ids {
                                let d2 = (cloud.positions[i as usize] - v).length_squared();
                                if d2 < best_d2 {
                                    best_d2 = d2;
                                    best = Some(i as usize);
                                }
                            }
                        }
                    }
                }
                if best.is_some() {
                    break;
                }
            }
            best.map_or([0.6, 0.6, 0.6], |i| cloud.colors[i])
        })
        .collect()
}
