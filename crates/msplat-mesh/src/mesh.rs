use glam::Vec3;

/// An indexed triangle mesh with per-vertex normals and colors.
pub struct TriMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Linear RGB in [0, 1].
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl TriMesh {
    pub fn num_vertices(&self) -> usize {
        self.positions.len()
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Uniform Laplacian smoothing (umbrella operator), `iters` passes.
    pub fn laplacian_smooth(&mut self, iters: u32, lambda: f32) {
        if iters == 0 || self.positions.is_empty() {
            return;
        }
        // Neighbor sums via the edge list; recomputed once, reused per pass.
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); self.positions.len()];
        for tri in self.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                neighbors[u as usize].push(v);
                neighbors[v as usize].push(u);
            }
        }
        for nbrs in &mut neighbors {
            nbrs.sort_unstable();
            nbrs.dedup();
        }

        let mut next = self.positions.clone();
        for _ in 0..iters {
            for (i, nbrs) in neighbors.iter().enumerate() {
                if nbrs.is_empty() {
                    continue;
                }
                let mean = nbrs
                    .iter()
                    .fold(Vec3::ZERO, |acc, &n| acc + self.positions[n as usize])
                    / nbrs.len() as f32;
                next[i] = self.positions[i].lerp(mean, lambda);
            }
            std::mem::swap(&mut self.positions, &mut next);
        }
    }

    /// Area-weighted vertex normals from the triangle faces.
    pub fn recompute_normals(&mut self) {
        let mut normals = vec![Vec3::ZERO; self.positions.len()];
        for tri in self.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            // Cross product length is proportional to area — free weighting.
            let n = (self.positions[b] - self.positions[a])
                .cross(self.positions[c] - self.positions[a]);
            normals[a] += n;
            normals[b] += n;
            normals[c] += n;
        }
        self.normals = normals
            .into_iter()
            .map(|n| n.normalize_or(Vec3::Z))
            .collect();
    }
}
