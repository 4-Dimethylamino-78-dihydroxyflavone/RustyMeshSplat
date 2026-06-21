use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::pipeline::FfBackend;

const SCRIPT: &str = include_str!("../python/mapanything_poses.py");

/// FNV-1a, enough to content-address the materialized script so upgraded
/// binaries never run a stale cached copy.
fn fnv1a(data: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Write the embedded script (with the backend's pip extras substituted into
/// its PEP 723 header) into the user cache and return its path.
/// `MESHSPLAT_FFPOSE_SCRIPT` overrides entirely — the seam for tests and for
/// users who want to patch the script.
pub fn materialize_script(backend: FfBackend) -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("MESHSPLAT_FFPOSE_SCRIPT") {
        let p = PathBuf::from(custom);
        anyhow::ensure!(
            p.is_file(),
            "MESHSPLAT_FFPOSE_SCRIPT points at {}, which is not a file",
            p.display()
        );
        return Ok(p);
    }

    // The script writes COLMAP text itself, so the heavy/native `colmap`
    // extra (pycolmap, open3d, lightglue) is never needed. Only the MASt3R
    // backend pulls an extra — and its non-commercial code only enters the
    // resolved environment when that backend is actually requested.
    let extras = match backend {
        FfBackend::MapAnything => "",
        FfBackend::Mast3r => "[mast3r]",
    };
    let body = SCRIPT.replace("{{EXTRAS}}", extras);

    let dir = cache_dir().context("No user cache directory available")?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("mapanything_poses-{:016x}.py", fnv1a(&body)));
    if !path.is_file() {
        std::fs::write(&path, &body)
            .with_context(|| format!("Writing {}", path.display()))?;
    }
    Ok(path)
}

/// Where materialized scripts and model checkpoints live.
pub fn cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("meshsplat").join("ffpose"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extras_are_substituted_per_backend() {
        assert!(SCRIPT.contains("{{EXTRAS}}"), "template token missing");
        // MapAnything needs no pip extra; only MASt3R pulls one.
        let map_anything = SCRIPT.replace("{{EXTRAS}}", "");
        let mast3r = SCRIPT.replace("{{EXTRAS}}", "[mast3r]");
        assert!(map_anything.contains("mapanything @ "));
        assert!(mast3r.contains("mapanything[mast3r] @ "));
        // Different content must produce different cache filenames.
        assert_ne!(fnv1a(&map_anything), fnv1a(&mast3r));
    }
}
