use std::path::{Path, PathBuf};
use std::process::Command;

/// A usable COLMAP installation.
#[derive(Debug, Clone)]
pub struct ColmapBinary {
    pub path: PathBuf,
}

impl ColmapBinary {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// `colmap --help` succeeds — cheap sanity check that the binary runs.
    pub fn works(&self) -> bool {
        Command::new(&self.path)
            .arg("help")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Version from the `colmap help` banner ("COLMAP 3.9.1 -- ...");
    /// `--version` only exists on newer releases.
    pub fn version(&self) -> Option<String> {
        let out = Command::new(&self.path).arg("help").output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.contains("COLMAP"))?.trim();
        Some(line.split("--").next().unwrap_or(line).trim().to_owned())
    }

    /// Does this build offer a subcommand (e.g. `global_mapper`)?
    pub fn has_command(&self, command: &str) -> bool {
        Command::new(&self.path)
            .arg("help")
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .any(|l| l.trim().split_whitespace().next() == Some(command))
            })
            .unwrap_or(false)
    }
}

/// The directory where an auto-downloaded COLMAP lives.
pub fn install_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("meshsplat").join("colmap"))
}

fn binary_names() -> &'static [&'static str] {
    if cfg!(windows) {
        // COLMAP.bat sets up the DLL path before calling colmap.exe.
        &["COLMAP.bat", "colmap.bat", "colmap.exe"]
    } else {
        &["colmap"]
    }
}

/// Search a directory tree (max depth 3) for a COLMAP binary.
fn search_dir(dir: &Path, depth: usize) -> Option<PathBuf> {
    for name in binary_names() {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        if p.is_dir()
            && let Some(found) = search_dir(&p, depth - 1)
        {
            return Some(found);
        }
    }
    None
}

/// Locate COLMAP: explicit path → `MESHSPLAT_COLMAP` env → `$PATH` → our cache dir.
pub fn locate(explicit: Option<&Path>) -> Option<ColmapBinary> {
    if let Some(p) = explicit {
        return Some(ColmapBinary::new(p.to_owned()));
    }
    if let Ok(env_path) = std::env::var("MESHSPLAT_COLMAP") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Some(ColmapBinary::new(p));
        }
    }
    for name in binary_names() {
        if let Ok(p) = which::which(name) {
            return Some(ColmapBinary::new(p));
        }
    }
    let cache = install_dir()?;
    search_dir(&cache, 3).map(ColmapBinary::new)
}
