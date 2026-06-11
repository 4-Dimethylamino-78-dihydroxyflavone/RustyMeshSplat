use std::path::{Path, PathBuf};
use std::process::Command;

/// A way to run the bundled pose-estimation script.
#[derive(Debug, Clone)]
pub enum PyRuntime {
    /// `uv run <script>` — resolves and caches the pinned Python environment
    /// automatically. The preferred runtime.
    Uv(PathBuf),
    /// A user-supplied Python that already has `mapanything` installed
    /// (`MESHSPLAT_FFPOSE_PYTHON`). Escape hatch for machines without uv.
    Python(PathBuf),
}

impl PyRuntime {
    /// `--version` succeeds — cheap sanity check that the runtime runs.
    pub fn works(&self) -> bool {
        Command::new(self.binary())
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    pub fn binary(&self) -> &Path {
        match self {
            Self::Uv(p) | Self::Python(p) => p,
        }
    }

    pub fn describe(&self) -> String {
        let version = Command::new(self.binary())
            .arg("--version")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .filter(|v| !v.is_empty());
        match (self, version) {
            (Self::Uv(p), Some(v)) => format!("{v} ({})", p.display()),
            (Self::Uv(p), None) => format!("uv ({})", p.display()),
            (Self::Python(p), Some(v)) => format!("{v} ({}, MESHSPLAT_FFPOSE_PYTHON)", p.display()),
            (Self::Python(p), None) => format!("python ({}, MESHSPLAT_FFPOSE_PYTHON)", p.display()),
        }
    }
}

/// Locate a runtime: explicit `--uv` path → `MESHSPLAT_UV` env → `uv` on
/// `$PATH` → `MESHSPLAT_FFPOSE_PYTHON` env (pre-provisioned environment).
pub fn locate(explicit_uv: Option<&Path>) -> Option<PyRuntime> {
    if let Some(p) = explicit_uv {
        return Some(PyRuntime::Uv(p.to_owned()));
    }
    if let Ok(env_path) = std::env::var("MESHSPLAT_UV") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Some(PyRuntime::Uv(p));
        }
    }
    if let Ok(p) = which::which("uv") {
        return Some(PyRuntime::Uv(p));
    }
    if let Ok(env_path) = std::env::var("MESHSPLAT_FFPOSE_PYTHON") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Some(PyRuntime::Python(p));
        }
        // Allow bare binary names ("python3") too.
        if let Ok(found) = which::which(&p) {
            return Some(PyRuntime::Python(found));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env vars are process-global; exercise every branch in one test to
    // avoid races between parallel test threads.
    #[test]
    fn locate_resolution_order() {
        // SAFETY: single-threaded within this test; no other test in this
        // crate touches these variables.
        unsafe {
            std::env::remove_var("MESHSPLAT_UV");
            std::env::remove_var("MESHSPLAT_FFPOSE_PYTHON");
        }

        // Explicit path wins unconditionally, even if it doesn't exist.
        let explicit = locate(Some(Path::new("/nonexistent/uv")));
        assert!(matches!(explicit, Some(PyRuntime::Uv(p)) if p == Path::new("/nonexistent/uv")));

        // MESHSPLAT_FFPOSE_PYTHON is honored when uv is absent. Use a file
        // that certainly exists to hit the is_file() branch.
        let this_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/locate.rs");
        unsafe {
            std::env::set_var("MESHSPLAT_FFPOSE_PYTHON", &this_file);
        }
        match locate(None) {
            // A real uv on PATH takes precedence over the python fallback.
            Some(PyRuntime::Uv(_)) => assert!(which::which("uv").is_ok()),
            Some(PyRuntime::Python(p)) => assert_eq!(p, this_file),
            None => panic!("python fallback should have been found"),
        }
        unsafe {
            std::env::remove_var("MESHSPLAT_FFPOSE_PYTHON");
        }
    }
}
