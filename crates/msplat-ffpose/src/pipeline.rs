use std::path::Path;
use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use msplat_colmap::{SfmEvent, SfmOutput, SfmStats, read_sparse_model};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::locate::PyRuntime;
use crate::script::{cache_dir, materialize_script};

/// Which feed-forward model produces the poses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfBackend {
    /// MapAnything (Meta; Apache-2.0 code, Apache or CC BY-NC weights).
    MapAnything,
    /// MASt3R + sparse global alignment via MapAnything's wrapper
    /// (Naver; CC BY-NC-SA — non-commercial use only).
    Mast3r,
}

impl FfBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::MapAnything => "mapanything",
            Self::Mast3r => "mast3r",
        }
    }
}

impl std::str::FromStr for FfBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mapanything" | "map-anything" => Ok(Self::MapAnything),
            "mast3r" => Ok(Self::Mast3r),
            other => Err(format!(
                "unknown pose backend '{other}' (expected colmap|mapanything|mast3r)"
            )),
        }
    }
}

/// Which MapAnything weights to load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelChoice {
    /// `facebook/map-anything-apache` — commercially usable (default).
    Apache,
    /// `facebook/map-anything` — best quality, CC BY-NC (non-commercial).
    Best,
    /// Any explicit Hugging Face model id.
    Custom(String),
}

impl ModelChoice {
    pub fn hf_id(&self) -> &str {
        match self {
            Self::Apache => "facebook/map-anything-apache",
            Self::Best => "facebook/map-anything",
            Self::Custom(id) => id,
        }
    }

    /// Weights that may not be used commercially.
    pub fn non_commercial(&self) -> bool {
        matches!(self, Self::Best)
    }
}

impl std::str::FromStr for ModelChoice {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "apache" => Self::Apache,
            "best" => Self::Best,
            _ => Self::Custom(s.to_owned()),
        })
    }
}

#[derive(Debug, Clone)]
pub struct FfposeOptions {
    pub backend: FfBackend,
    pub model: ModelChoice,
    /// "auto", "cuda", "mps" or "cpu".
    pub device: String,
    /// Proceed on CPU instead of erroring when no GPU is available.
    pub allow_cpu: bool,
    /// Feed EXIF 35mm-equivalent focal lengths to the model as intrinsics.
    pub exif_intrinsics: bool,
}

impl Default for FfposeOptions {
    fn default() -> Self {
        Self {
            backend: FfBackend::MapAnything,
            model: ModelChoice::Apache,
            device: "auto".to_owned(),
            allow_cpu: false,
            exif_intrinsics: true,
        }
    }
}

/// One JSON line on the script's stdout.
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ScriptEvent {
    Stage { name: String, index: usize, total: usize },
    Progress { done: u64, total: u64 },
    Status { message: String },
    Error { message: String, hint: Option<String> },
    Done { views: usize },
}

/// Run the feed-forward script on `images_dir`, writing intermediates under
/// `work_dir` and assembling the same Brush-compatible dataset layout
/// `msplat_colmap::run_sfm` produces.
pub async fn run_ffpose(
    runtime: &PyRuntime,
    images_dir: &Path,
    work_dir: &Path,
    opts: &FfposeOptions,
    mut on_event: impl FnMut(SfmEvent),
) -> Result<SfmOutput> {
    let started = Instant::now();
    let images = msplat_colmap::list_images(images_dir)?;
    if images.len() < 2 {
        bail!(
            "Need at least 2 images for feed-forward pose estimation, found {} in {}",
            images.len(),
            images_dir.display()
        );
    }

    let script = materialize_script(opts.backend)?;
    let raw_dir = work_dir.join("raw");
    // Stale exports would mix with fresh ones; start clean.
    if raw_dir.exists() {
        tokio::fs::remove_dir_all(&raw_dir).await?;
    }
    tokio::fs::create_dir_all(&raw_dir).await?;

    let mut cmd = Command::new(runtime.binary());
    match runtime {
        PyRuntime::Uv(_) => {
            cmd.args(["run", "--no-progress"]).arg(&script);
        }
        PyRuntime::Python(_) => {
            cmd.arg(&script);
        }
    }
    cmd.arg("--images-dir")
        .arg(images_dir)
        .arg("--output-dir")
        .arg(&raw_dir)
        .args(["--backend", opts.backend.name()])
        .args(["--model-id", opts.model.hf_id()])
        .args(["--device", &opts.device]);
    if opts.allow_cpu {
        cmd.arg("--allow-cpu");
    }
    if !opts.exif_intrinsics {
        cmd.arg("--no-exif-intrinsics");
    }
    if let Some(cache) = cache_dir() {
        cmd.arg("--cache-dir").arg(cache);
    }

    run_script(cmd, runtime, &mut on_event)
        .await
        .with_context(|| format!("{} pose estimation failed", opts.backend.name()))?;

    on_event(SfmEvent::Status("Assembling dataset".to_owned()));
    let dataset_dir = work_dir.join("dataset");
    let sparse_dir = dataset_dir.join("sparse").join("0");
    assemble_dataset(&raw_dir, &dataset_dir).await?;

    let model = read_sparse_model(&sparse_dir).await?;
    if model.images.is_empty() {
        bail!("The model produced no camera poses — the export is empty.");
    }

    Ok(SfmOutput {
        sparse_dir,
        dataset_dir,
        stats: SfmStats {
            input_images: images.len(),
            registered_images: model.images.len(),
            sparse_points: model.points.len(),
            elapsed: started.elapsed(),
        },
    })
}

/// Spawn the script, stream stdout JSON events into `on_event`, keep a stderr
/// tail for error context (mirrors `msplat-colmap`'s COLMAP runner).
async fn run_script(
    mut cmd: Command,
    runtime: &PyRuntime,
    on_event: &mut impl FnMut(SfmEvent),
) -> Result<()> {
    log::debug!("Running: {cmd:?}");
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .with_context(|| format!("Spawning {}", runtime.binary().display()))?;

    let stdout = child.stdout.take().context("No stdout from script")?;
    let stderr = child.stderr.take().context("No stderr from script")?;
    let mut out_lines = BufReader::new(stdout).lines();
    let mut err_lines = BufReader::new(stderr).lines();
    let mut err_tail: Vec<String> = Vec::new();
    // The script reports failures as a structured event before exiting;
    // prefer that over a raw stderr dump.
    let mut script_error: Option<(String, Option<String>)> = None;

    let push_err = |tail: &mut Vec<String>, line: String| {
        log::debug!("ffpose[err]: {line}");
        tail.push(line);
        if tail.len() > 20 {
            tail.remove(0);
        }
    };

    loop {
        tokio::select! {
            line = out_lines.next_line() => match line? {
                Some(l) => handle_line(&l, on_event, &mut script_error),
                None => break,
            },
            line = err_lines.next_line() => if let Some(l) = line? {
                push_err(&mut err_tail, l);
            },
        }
    }
    while let Some(l) = err_lines.next_line().await? {
        push_err(&mut err_tail, l);
    }

    let status = child.wait().await?;
    if !status.success() {
        if let Some((message, hint)) = script_error {
            match hint {
                Some(hint) => bail!("{message}\n  hint: {hint}"),
                None => bail!("{message}"),
            }
        }
        bail!(
            "pose-estimation script exited with {status}:\n{}",
            err_tail.join("\n")
        );
    }
    Ok(())
}

fn handle_line(
    line: &str,
    on_event: &mut impl FnMut(SfmEvent),
    script_error: &mut Option<(String, Option<String>)>,
) {
    match serde_json::from_str::<ScriptEvent>(line) {
        Ok(ScriptEvent::Stage { name, index, total }) => {
            on_event(SfmEvent::StageStarted { name, index, total });
        }
        Ok(ScriptEvent::Progress { done, total }) => {
            on_event(SfmEvent::Progress { done, total });
        }
        Ok(ScriptEvent::Status { message }) => on_event(SfmEvent::Status(message)),
        Ok(ScriptEvent::Error { message, hint }) => *script_error = Some((message, hint)),
        Ok(ScriptEvent::Done { views }) => {
            log::debug!("script exported {views} views");
        }
        // uv resolution notes or stray prints: not part of the protocol.
        Err(_) => log::debug!("ffpose: {line}"),
    }
}

/// Build `<dataset>/images/` + `<dataset>/sparse/0/` from the script's raw
/// export (`raw/images/` + `raw/sparse/`). The exported images are the
/// model-resolution copies the predicted intrinsics actually describe, so
/// they — not the originals — go into the dataset. Accepts binary or text
/// models and skips extras like `points.ply`.
async fn assemble_dataset(raw_dir: &Path, dataset_dir: &Path) -> Result<()> {
    let img_src = raw_dir.join("images");
    let sparse_src = raw_dir.join("sparse");
    let img_dest = dataset_dir.join("images");
    let sparse_dest = dataset_dir.join("sparse").join("0");
    if dataset_dir.exists() {
        tokio::fs::remove_dir_all(dataset_dir).await?;
    }
    tokio::fs::create_dir_all(&img_dest).await?;
    tokio::fs::create_dir_all(&sparse_dest).await?;

    for stem in ["cameras", "images", "points3D"] {
        let mut copied = false;
        for ext in ["bin", "txt"] {
            let src = sparse_src.join(format!("{stem}.{ext}"));
            if src.is_file() {
                tokio::fs::copy(&src, sparse_dest.join(format!("{stem}.{ext}")))
                    .await
                    .with_context(|| format!("Copying {stem}.{ext}"))?;
                copied = true;
                break;
            }
        }
        if !copied {
            bail!(
                "The export at {} has no {stem}.bin/.txt — incomplete COLMAP model",
                sparse_src.display()
            );
        }
    }

    for image in msplat_colmap::list_images(&img_src)? {
        let Some(name) = image.file_name() else {
            continue;
        };
        let dest = img_dest.join(name);
        if tokio::fs::hard_link(&image, &dest).await.is_err() {
            tokio::fs::copy(&image, &dest)
                .await
                .with_context(|| format!("Copying {} into dataset", image.display()))?;
        }
    }
    Ok(())
}
