use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::locate::ColmapBinary;
use crate::sparse::read_sparse_model;

/// How the photos were captured. Drives the matching strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    /// Exhaustive matching for small sets, sequential for large ordered sets —
    /// and if a large sequential pass registers under half the photos, it
    /// auto-escalates to exhaustive matching once.
    Auto,
    /// Unordered photos of a region/object from arbitrary viewpoints.
    Unordered,
    /// Ordered captures: video frames, turntable / rotated-specimen rigs.
    Sequential,
}

impl std::str::FromStr for CaptureMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "unordered" => Ok(Self::Unordered),
            "sequential" | "turntable" | "video" => Ok(Self::Sequential),
            other => Err(format!(
                "unknown capture mode '{other}' (expected auto|unordered|sequential)"
            )),
        }
    }
}

/// Which COLMAP reconstruction algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapperKind {
    /// Incremental mapping — the proven default.
    Incremental,
    /// Global mapping (GLOMAP, merged into COLMAP in 4.0.0): 1–2 orders of
    /// magnitude faster at comparable accuracy. Requires a COLMAP build that
    /// exposes the `global_mapper` command; older builds error out rather than
    /// silently using the incremental mapper.
    Global,
}

impl std::str::FromStr for MapperKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "incremental" => Ok(Self::Incremental),
            "global" | "glomap" => Ok(Self::Global),
            other => Err(format!(
                "unknown mapper '{other}' (expected incremental|global)"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SfmOptions {
    pub capture: CaptureMode,
    /// All photos share one physical camera+lens (turntable rigs, single drone).
    pub single_camera: bool,
    /// Force CPU feature extraction/matching (skip the GPU attempt).
    pub cpu_only: bool,
    /// Downscale images larger than this for feature extraction.
    pub max_image_size: u32,
    /// Above this image count, Auto switches from exhaustive to sequential matching.
    pub exhaustive_limit: usize,
    /// Directory of per-image masks (`<image>.png`, black = ignored).
    /// Essential for turntable captures: masking the static background keeps
    /// SfM locked onto the rotating object instead of the room.
    pub mask_dir: Option<PathBuf>,
    pub mapper: MapperKind,
}

impl Default for SfmOptions {
    fn default() -> Self {
        Self {
            capture: CaptureMode::Auto,
            single_camera: false,
            cpu_only: false,
            max_image_size: 3200,
            exhaustive_limit: 150,
            mask_dir: None,
            mapper: MapperKind::Incremental,
        }
    }
}

/// Progress events from the SfM pipeline.
pub enum SfmEvent {
    StageStarted {
        name: String,
        index: usize,
        total: usize,
    },
    /// Determinate progress within the current stage.
    Progress {
        done: u64,
        total: u64,
    },
    /// One-line status for spinners (e.g. "Registering image #42").
    Status(String),
}

pub struct SfmStats {
    pub input_images: usize,
    pub registered_images: usize,
    pub sparse_points: usize,
    pub elapsed: Duration,
}

pub struct SfmOutput {
    /// Brush-compatible dataset: `<dataset_dir>/images/` + `<dataset_dir>/sparse/0/`.
    pub dataset_dir: PathBuf,
    /// The chosen reconstruction (`sparse/0` inside the dataset dir).
    pub sparse_dir: PathBuf,
    pub stats: SfmStats,
}

/// Run feature extraction → matching → mapping on `images_dir`, writing
/// intermediates under `work_dir` and assembling a dataset directory.
pub async fn run_sfm(
    colmap: &ColmapBinary,
    images_dir: &Path,
    work_dir: &Path,
    opts: &SfmOptions,
    mut on_event: impl FnMut(SfmEvent),
) -> Result<SfmOutput> {
    let started = Instant::now();
    let images = crate::list_images(images_dir)?;
    if images.len() < 3 {
        bail!(
            "Need at least 3 images to reconstruct, found {} in {}",
            images.len(),
            images_dir.display()
        );
    }

    tokio::fs::create_dir_all(work_dir).await?;
    let db_path = work_dir.join("database.db");
    // A stale database makes COLMAP re-use old features; start clean.
    let _ = tokio::fs::remove_file(&db_path).await;
    let sparse_out = work_dir.join("sparse");
    tokio::fs::create_dir_all(&sparse_out).await?;

    let mut exhaustive = match opts.capture {
        CaptureMode::Unordered => true,
        CaptureMode::Sequential => false,
        CaptureMode::Auto => images.len() <= opts.exhaustive_limit,
    };
    let total_stages = 3;

    // Stage 1: feature extraction.
    on_event(SfmEvent::StageStarted {
        name: "Extracting features".to_owned(),
        index: 1,
        total: total_stages,
    });
    let mut extract_args = vec![
        "feature_extractor".to_owned(),
        "--database_path".to_owned(),
        db_path.display().to_string(),
        "--image_path".to_owned(),
        images_dir.display().to_string(),
        "--ImageReader.camera_model".to_owned(),
        "SIMPLE_RADIAL".to_owned(),
        "--ImageReader.single_camera".to_owned(),
        if opts.single_camera { "1" } else { "0" }.to_owned(),
        "--SiftExtraction.max_image_size".to_owned(),
        opts.max_image_size.to_string(),
    ];
    if let Some(mask_dir) = &opts.mask_dir {
        if !mask_dir.is_dir() {
            bail!("--masks directory {} does not exist", mask_dir.display());
        }
        extract_args.extend([
            "--ImageReader.mask_path".to_owned(),
            mask_dir.display().to_string(),
        ]);
    }
    let n_images = images.len() as u64;
    run_with_gpu_fallback(
        colmap,
        &mut extract_args,
        "--SiftExtraction.use_gpu",
        opts.cpu_only,
        |line| report_extract_progress(line, n_images, &mut on_event),
    )
    .await
    .context("COLMAP feature extraction failed")?;

    // Stages 2 & 3 loop together so a collapsed sequential pass can auto-escalate
    // to exhaustive matching. Feature extraction (Stage 1) is reused — only
    // matching + mapping repeat — and each attempt maps into its own dir so the
    // larger reconstruction wins even if the retry somehow does worse.
    let mut best: Option<(PathBuf, crate::sparse::SparseModel)> = None;
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;

        // Stage 2: matching.
        on_event(SfmEvent::StageStarted {
            name: if exhaustive {
                "Matching images (exhaustive)"
            } else {
                "Matching images (sequential)"
            }
            .to_owned(),
            index: 2,
            total: total_stages,
        });
        let mut match_args = if exhaustive {
            vec![
                "exhaustive_matcher".to_owned(),
                "--database_path".to_owned(),
                db_path.display().to_string(),
            ]
        } else {
            vec![
                "sequential_matcher".to_owned(),
                "--database_path".to_owned(),
                db_path.display().to_string(),
                "--SequentialMatching.overlap".to_owned(),
                "15".to_owned(),
            ]
        };
        run_with_gpu_fallback(
            colmap,
            &mut match_args,
            "--SiftMatching.use_gpu",
            opts.cpu_only,
            |line| report_match_progress(line, &mut on_event),
        )
        .await
        .context("COLMAP feature matching failed")?;

        // Stage 3: mapping (bundle adjustment included).
        on_event(SfmEvent::StageStarted {
            name: "Reconstructing camera poses".to_owned(),
            index: 3,
            total: total_stages,
        });
        let mapper_cmd = match opts.mapper {
            MapperKind::Global if colmap.has_command("global_mapper") => "global_mapper",
            // GLOMAP was merged into COLMAP in 4.0.0 (2026-03) and the standalone
            // glomap project is archived. Builds without it have no global mapper;
            // fail loudly instead of silently downgrading to incremental, which
            // would quietly ignore what the user asked for.
            MapperKind::Global => bail!(
                "--sfm-mapper global needs COLMAP 4.0+ (the GLOMAP global SfM \
                 pipeline was merged into COLMAP in 4.0.0; the standalone glomap \
                 project is now archived). This COLMAP build exposes no \
                 `global_mapper` command — upgrade COLMAP, or rerun with \
                 --sfm-mapper incremental."
            ),
            MapperKind::Incremental => "mapper",
        };
        let attempt_dir = sparse_out.join(format!("run{attempt}"));
        // A stale model dir would confuse the mapper and pick_best_model.
        let _ = tokio::fs::remove_dir_all(&attempt_dir).await;
        tokio::fs::create_dir_all(&attempt_dir).await?;
        let map_args = vec![
            mapper_cmd.to_owned(),
            "--database_path".to_owned(),
            db_path.display().to_string(),
            "--image_path".to_owned(),
            images_dir.display().to_string(),
            "--output_path".to_owned(),
            attempt_dir.display().to_string(),
        ];
        run_colmap(colmap, &map_args, |line| {
            report_mapper_progress(line, n_images, &mut on_event)
        })
        .await
        .context("COLMAP mapping failed")?;

        // The mapper may emit several disconnected models (run/0, run/1, ...).
        // Keep this attempt's largest, and across attempts the largest overall.
        let attempt_model_dir = pick_best_model(&attempt_dir).await?;
        let attempt_model = read_sparse_model(&attempt_model_dir).await?;
        let registered = attempt_model.images.len();
        let improved = match &best {
            Some((_, prev)) => registered > prev.images.len(),
            None => true,
        };
        if improved {
            best = Some((attempt_model_dir, attempt_model));
        }
        let best_registered = best.as_ref().map_or(0, |(_, m)| m.images.len());

        // Auto-escalation: when Auto picked sequential and it registered less
        // than half the photos, the views are likely multi-session / unordered —
        // retry once with exhaustive matching to bridge the disconnected graph
        // instead of handing back a near-empty reconstruction.
        let collapsed = best_registered < images.len() / 2;
        if attempt == 1 && !exhaustive && opts.capture == CaptureMode::Auto && collapsed {
            log::warn!(
                "Sequential matching registered only {best_registered}/{} images; \
                 retrying with exhaustive matching to bridge disconnected views.",
                images.len()
            );
            on_event(SfmEvent::Status(
                "Sequential matching underperformed — retrying exhaustively".to_owned(),
            ));
            exhaustive = true;
            continue;
        }
        break;
    }

    let (best_model, model) = best.expect("at least one mapping attempt ran");
    if model.images.is_empty() {
        bail!(
            "COLMAP could not register any cameras. The photos may lack overlap, \
             texture, or coverage. Try more photos with ~70% overlap between views."
        );
    }
    if model.images.len() < images.len() / 2 {
        log::warn!(
            "Only {}/{} images were registered; reconstruction may be partial.",
            model.images.len(),
            images.len()
        );
    }

    on_event(SfmEvent::Status("Assembling dataset".to_owned()));
    let dataset_dir = work_dir.join("dataset");
    let sparse_dir = dataset_dir.join("sparse").join("0");
    assemble_dataset(images_dir, &best_model, &dataset_dir).await?;

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

/// COLMAP prints `Processed file [123/456]` during extraction.
fn report_extract_progress(line: &str, total: u64, on_event: &mut impl FnMut(SfmEvent)) {
    if let Some(done) = parse_bracket_progress(line, "Processed file [") {
        on_event(SfmEvent::Progress { done, total });
    }
}

/// Matchers print `Matching block [1/9, 3/9]` or `Matching image [12/100]`.
fn report_match_progress(line: &str, on_event: &mut impl FnMut(SfmEvent)) {
    let trimmed = line.trim();
    if trimmed.starts_with("Matching block [") || trimmed.starts_with("Matching image [") {
        on_event(SfmEvent::Status(trimmed.to_owned()));
    }
}

/// The mapper prints `Registering image #42 (17)` as it grows the model.
fn report_mapper_progress(line: &str, total: u64, on_event: &mut impl FnMut(SfmEvent)) {
    let trimmed = line.trim();
    if trimmed.starts_with("Registering image #") {
        // The parenthesised number is how many images are registered so far.
        if let Some(done) = trimmed
            .rsplit_once('(')
            .and_then(|(_, tail)| tail.trim_end_matches(')').trim().parse::<u64>().ok())
        {
            on_event(SfmEvent::Progress { done, total });
        }
        on_event(SfmEvent::Status(trimmed.to_owned()));
    } else if trimmed.starts_with("Global bundle adjustment") {
        on_event(SfmEvent::Status("Global bundle adjustment".to_owned()));
    }
}

/// Parse `prefix[N/M]`-style lines, returning N.
fn parse_bracket_progress(line: &str, prefix: &str) -> Option<u64> {
    let rest = line.trim().strip_prefix(prefix)?;
    let (n, _) = rest.split_once('/')?;
    n.trim().parse().ok()
}

/// Run COLMAP with `gpu_flag 1`; if that fails (no GL context, headless box,
/// driver issues) transparently retry on CPU.
async fn run_with_gpu_fallback(
    colmap: &ColmapBinary,
    args: &mut Vec<String>,
    gpu_flag: &str,
    cpu_only: bool,
    mut on_line: impl FnMut(&str),
) -> Result<()> {
    if cpu_only {
        args.extend([gpu_flag.to_owned(), "0".to_owned()]);
        return run_colmap(colmap, args, &mut on_line).await;
    }
    let mut gpu_args = args.clone();
    gpu_args.extend([gpu_flag.to_owned(), "1".to_owned()]);
    match run_colmap(colmap, &gpu_args, &mut on_line).await {
        Ok(()) => Ok(()),
        Err(err) => {
            log::warn!("COLMAP GPU path failed ({err}); retrying on CPU");
            args.extend([gpu_flag.to_owned(), "0".to_owned()]);
            run_colmap(colmap, args, &mut on_line).await
        }
    }
}

/// Spawn COLMAP, stream stdout+stderr lines into `on_line`, fail on non-zero exit.
async fn run_colmap(
    colmap: &ColmapBinary,
    args: &[String],
    mut on_line: impl FnMut(&str),
) -> Result<()> {
    log::debug!("Running: {} {}", colmap.path.display(), args.join(" "));
    let mut child = Command::new(&colmap.path)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Spawning {}", colmap.path.display()))?;

    let stdout = child.stdout.take().context("No stdout from COLMAP")?;
    let stderr = child.stderr.take().context("No stderr from COLMAP")?;
    let mut out_lines = BufReader::new(stdout).lines();
    let mut err_lines = BufReader::new(stderr).lines();
    // Keep the error tail: COLMAP reports the actual cause on stderr.
    let mut err_tail: Vec<String> = Vec::new();

    loop {
        tokio::select! {
            line = out_lines.next_line() => match line? {
                Some(l) => { log::trace!("colmap: {l}"); on_line(&l); }
                None => break,
            },
            line = err_lines.next_line() => if let Some(l) = line? {
                log::debug!("colmap[err]: {l}");
                err_tail.push(l);
                if err_tail.len() > 20 { err_tail.remove(0); }
            },
        }
    }
    // Drain whatever stderr remains after stdout closed.
    while let Some(l) = err_lines.next_line().await? {
        err_tail.push(l);
        if err_tail.len() > 20 {
            err_tail.remove(0);
        }
    }

    let status = child.wait().await?;
    if !status.success() {
        bail!(
            "colmap {} exited with {status}:\n{}",
            args.first().map(String::as_str).unwrap_or(""),
            err_tail.join("\n")
        );
    }
    Ok(())
}

/// Choose the reconstruction that registered the most images.
async fn pick_best_model(sparse_out: &Path) -> Result<PathBuf> {
    let mut entries = tokio::fs::read_dir(sparse_out).await?;
    let mut candidates = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let p = entry.path();
        if p.is_dir() && p.join("images.bin").is_file() {
            let size = tokio::fs::metadata(p.join("images.bin")).await?.len();
            candidates.push((size, p));
        }
    }
    if candidates.is_empty() {
        bail!(
            "COLMAP produced no reconstruction in {}. The photos may not have \
             enough overlap or texture.",
            sparse_out.display()
        );
    }
    if candidates.len() > 1 {
        log::warn!(
            "COLMAP split the scene into {} disconnected models; using the largest. \
             Capturing with more overlap avoids this.",
            candidates.len()
        );
    }
    candidates.sort_by_key(|(size, _)| *size);
    Ok(candidates.pop().expect("non-empty").1)
}

/// Build `<dataset>/images/` + `<dataset>/sparse/0/` the way Brush expects.
/// Images are hard-linked when possible, copied otherwise.
async fn assemble_dataset(images_dir: &Path, model_dir: &Path, dataset_dir: &Path) -> Result<()> {
    let img_dest = dataset_dir.join("images");
    let sparse_dest = dataset_dir.join("sparse").join("0");
    // Re-runs leave stale state behind; rebuild from scratch.
    if dataset_dir.exists() {
        tokio::fs::remove_dir_all(dataset_dir).await?;
    }
    tokio::fs::create_dir_all(&img_dest).await?;
    tokio::fs::create_dir_all(&sparse_dest).await?;

    for name in ["cameras.bin", "images.bin", "points3D.bin"] {
        tokio::fs::copy(model_dir.join(name), sparse_dest.join(name))
            .await
            .with_context(|| format!("Copying {name}"))?;
    }

    for image in crate::list_images(images_dir)? {
        let Some(name) = image.file_name() else {
            continue;
        };
        let dest = img_dest.join(name);
        if tokio::fs::hard_link(&image, &dest).await.is_err() {
            tokio::fs::copy(&image, &dest).await.with_context(|| {
                format!("Copying {} into dataset", image.display())
            })?;
        }
    }
    Ok(())
}
