use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::{DataSource, create_process};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio_stream::StreamExt;

pub struct TrainSettings {
    pub iters: u32,
    pub max_splats: u32,
    pub max_resolution: u32,
    /// Spherical-harmonics degree for view-dependent colour (3 = standard 3DGS;
    /// 0 = flat colour). Higher costs more memory per splat.
    pub sh_degree: u32,
    pub seed: u64,
    /// Absolute directory the trained splat PLY is exported into.
    pub export_dir: PathBuf,
}

pub struct TrainOutcome {
    pub splat_path: PathBuf,
    pub final_splats: u32,
    pub last_psnr: Option<f32>,
    pub elapsed: Duration,
}

const SPLAT_FILE: &str = "splat.ply";

/// `element vertex N` from a PLY header.
fn ply_vertex_count(path: &Path) -> Option<u32> {
    use std::io::BufRead;
    let reader = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    for line in reader.lines().take(64) {
        let line = line.ok()?;
        if line.trim() == "end_header" {
            break;
        }
        if let Some(rest) = line.trim().strip_prefix("element vertex ") {
            return rest.trim().parse().ok();
        }
    }
    None
}

/// Train a gaussian splat on a Brush-compatible dataset directory
/// (COLMAP `images/` + `sparse/0/`, or nerfstudio `transforms.json`).
pub async fn train_splat(
    dataset_dir: &Path,
    settings: TrainSettings,
    multi: &MultiProgress,
) -> Result<TrainOutcome> {
    let started = std::time::Instant::now();
    tokio::fs::create_dir_all(&settings.export_dir).await?;

    let export_dir = settings.export_dir.clone();
    let iters = settings.iters;
    let mut process = create_process(
        DataSource::Path(dataset_dir.display().to_string()),
        move |mut config| async move {
            // `config` starts from the dataset's args.txt (if present) or
            // defaults; pipeline settings override it.
            config.train_config.total_train_iters = settings.iters;
            config.train_config.max_splats = settings.max_splats;
            config.load_config.max_resolution = settings.max_resolution;
            // Brush only emits view-dependent SH coefficients when the initial
            // splats carry the degree; default leaves it at 0 (flat colour).
            config.model_config.sh_degree = settings.sh_degree;
            config.process_config.seed = settings.seed;
            config.process_config.export_path = export_dir.display().to_string();
            config.process_config.export_name = SPLAT_FILE.to_owned();
            // Export only the final model (the last step always exports).
            config.process_config.export_every = settings.iters.max(1);
            Some(config)
        },
    );

    // The GPU was already brought up (and registered with Brush) by
    // gpu::init_for_training before any pipeline stage ran.
    let bar = multi.add(
        ProgressBar::new(iters as u64).with_style(
            ProgressStyle::with_template(
                "  [{elapsed_precise}] {bar:38.cyan/dim} {pos}/{len} steps ({per_sec}, ~{eta} left) {msg}",
            )
            .expect("static template")
            .progress_chars("=> "),
        ),
    );
    bar.enable_steady_tick(Duration::from_millis(500));

    let mut final_splats = 0u32;
    let mut last_psnr = None;

    while let Some(msg) = process.stream.next().await {
        match msg? {
            ProcessMessage::StartLoading { training, .. } => {
                if !training {
                    anyhow::bail!(
                        "Dataset at {} loaded as a viewable splat, not a trainable dataset",
                        dataset_dir.display()
                    );
                }
                bar.set_message("loading dataset");
            }
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::Dataset { dataset } => {
                    log::info!(
                        "Dataset loaded: {} training views",
                        dataset.train.views.len()
                    );
                    bar.set_message(format!("{} views", dataset.train.views.len()));
                }
                TrainMessage::TrainStep { iter, .. } => {
                    bar.set_position(iter as u64);
                }
                TrainMessage::RefineStep {
                    cur_splat_count, ..
                } => {
                    final_splats = cur_splat_count;
                    bar.set_message(format!("{cur_splat_count} splats"));
                }
                TrainMessage::EvalResult { avg_psnr, .. } => {
                    last_psnr = Some(avg_psnr);
                }
                TrainMessage::DoneTraining => {
                    bar.set_position(iters as u64);
                }
                TrainMessage::TrainConfig { .. } => {}
            },
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
            }
            _ => {}
        }
    }
    bar.finish_with_message("done");

    let splat_path = settings.export_dir.join(SPLAT_FILE);
    if !splat_path.is_file() {
        anyhow::bail!(
            "Training finished but no splat was exported to {}",
            splat_path.display()
        );
    }
    // Refine messages only fire every few hundred iters; for short runs read
    // the authoritative count from the exported PLY header.
    if final_splats == 0 {
        final_splats = ply_vertex_count(&splat_path).unwrap_or(0);
    }
    Ok(TrainOutcome {
        splat_path,
        final_splats,
        last_psnr,
        elapsed: started.elapsed(),
    })
}
