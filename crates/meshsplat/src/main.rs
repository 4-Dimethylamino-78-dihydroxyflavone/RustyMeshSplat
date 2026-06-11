//! meshsplat — throw a directory of photos at it, get a splat + mesh back.
//!
//! Pipeline: COLMAP camera poses → Brush gaussian splat training on the
//! universal wgpu GPU layer (Vulkan / DX12 / Metal — no CUDA anywhere) →
//! portable mesh extraction (TSDF + surface nets).

#![recursion_limit = "256"]

mod gpu;
mod report;
mod train;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use msplat_colmap::{CaptureMode, FetchEvent, SfmEvent, SfmOptions};
use msplat_mesh::{ExtractProgress, MeshParams};
use report::{MeshReport, RunReport, SfmReport, TrainReport};
use train::{TrainSettings, train_splat};

#[derive(Parser)]
#[command(
    name = "meshsplat",
    version,
    about = "Photos in, gaussian splat + mesh out. Any GPU, no CUDA.",
    arg_required_else_help = true
)]
struct Cli {
    /// A directory of photos, a prepared COLMAP/nerfstudio dataset,
    /// or (with --mesh-only) a trained splat .ply.
    input: Option<PathBuf>,

    /// Output directory for the splat, meshes and report.
    #[arg(short, long, default_value = "./meshsplat_out")]
    output: PathBuf,

    /// Quality preset; individual flags below override it.
    #[arg(long, value_enum, default_value_t = Quality::Balanced)]
    quality: Quality,

    /// Training iterations.
    #[arg(long)]
    iters: Option<u32>,
    /// Max number of gaussians.
    #[arg(long)]
    max_splats: Option<u32>,
    /// Max training image resolution (larger photos are downscaled).
    #[arg(long)]
    max_resolution: Option<u32>,

    /// How the photos were captured: auto, unordered, sequential (turntable/video).
    #[arg(long, default_value = "auto")]
    capture: String,
    /// All photos share one camera + lens (turntable rigs, single drone).
    #[arg(long)]
    single_camera: bool,
    /// Force CPU feature extraction/matching in COLMAP.
    #[arg(long)]
    cpu_sfm: bool,
    /// Path to a COLMAP binary (otherwise auto-detected / auto-downloaded).
    #[arg(long)]
    colmap: Option<PathBuf>,
    /// Directory of per-image masks (black = ignored, e.g. "photo.jpg.png").
    /// Strongly recommended for turntable captures: mask the static
    /// background so SfM locks onto the rotating object.
    #[arg(long)]
    masks: Option<PathBuf>,
    /// Pose reconstruction algorithm: incremental (default, proven) or
    /// global (GLOMAP, much faster on COLMAP 4+; auto-falls back).
    #[arg(long, default_value = "incremental")]
    sfm_mapper: String,

    /// Stop after training: produce only the splat, skip mesh extraction.
    #[arg(long)]
    splat_only: bool,
    /// Treat the input as a trained splat .ply and only extract the mesh.
    #[arg(long)]
    mesh_only: bool,
    /// With --mesh-only: a COLMAP sparse dir (sparse/0) for camera-aware normals.
    #[arg(long)]
    sparse: Option<PathBuf>,

    /// TSDF grid resolution along the longest axis.
    #[arg(long)]
    grid_res: Option<u32>,
    /// Ignore gaussians more transparent than this when meshing.
    #[arg(long)]
    opacity_min: Option<f32>,
    /// Laplacian smoothing passes on the mesh.
    #[arg(long)]
    smooth_iters: Option<u32>,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// GPU backend for training: auto (recommended), vulkan, dx12, metal, gl.
    /// Auto tries every backend in reliability order and falls through on
    /// failure instead of crashing.
    #[arg(long, default_value = "auto")]
    gpu_backend: String,
    /// Pin a specific adapter by its index in the `--doctor` list.
    #[arg(long)]
    gpu_index: Option<usize>,
    /// Permit software rasterizers (WARP/llvmpipe). Extremely slow, but
    /// produces a result on machines with no working GPU at all.
    #[arg(long)]
    allow_software: bool,

    /// Print GPU + COLMAP diagnostics and exit.
    #[arg(long)]
    doctor: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Quality {
    /// Quick preview: fewer iterations, smaller images and grid.
    Fast,
    /// Good default trade-off.
    Balanced,
    /// Maximum fidelity; slowest.
    High,
}

struct Preset {
    iters: u32,
    max_splats: u32,
    max_resolution: u32,
    grid_res: u32,
}

impl Quality {
    fn preset(self) -> Preset {
        match self {
            Self::Fast => Preset {
                iters: 7_000,
                max_splats: 1_500_000,
                max_resolution: 1024,
                grid_res: 192,
            },
            Self::Balanced => Preset {
                iters: 18_000,
                max_splats: 4_000_000,
                max_resolution: 1600,
                grid_res: 256,
            },
            Self::High => Preset {
                iters: 30_000,
                max_splats: 8_000_000,
                max_resolution: 1920,
                grid_res: 320,
            },
        }
    }
}

/// What the input path turned out to be.
enum InputKind {
    /// Raw photos: run the full SfM → train → mesh pipeline.
    Photos(PathBuf),
    /// Already-posed dataset (COLMAP sparse/ or nerfstudio transforms.json).
    Dataset(PathBuf),
    /// A trained splat checkpoint.
    SplatPly(PathBuf),
}

fn classify_input(path: &Path, mesh_only: bool) -> Result<InputKind> {
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("ply") {
            return Ok(InputKind::SplatPly(path.to_owned()));
        }
        bail!(
            "Input file {} is not a .ply — pass a photo directory or a splat file",
            path.display()
        );
    }
    if !path.is_dir() {
        bail!("Input path {} does not exist", path.display());
    }
    if mesh_only {
        bail!("--mesh-only expects a splat .ply file as input");
    }
    let has_sparse = path.join("sparse").is_dir();
    let has_nerfstudio = path.join("transforms.json").is_file()
        || path.join("transforms_train.json").is_file();
    if has_sparse || has_nerfstudio {
        return Ok(InputKind::Dataset(path.to_owned()));
    }
    let images = msplat_colmap::list_images(path)?;
    if images.is_empty() {
        bail!(
            "No images found in {} (looked for {}). Point meshsplat at a folder of photos.",
            path.display(),
            msplat_colmap::IMAGE_EXTENSIONS.join("/")
        );
    }
    Ok(InputKind::Photos(path.to_owned()))
}

fn spinner(multi: &MultiProgress, msg: String) -> ProgressBar {
    let bar = multi.add(
        ProgressBar::new_spinner()
            .with_style(ProgressStyle::with_template("{spinner:.green} {msg}").expect("static")),
    );
    bar.set_message(msg);
    bar.enable_steady_tick(Duration::from_millis(120));
    bar
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        // One clean, actionable error — no Rust backtrace wall.
        eprintln!("\n✘ {err:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    // Default to warnings so the progress UI stays readable; RUST_LOG overrides.
    // wgpu_hal spams ERROR logs while probing backends that have no driver
    // (expected on most machines) — keep those out of the default view.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,wgpu_hal=off,wgpu_core=off"),
    )
    .init();
    let cli = Cli::parse();

    let gpu_opts = gpu::GpuOptions {
        backend: cli
            .gpu_backend
            .parse::<gpu::BackendChoice>()
            .map_err(anyhow::Error::msg)?,
        index: cli.gpu_index,
        allow_software: cli.allow_software,
    };
    if cli.doctor {
        return doctor(&cli, &gpu_opts).await;
    }

    let input = cli
        .input
        .clone()
        .context("Missing input: pass a directory of photos (or --help)")?;
    let kind = classify_input(&input, cli.mesh_only)?;

    let preset = cli.quality.preset();
    let out_dir = std::path::absolute(&cli.output)?;
    std::fs::create_dir_all(&out_dir)?;

    let mut run = RunReport::default();

    // Bring the GPU up *first* when training is needed: a machine that can't
    // train should fail in milliseconds, not after minutes of pose estimation.
    let needs_training = !matches!(kind, InputKind::SplatPly(_));
    if needs_training {
        let active = gpu::init_for_training(&gpu_opts).await?;
        eprintln!("◆ GPU: {}", active.describe());
        if active.software {
            eprintln!(
                "  ⚠ software rasterizer — expect training to take hours; \
                 any real GPU will be dramatically faster"
            );
        }
        run.gpu = Some(active.describe());
    } else {
        eprintln!("◆ GPU: not needed (--mesh-only runs on CPU)");
    }

    let multi = MultiProgress::new();
    let total_start = std::time::Instant::now();

    // ── Stage 1: camera poses ───────────────────────────────────────────
    let (dataset_dir, sparse_dir): (PathBuf, Option<PathBuf>) = match &kind {
        InputKind::Photos(photos) => {
            eprintln!("\n■ Stage 1/3 — camera poses (COLMAP)");
            let work_dir = out_dir.join("work").join("colmap");
            let sfm = run_sfm_stage(&cli, photos, &work_dir, &multi).await?;
            eprintln!(
                "  registered {}/{} photos, {} sparse points in {}",
                sfm.stats.registered_images,
                sfm.stats.input_images,
                sfm.stats.sparse_points,
                humantime::format_duration(Duration::from_secs(sfm.stats.elapsed.as_secs()))
            );
            run.sfm = Some(SfmReport {
                input_images: sfm.stats.input_images,
                registered_images: sfm.stats.registered_images,
                sparse_points: sfm.stats.sparse_points,
                seconds: sfm.stats.elapsed.as_secs_f64(),
            });
            (sfm.dataset_dir, Some(sfm.sparse_dir))
        }
        InputKind::Dataset(dir) => {
            eprintln!("\n■ Stage 1/3 — camera poses: dataset already posed, skipping");
            let sparse = dir.join("sparse").join("0");
            (dir.clone(), sparse.is_dir().then_some(sparse))
        }
        InputKind::SplatPly(_) => (PathBuf::new(), cli.sparse.clone()),
    };

    // ── Stage 2: splat training ─────────────────────────────────────────
    let splat_path = match &kind {
        InputKind::SplatPly(ply) => {
            eprintln!("\n■ Stage 2/3 — training: skipped (--mesh-only)");
            ply.clone()
        }
        _ => {
            eprintln!("\n■ Stage 2/3 — training gaussian splat (Brush · wgpu)");
            let outcome = train_splat(
                &dataset_dir,
                TrainSettings {
                    iters: cli.iters.unwrap_or(preset.iters),
                    max_splats: cli.max_splats.unwrap_or(preset.max_splats),
                    max_resolution: cli.max_resolution.unwrap_or(preset.max_resolution),
                    seed: cli.seed,
                    export_dir: out_dir.clone(),
                },
                &multi,
            )
            .await?;
            eprintln!(
                "  trained {} splats in {}{}",
                outcome.final_splats,
                humantime::format_duration(Duration::from_secs(outcome.elapsed.as_secs())),
                outcome
                    .last_psnr
                    .map(|p| format!(", PSNR {p:.2}"))
                    .unwrap_or_default()
            );
            run.training = Some(TrainReport {
                iterations: cli.iters.unwrap_or(preset.iters),
                final_splat_count: outcome.final_splats,
                last_psnr: outcome.last_psnr,
                seconds: outcome.elapsed.as_secs_f64(),
            });
            run.outputs.push(outcome.splat_path.clone());
            outcome.splat_path
        }
    };

    // ── Stage 3: mesh extraction ────────────────────────────────────────
    if cli.splat_only {
        eprintln!("\n■ Stage 3/3 — mesh: skipped (--splat-only)");
    } else {
        eprintln!("\n■ Stage 3/3 — extracting mesh (TSDF + surface nets)");
        let mesh_start = std::time::Instant::now();

        let camera_centers = match &sparse_dir {
            Some(dir) if dir.is_dir() => msplat_colmap::read_sparse_model(dir)
                .await
                .map(|m| m.camera_centers())
                .unwrap_or_else(|e| {
                    log::warn!("Could not read camera poses for normal orientation: {e}");
                    Vec::new()
                }),
            _ => Vec::new(),
        };

        let params = MeshParams {
            grid_res: cli.grid_res.unwrap_or(preset.grid_res),
            opacity_min: cli.opacity_min.unwrap_or(MeshParams::default().opacity_min),
            smooth_iters: cli
                .smooth_iters
                .unwrap_or(MeshParams::default().smooth_iters),
            ..MeshParams::default()
        };

        let bar = multi.add(
            ProgressBar::new(100).with_style(
                ProgressStyle::with_template("  {bar:38.magenta/dim} {percent}% {msg}")
                    .expect("static template")
                    .progress_chars("=> "),
            ),
        );

        let splat_for_task = splat_path.clone();
        let bar_for_task = bar.clone();
        let mesh = tokio::task::spawn_blocking(move || -> Result<msplat_mesh::TriMesh> {
            let cloud = msplat_mesh::load_splat_ply(&splat_for_task)?;
            msplat_mesh::extract_mesh(&cloud, &camera_centers, &params, |p| match p {
                ExtractProgress::Phase(name) => bar_for_task.set_message(name),
                ExtractProgress::Fusing(frac) => {
                    bar_for_task.set_position((frac * 100.0) as u64);
                }
            })
        })
        .await
        .context("Mesh extraction task panicked")??;
        bar.finish_with_message("done");

        type MeshWriter = fn(&msplat_mesh::TriMesh, &Path) -> Result<()>;
        let mut outputs = Vec::new();
        let writers: [(&str, MeshWriter); 3] = [
            ("mesh.ply", msplat_mesh::write_ply),
            ("mesh.obj", msplat_mesh::write_obj),
            ("mesh.glb", msplat_mesh::write_glb),
        ];
        for (name, writer) in writers {
            let path = out_dir.join(name);
            writer(&mesh, &path)?;
            outputs.push(path);
        }
        eprintln!(
            "  {} vertices, {} triangles in {}",
            mesh.num_vertices(),
            mesh.num_triangles(),
            humantime::format_duration(Duration::from_secs(mesh_start.elapsed().as_secs()))
        );
        run.mesh = Some(MeshReport {
            vertices: mesh.num_vertices(),
            triangles: mesh.num_triangles(),
            seconds: mesh_start.elapsed().as_secs_f64(),
        });
        run.outputs.extend(outputs);
    }

    let report_path = run.save(&out_dir)?;
    run.outputs.push(report_path);

    eprintln!(
        "\n✔ Done in {}. Outputs in {}:",
        humantime::format_duration(Duration::from_secs(total_start.elapsed().as_secs())),
        out_dir.display()
    );
    for output in &run.outputs {
        if let Some(name) = output.file_name().and_then(|n| n.to_str()) {
            eprintln!("    {name}");
        }
    }
    Ok(())
}

/// Stage 1: acquire COLMAP and run SfM with live progress.
async fn run_sfm_stage(
    cli: &Cli,
    photos: &Path,
    work_dir: &Path,
    multi: &MultiProgress,
) -> Result<msplat_colmap::SfmOutput> {
    let fetch_bar = spinner(multi, "Locating COLMAP...".into());
    let colmap = msplat_colmap::ensure_colmap(cli.colmap.as_deref(), |event| match event {
        FetchEvent::Found { path, version } => {
            fetch_bar.finish_with_message(format!(
                "COLMAP: {}{}",
                version.unwrap_or_else(|| "found".into()),
                format_args!(" ({path})")
            ));
        }
        FetchEvent::Downloading { received, total } => {
            let mb = received / (1024 * 1024);
            match total {
                Some(t) => fetch_bar.set_message(format!(
                    "Downloading COLMAP... {mb}/{} MB",
                    t / (1024 * 1024)
                )),
                None => fetch_bar.set_message(format!("Downloading COLMAP... {mb} MB")),
            }
        }
        FetchEvent::Extracting => fetch_bar.set_message("Extracting COLMAP..."),
    })
    .await?;

    let capture = cli
        .capture
        .parse::<CaptureMode>()
        .map_err(anyhow::Error::msg)?;
    let opts = SfmOptions {
        capture,
        // Turntable / video captures are taken with one physical camera;
        // sharing intrinsics across frames makes COLMAP much more robust.
        single_camera: cli.single_camera || capture == CaptureMode::Sequential,
        cpu_only: cli.cpu_sfm,
        mask_dir: cli.masks.clone(),
        mapper: cli
            .sfm_mapper
            .parse::<msplat_colmap::MapperKind>()
            .map_err(anyhow::Error::msg)?,
        ..SfmOptions::default()
    };

    let stage_bar = multi.add(
        ProgressBar::new(1).with_style(
            ProgressStyle::with_template("  {bar:38.cyan/dim} {pos}/{len} {msg}")
                .expect("static template")
                .progress_chars("=> "),
        ),
    );
    stage_bar.enable_steady_tick(Duration::from_millis(250));

    let output = msplat_colmap::run_sfm(&colmap, photos, work_dir, &opts, |event| match event {
        SfmEvent::StageStarted { name, index, total } => {
            stage_bar.set_position(0);
            stage_bar.set_length(1);
            stage_bar.set_message(format!("[{index}/{total}] {name}"));
        }
        SfmEvent::Progress { done, total } => {
            stage_bar.set_length(total);
            stage_bar.set_position(done);
        }
        SfmEvent::Status(status) => stage_bar.set_message(status),
    })
    .await?;
    stage_bar.finish_with_message("poses ready");
    Ok(output)
}

/// `--doctor`: report GPU adapters (in selection-ladder order) and COLMAP
/// availability.
async fn doctor(cli: &Cli, gpu_opts: &gpu::GpuOptions) -> Result<()> {
    println!("meshsplat doctor\n────────────────");
    let candidates = gpu::enumerate(gpu_opts.backend).await;
    if candidates.is_empty() {
        println!(
            "GPU: no adapters found — training will not work on this machine.\n\
             Training needs Vulkan, DirectX 12, Metal or OpenGL via your normal\n\
             graphics driver (any vendor, no CUDA). Updating the driver usually\n\
             fixes this. Mesh extraction (--mesh-only) works without a GPU."
        );
    } else {
        println!("GPU adapters, in selection order (pin one with --gpu-index N):");
        let mut chosen = false;
        for (i, c) in candidates.iter().enumerate() {
            let mark = if !c.software && !chosen {
                chosen = true;
                "→ would use"
            } else if c.software {
                "(software — needs --allow-software)"
            } else {
                ""
            };
            println!("  [{i}] {} {mark}", c.describe());
        }
        if !chosen {
            println!(
                "  Only software rasterizers found; training requires \
                 --allow-software (very slow)."
            );
        }
    }
    match msplat_colmap::locate_colmap(cli.colmap.as_deref()) {
        Some(bin) => println!(
            "COLMAP: {} ({})",
            bin.path.display(),
            bin.version().unwrap_or_else(|| "version unknown".into())
        ),
        None => println!(
            "COLMAP: not found — it will be downloaded automatically on Windows, \
             or install it via your package manager (apt/brew install colmap)."
        ),
    }
    Ok(())
}
