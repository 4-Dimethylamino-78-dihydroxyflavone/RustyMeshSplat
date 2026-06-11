//! End-to-end test of the orchestration path (spawn → JSON event parsing →
//! dataset assembly → stats) against a fake Python script that emits the full
//! protocol and a tiny COLMAP text model. Needs only a Python interpreter;
//! self-skips when none is installed.

use std::path::{Path, PathBuf};

use msplat_colmap::SfmEvent;
use msplat_ffpose::{FfposeOptions, PyRuntime, run_ffpose};

fn find_python() -> Option<PathBuf> {
    ["python3", "python"]
        .iter()
        .find_map(|name| which::which(name).ok())
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn make_photo_dir(root: &Path) -> PathBuf {
    let photos = root.join("photos");
    std::fs::create_dir_all(&photos).expect("create photo dir");
    // Content is never decoded on this path; only names/extensions matter.
    for name in ["img0.png", "img1.png", "img2.png"] {
        std::fs::write(photos.join(name), b"not a real png").expect("write photo");
    }
    photos
}

#[tokio::test]
async fn orchestration_end_to_end() {
    let Some(python) = find_python() else {
        eprintln!("skipping: no python interpreter on PATH");
        return;
    };
    // SAFETY: this is the only test in this binary that touches the variable.
    unsafe {
        std::env::set_var(
            "MESHSPLAT_FFPOSE_SCRIPT",
            fixtures_dir().join("fake_mapanything.py"),
        );
    }

    let root = std::env::temp_dir().join(format!("msplat-ffpose-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let photos = make_photo_dir(&root);
    let work = root.join("work");

    let runtime = PyRuntime::Python(python);
    let mut stages = Vec::new();
    let mut saw_progress = false;
    let output = run_ffpose(
        &runtime,
        &photos,
        &work,
        &FfposeOptions::default(),
        |event| match event {
            SfmEvent::StageStarted { name, .. } => stages.push(name),
            SfmEvent::Progress { .. } => saw_progress = true,
            SfmEvent::Status(_) => {}
        },
    )
    .await
    .expect("fake pipeline should succeed");

    assert_eq!(stages.len(), 4, "expected all four protocol stages: {stages:?}");
    assert_eq!(stages[0], "Loading PyTorch");
    assert!(saw_progress, "progress events should be forwarded");

    assert_eq!(output.stats.input_images, 3);
    assert_eq!(output.stats.registered_images, 3);
    assert_eq!(output.stats.sparse_points, 5);

    // Brush-compatible layout: images/ + sparse/0/, no stray points.ply.
    let dataset = &output.dataset_dir;
    assert!(dataset.join("images/img0.png").is_file());
    assert!(dataset.join("images/img2.png").is_file());
    assert!(output.sparse_dir.join("cameras.txt").is_file());
    assert!(output.sparse_dir.join("images.txt").is_file());
    assert!(output.sparse_dir.join("points3D.txt").is_file());
    assert!(!output.sparse_dir.join("points.ply").exists());

    // A structured script error must surface its message and hint.
    unsafe {
        std::env::set_var("FAKE_FFPOSE_FAIL", "1");
    }
    let err = match run_ffpose(
        &runtime,
        &photos,
        &work,
        &FfposeOptions::default(),
        |_| {},
    )
    .await
    {
        Ok(_) => panic!("fake failure should propagate"),
        Err(err) => err,
    };
    let text = format!("{err:#}");
    assert!(text.contains("synthetic failure"), "got: {text}");
    assert!(text.contains("this is a test"), "got: {text}");

    let _ = std::fs::remove_dir_all(&root);
}
