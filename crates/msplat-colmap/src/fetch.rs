use std::path::Path;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;

use crate::locate::{ColmapBinary, install_dir, locate};

/// Official prebuilt for Windows, no-CUDA variant. SIFT runs on CPU or GL —
/// works on any GPU vendor.
const WINDOWS_NOCUDA_ZIP_URL: &str =
    "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-nocuda.zip";

/// Official prebuilt for Windows, CUDA variant. Much faster SIFT extraction +
/// matching on an NVIDIA GPU — the dominant cost on large image sets. Needs a
/// recent NVIDIA driver (the CUDA runtime DLLs are bundled).
const WINDOWS_CUDA_ZIP_URL: &str =
    "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip";

/// Progress events while acquiring COLMAP.
pub enum FetchEvent {
    Found { path: String, version: Option<String> },
    Downloading { received: u64, total: Option<u64> },
    Extracting,
}

/// Find COLMAP, or (on Windows) download the official prebuilt into the
/// cache dir. On Linux/macOS we refuse to guess and print install steps.
/// `prefer_cuda` selects the CUDA prebuilt for the auto-download (NVIDIA GPUs);
/// `MESHSPLAT_COLMAP_URL` overrides the choice entirely.
pub async fn ensure_colmap(
    explicit: Option<&Path>,
    prefer_cuda: bool,
    mut on_event: impl FnMut(FetchEvent),
) -> Result<ColmapBinary> {
    if let Some(bin) = locate(explicit) {
        if bin.works() {
            on_event(FetchEvent::Found {
                path: bin.path.display().to_string(),
                version: bin.version(),
            });
            return Ok(bin);
        }
        bail!(
            "COLMAP at {} exists but failed to run (`colmap help`). \
             Check the install or pass a different --colmap path.",
            bin.path.display()
        );
    }

    if !cfg!(windows) {
        bail!(
            "COLMAP was not found on this system and automatic download is only \
             available for Windows.\n\
             Install it with your package manager and re-run:\n\
             \u{2022} Debian/Ubuntu:  sudo apt install colmap\n\
             \u{2022} macOS:          brew install colmap\n\
             \u{2022} Other:          https://colmap.github.io/install.html\n\
             Then re-run, or point meshsplat at the binary with --colmap <PATH>."
        );
    }

    let default_url = if prefer_cuda {
        WINDOWS_CUDA_ZIP_URL
    } else {
        WINDOWS_NOCUDA_ZIP_URL
    };
    let url = std::env::var("MESHSPLAT_COLMAP_URL").unwrap_or_else(|_| default_url.to_owned());
    let dest = install_dir().context("No usable cache directory on this system")?;
    tokio::fs::create_dir_all(&dest).await?;

    log::info!("Downloading COLMAP from {url}");
    let response = reqwest::get(&url)
        .await
        .with_context(|| format!("Downloading COLMAP from {url}"))?
        .error_for_status()?;
    let total = response.content_length();

    let mut bytes = Vec::with_capacity(total.unwrap_or(0) as usize);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        bytes.extend_from_slice(&chunk);
        on_event(FetchEvent::Downloading {
            received: bytes.len() as u64,
            total,
        });
    }

    on_event(FetchEvent::Extracting);
    let dest_clone = dest.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
        zip.extract(&dest_clone)?;
        Ok(())
    })
    .await
    .context("Extraction task panicked")??;

    let bin = locate(None).with_context(|| {
        format!(
            "COLMAP archive extracted to {} but no COLMAP binary was found inside",
            dest.display()
        )
    })?;
    on_event(FetchEvent::Found {
        path: bin.path.display().to_string(),
        version: bin.version(),
    });
    Ok(bin)
}
