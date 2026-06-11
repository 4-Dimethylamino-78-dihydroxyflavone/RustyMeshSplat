//! Fail-proof GPU bring-up for training.
//!
//! Brush's stock init hard-codes Vulkan on Windows/Linux and panics if the
//! adapter request fails. This module replaces that with a selection ladder:
//! every backend is probed (Windows: Vulkan → DX12 → OpenGL; macOS: Metal;
//! Linux: Vulkan → OpenGL), real GPUs are preferred over software
//! rasterizers, the user can pin a backend or adapter, and every failure
//! falls through to the next candidate instead of crashing. Only when the
//! whole ladder is exhausted do we return one comprehensive, actionable
//! error — before any pipeline time has been spent.

use anyhow::{Result, bail};
use wgpu::{Adapter, Backend, DeviceType};

/// User-selectable backend constraint (`--gpu-backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Vulkan,
    Dx12,
    Metal,
    Gl,
}

impl std::str::FromStr for BackendChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "vulkan" | "vk" => Ok(Self::Vulkan),
            "dx12" | "directx" | "d3d12" => Ok(Self::Dx12),
            "metal" => Ok(Self::Metal),
            "gl" | "opengl" | "gles" => Ok(Self::Gl),
            other => Err(format!(
                "unknown GPU backend '{other}' (expected auto|vulkan|dx12|metal|gl)"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuOptions {
    pub backend: BackendChoice,
    /// Pick the Nth candidate from the ranked list (as shown by --doctor).
    pub index: Option<usize>,
    /// Allow software rasterizers (WARP, llvmpipe) — extremely slow, but
    /// turns "no GPU" into "still produces a result".
    pub allow_software: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            backend: BackendChoice::Auto,
            index: None,
            allow_software: false,
        }
    }
}

/// One usable adapter candidate, in ranking order.
pub struct Candidate {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub software: bool,
    adapter: Adapter,
}

impl Candidate {
    pub fn backend_name(&self) -> &'static str {
        backend_name(self.backend)
    }

    pub fn type_name(&self) -> &'static str {
        match self.device_type {
            DeviceType::DiscreteGpu => "discrete GPU",
            DeviceType::IntegratedGpu => "integrated GPU",
            DeviceType::VirtualGpu => "virtual GPU",
            DeviceType::Cpu => "software rasterizer",
            DeviceType::Other => "unknown device",
        }
    }

    pub fn describe(&self) -> String {
        format!("{} — {} [{}]", self.name, self.backend_name(), self.type_name())
    }
}

pub fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Vulkan",
        Backend::Dx12 => "DirectX 12",
        Backend::Metal => "Metal",
        Backend::Gl => "OpenGL",
        Backend::BrowserWebGpu => "WebGPU",
        Backend::Noop => "noop",
    }
}

/// Backends to try, most-reliable-first for the current OS.
fn backend_order(choice: BackendChoice) -> Vec<Backend> {
    match choice {
        BackendChoice::Vulkan => vec![Backend::Vulkan],
        BackendChoice::Dx12 => vec![Backend::Dx12],
        BackendChoice::Metal => vec![Backend::Metal],
        BackendChoice::Gl => vec![Backend::Gl],
        BackendChoice::Auto => {
            if cfg!(target_os = "windows") {
                // Vulkan first (fastest compute path when drivers are good),
                // DX12 second (present and healthy on virtually every
                // Windows 10/11 machine), GL as the last resort.
                vec![Backend::Vulkan, Backend::Dx12, Backend::Gl]
            } else if cfg!(target_os = "macos") {
                vec![Backend::Metal]
            } else {
                vec![Backend::Vulkan, Backend::Gl]
            }
        }
    }
}

fn is_software(name: &str, device_type: DeviceType) -> bool {
    let lower = name.to_ascii_lowercase();
    device_type == DeviceType::Cpu
        || lower.contains("llvmpipe")
        || lower.contains("swiftshader")
        || lower.contains("basic render") // Microsoft Basic Render Driver (WARP)
        || lower.contains("warp")
}

fn type_rank(t: DeviceType) -> u8 {
    match t {
        DeviceType::DiscreteGpu => 0,
        DeviceType::IntegratedGpu => 1,
        DeviceType::VirtualGpu => 2,
        DeviceType::Other => 3,
        DeviceType::Cpu => 4,
    }
}

/// Enumerate every candidate adapter in ladder order: backend preference
/// first, then real GPUs before software rasterizers within each backend.
pub async fn enumerate(choice: BackendChoice) -> Vec<Candidate> {
    let mut all = Vec::new();
    for backend in backend_order(choice) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend.into(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let mut candidates: Vec<Candidate> = instance
            .enumerate_adapters(backend.into())
            .await
            .into_iter()
            .map(|adapter| {
                let info = adapter.get_info();
                Candidate {
                    software: is_software(&info.name, info.device_type),
                    name: info.name,
                    backend: info.backend,
                    device_type: info.device_type,
                    adapter,
                }
            })
            .collect();
        candidates.sort_by_key(|c| (c.software, type_rank(c.device_type)));
        all.extend(candidates);
    }
    all
}

/// The adapter that was brought up successfully.
pub struct ActiveGpu {
    pub name: String,
    pub backend: &'static str,
    pub device_type: &'static str,
    pub software: bool,
}

impl ActiveGpu {
    pub fn describe(&self) -> String {
        format!("{} via {} [{}]", self.name, self.backend, self.device_type)
    }
}

/// Walk the ladder until one adapter yields a working device, then register
/// it with Brush/Burn. Never panics: failures accumulate into the final
/// error report instead.
pub async fn init_for_training(opts: &GpuOptions) -> Result<ActiveGpu> {
    let candidates = enumerate(opts.backend).await;
    let mut failures: Vec<String> = Vec::new();
    let mut skipped_software = 0usize;

    let selected: Vec<&Candidate> = match opts.index {
        Some(i) => match candidates.get(i) {
            Some(c) => vec![c],
            None => bail!(
                "--gpu-index {i} is out of range: only {} adapter(s) found. \
                 Run `meshsplat --doctor` to list them.",
                candidates.len()
            ),
        },
        None => candidates
            .iter()
            .filter(|c| {
                if c.software && !opts.allow_software {
                    skipped_software += 1;
                    false
                } else {
                    true
                }
            })
            .collect(),
    };

    for candidate in selected {
        if candidate.software {
            log::warn!(
                "Using software rasterizer '{}' — training will be extremely slow.",
                candidate.name
            );
        }
        match bring_up(candidate).await {
            Ok(active) => return Ok(active),
            Err(err) => {
                log::warn!("{} failed to initialize: {err:#}", candidate.describe());
                failures.push(format!("  ✗ {} — {err:#}", candidate.describe()));
            }
        }
    }

    // Exhausted: build one actionable report.
    let mut msg = String::from("No GPU could be initialized for training.\n");
    if candidates.is_empty() {
        msg.push_str(
            "No graphics adapters were found at all. Training runs on Vulkan, \
             DirectX 12, Metal or OpenGL through your normal graphics driver \
             (any vendor — AMD, Intel, NVIDIA, Apple — no CUDA needed).\n",
        );
    } else if !failures.is_empty() {
        msg.push_str("Adapters tried:\n");
        msg.push_str(&failures.join("\n"));
        msg.push('\n');
    }
    if skipped_software > 0 {
        msg.push_str(&format!(
            "{skipped_software} software rasterizer(s) were skipped; pass \
             --allow-software to use one (very slow, but works).\n"
        ));
    }
    msg.push_str(
        "Things to try:\n\
         \u{2022} update your GPU driver (the #1 fix)\n\
         \u{2022} on Windows: --gpu-backend dx12 (or vulkan) to pin a backend\n\
         \u{2022} meshsplat --doctor to see every adapter and backend\n\
         \u{2022} --mesh-only still works without any GPU on a splat trained elsewhere",
    );
    bail!(msg)
}

/// Create the device + queue the way cubecl does (all adapter features
/// except MAPPABLE_PRIMARY_BUFFERS, native limits, memory-usage hints) and
/// hand the live setup to Brush. `burn_init_device` is Brush's documented
/// entry point for hosts that own the wgpu setup.
async fn bring_up(candidate: &Candidate) -> Result<ActiveGpu> {
    let (device, queue) = candidate
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("meshsplat-training"),
            required_features: candidate
                .adapter
                .features()
                .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: candidate.adapter.limits(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            // SAFETY: mirrors cubecl-wgpu's own device request; kernels are
            // compiled by the same stack that sets this flag upstream.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        })
        .await
        .map_err(|e| anyhow::anyhow!("device request failed: {e}"))?;

    let adapter = candidate.adapter.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        brush_process::burn_init_device(adapter, device, queue)
    }));
    match result {
        Ok(_burn_device) => Ok(ActiveGpu {
            name: candidate.name.clone(),
            backend: candidate.backend_name(),
            device_type: candidate.type_name(),
            software: candidate.software,
        }),
        Err(panic) => {
            let detail = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            bail!("compute backend setup panicked: {detail}")
        }
    }
}
