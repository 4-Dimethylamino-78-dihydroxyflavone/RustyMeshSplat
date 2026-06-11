use wgpu::{Adapter, Backends, DeviceType};

/// What we know about the machine's GPU situation, via wgpu — the same
/// universal layer (Vulkan / DX12 / Metal / GL) the trainer runs on.
pub struct GpuReport {
    pub adapters: Vec<AdapterSummary>,
}

pub struct AdapterSummary {
    pub name: String,
    pub backend: &'static str,
    pub device_type: &'static str,
    pub discrete: bool,
}

fn summarize(adapter: &Adapter) -> AdapterSummary {
    let info = adapter.get_info();
    AdapterSummary {
        name: info.name.clone(),
        backend: match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Noop => "noop",
        },
        device_type: match info.device_type {
            DeviceType::DiscreteGpu => "discrete GPU",
            DeviceType::IntegratedGpu => "integrated GPU",
            DeviceType::VirtualGpu => "virtual GPU",
            DeviceType::Cpu => "software rasterizer",
            DeviceType::Other => "unknown device",
        },
        discrete: info.device_type == DeviceType::DiscreteGpu,
    }
}

pub async fn probe() -> GpuReport {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapters: Vec<AdapterSummary> = instance
        .enumerate_adapters(Backends::all())
        .await
        .iter()
        .map(summarize)
        .collect();
    // Surface the most useful adapter first for display.
    adapters.sort_by_key(|a| (!a.discrete, a.device_type == "software rasterizer"));
    GpuReport { adapters }
}

impl GpuReport {
    pub fn best(&self) -> Option<&AdapterSummary> {
        self.adapters.first()
    }

    pub fn has_usable_gpu(&self) -> bool {
        self.adapters
            .iter()
            .any(|a| a.device_type != "software rasterizer")
    }
}
