use std::sync::{Arc, Mutex};

use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, Nvml};
use sysinfo::{Components, System};
use tauri::State;

pub struct HardwareMonitor(Arc<Mutex<System>>);

impl HardwareMonitor {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(System::new_all())))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareSnapshot {
    pub cpu_name: String,
    pub cpu_usage_percent: f64,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub gpu_name: Option<String>,
    pub gpu_usage_percent: Option<f64>,
    pub vram_used_mb: Option<f64>,
    pub vram_total_mb: Option<f64>,
    pub gpu_temperature_c: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub cpu_temperature_c: Option<f64>,
    pub cpu_power_w: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NvidiaTelemetry {
    pub name: String,
    pub usage_percent: Option<f64>,
    pub vram_used_mb: Option<f64>,
    pub vram_total_mb: Option<f64>,
    pub temperature_c: Option<f64>,
    pub power_w: Option<f64>,
}

pub fn bytes_to_mebibytes(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

pub fn milliwatts_to_watts(milliwatts: u32) -> f64 {
    milliwatts as f64 / 1000.0
}

pub fn select_cpu_temperature(readings: &[(&str, Option<f32>)]) -> Option<f64> {
    readings
        .iter()
        .filter_map(|(label, temperature)| {
            let label = label.to_ascii_lowercase();
            let priority = if label.contains("cpu package") || label.contains("package id") {
                2
            } else if label.contains("cpu") || label.contains("processor") {
                1
            } else {
                0
            };
            let temperature = temperature.filter(|value| value.is_finite())?;
            (priority > 0).then_some((priority, temperature))
        })
        .max_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        })
        .map(|(_, temperature)| temperature as f64)
}

fn read_nvidia_telemetry() -> Option<NvidiaTelemetry> {
    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let memory = device.memory_info().ok();
    let utilization = device.utilization_rates().ok();

    Some(NvidiaTelemetry {
        name: device.name().ok()?,
        usage_percent: utilization.map(|item| item.gpu as f64),
        vram_used_mb: memory.as_ref().map(|item| bytes_to_mebibytes(item.used)),
        vram_total_mb: memory.as_ref().map(|item| bytes_to_mebibytes(item.total)),
        temperature_c: device
            .temperature(TemperatureSensor::Gpu)
            .ok()
            .map(|item| item as f64),
        power_w: device.power_usage().ok().map(milliwatts_to_watts),
    })
}

fn collect_hardware_snapshot(system: Arc<Mutex<System>>) -> Result<HardwareSnapshot, String> {
    let (cpu_name, cpu_usage_percent, ram_used_bytes, ram_total_bytes) = {
        let mut system = system.lock().map_err(|error| error.to_string())?;
        system.refresh_cpu_usage();
        system.refresh_memory();

        (
            system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim())
                .filter(|name| !name.is_empty())
                .unwrap_or("CPU")
                .to_string(),
            system.global_cpu_usage() as f64,
            system.used_memory(),
            system.total_memory(),
        )
    };

    let components = Components::new_with_refreshed_list();
    let component_readings = components
        .list()
        .iter()
        .map(|component| (component.label(), component.temperature()))
        .collect::<Vec<_>>();
    let cpu_temperature_c = select_cpu_temperature(&component_readings);
    let gpu = read_nvidia_telemetry();

    Ok(HardwareSnapshot {
        cpu_name,
        cpu_usage_percent,
        ram_used_bytes,
        ram_total_bytes,
        gpu_name: gpu.as_ref().map(|item| item.name.clone()),
        gpu_usage_percent: gpu.as_ref().and_then(|item| item.usage_percent),
        vram_used_mb: gpu.as_ref().and_then(|item| item.vram_used_mb),
        vram_total_mb: gpu.as_ref().and_then(|item| item.vram_total_mb),
        gpu_temperature_c: gpu.as_ref().and_then(|item| item.temperature_c),
        gpu_power_w: gpu.as_ref().and_then(|item| item.power_w),
        cpu_temperature_c,
        cpu_power_w: None,
    })
}

#[tauri::command]
pub async fn get_hardware_snapshot(
    monitor: State<'_, HardwareMonitor>,
) -> Result<HardwareSnapshot, String> {
    let system = Arc::clone(&monitor.0);
    tauri::async_runtime::spawn_blocking(move || collect_hardware_snapshot(system))
        .await
        .map_err(|error| error.to_string())?
}

#[cfg(all(test, windows))]
mod tests {
    use std::ffi::c_void;
    use std::ptr;

    use super::HardwareMonitor;

    const RPC_E_CHANGED_MODE: u32 = 0x80010106;

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn OleInitialize(reserved: *mut c_void) -> i32;
        fn OleUninitialize();
    }

    #[test]
    fn monitor_construction_does_not_change_the_thread_com_mode() {
        let _monitor = HardwareMonitor::new();

        let result = unsafe { OleInitialize(ptr::null_mut()) };
        assert_ne!(result as u32, RPC_E_CHANGED_MODE);

        if result >= 0 {
            unsafe { OleUninitialize() };
        }
    }
}
