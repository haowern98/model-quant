use model_surgery::commands::hardware::{
    bytes_to_mebibytes, milliwatts_to_watts, select_cpu_temperature,
};

#[test]
fn hardware_snapshot_does_not_spawn_nvidia_smi() {
    let hardware_source = include_str!("../src/commands/hardware.rs");

    assert!(!hardware_source.contains("std::process::Command"));
    assert!(!hardware_source.contains("nvidia-smi"));
}

#[test]
fn converts_nvml_memory_bytes_to_mebibytes() {
    assert_eq!(bytes_to_mebibytes(8 * 1024 * 1024), 8.0);
}

#[test]
fn converts_nvml_power_milliwatts_to_watts() {
    assert_eq!(milliwatts_to_watts(286_500), 286.5);
}

#[test]
fn selects_cpu_temperature_from_component_sensors() {
    let readings = [
        ("GPU", Some(62.0)),
        ("CPU Package", Some(54.0)),
        ("CPU Core 0", Some(50.0)),
    ];

    assert_eq!(select_cpu_temperature(&readings), Some(54.0));
}
