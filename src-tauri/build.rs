fn main() {
    println!("cargo:rerun-if-changed=icons/icon.ico");
    println!("cargo:rerun-if-changed=icons/32x32.png");
    println!("cargo:rerun-if-changed=icons/128x128.png");

    // Compile C++ CUDA profiler (only if CUDA toolkit is present)
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5".to_string()
    });
    let cuda_include = std::path::Path::new(&cuda_path).join("include");

    if cuda_include.exists() {
        cc::Build::new()
            .cpp(true)
            .file("cpp-profiling/profiler.cpp")
            .include(&cuda_include)
            .flag_if_supported("-std=c++17")
            .compile("profiler");

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        cc::Build::new()
            .cpp(true)
            .file("cpp-profiling/profiler_stub.cpp")
            .flag_if_supported("-std=c++17")
            .compile("profiler");
    }

    link_native_runtime();

    tauri_build::build()
}

fn link_native_runtime() {
    use std::path::{Path, PathBuf};

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_dir = manifest_dir
        .parent()
        .expect("src-tauri should live under the repository root");
    let native_build_dir = repo_dir.join("native").join("cpp").join("build");
    let native_profile = "Release";

    let mut runtime_dir = native_build_dir
        .join("model_surgery_runtime")
        .join(native_profile);
    let mut llama_bin_dir = native_build_dir.join("bin").join(native_profile);
    let mut mtmd_dir = native_build_dir
        .join("llama.cpp")
        .join("tools")
        .join("mtmd")
        .join(native_profile);

    if !runtime_dir.join("model_surgery_runtime.lib").exists() {
        runtime_dir = native_build_dir.join("model_surgery_runtime").join("Debug");
        llama_bin_dir = native_build_dir.join("bin").join("Debug");
        mtmd_dir = native_build_dir
            .join("llama.cpp")
            .join("tools")
            .join("mtmd")
            .join("Debug");
    }

    if !runtime_dir.join("model_surgery_runtime.lib").exists() {
        panic!("native runtime import library not found; build native/cpp first with CMake");
    }

    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=dylib=model_surgery_runtime");
    println!(
        "cargo:rerun-if-changed={}",
        repo_dir
            .join("native/cpp/model_surgery_runtime/include/model_surgery_runtime.h")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_dir
            .join("native/cpp/model_surgery_runtime/src/model_surgery_runtime.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        runtime_dir.join("model_surgery_runtime.dll").display()
    );

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR should be inside target/<profile>/build");
    let target_test_deps_dir = target_profile_dir.join("deps");

    copy_dll(
        &runtime_dir.join("model_surgery_runtime.dll"),
        target_profile_dir,
    );
    copy_dll(
        &runtime_dir.join("model_surgery_runtime.dll"),
        &target_test_deps_dir,
    );
    copy_dll(&llama_bin_dir.join("llama.dll"), target_profile_dir);
    copy_dll(&llama_bin_dir.join("llama.dll"), &target_test_deps_dir);
    copy_dll(&llama_bin_dir.join("llama-common.dll"), target_profile_dir);
    copy_dll(
        &llama_bin_dir.join("llama-common.dll"),
        &target_test_deps_dir,
    );
    copy_dll(&mtmd_dir.join("mtmd.dll"), target_profile_dir);
    copy_dll(&mtmd_dir.join("mtmd.dll"), &target_test_deps_dir);
    copy_dll(&llama_bin_dir.join("ggml.dll"), target_profile_dir);
    copy_dll(&llama_bin_dir.join("ggml.dll"), &target_test_deps_dir);
    copy_all_matching_dlls(&llama_bin_dir, "ggml", target_profile_dir);
    copy_all_matching_dlls(&llama_bin_dir, "ggml", &target_test_deps_dir);

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_bin = PathBuf::from(cuda_path).join("bin");
        for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"] {
            let source = cuda_bin.join(dll);
            if source.exists() {
                copy_dll(&source, target_profile_dir);
                copy_dll(&source, &target_test_deps_dir);
            }
        }
    }

    fn copy_dll(source: &Path, target_dir: &Path) {
        if !source.exists() {
            panic!("required native DLL not found: {}", source.display());
        }
        std::fs::create_dir_all(target_dir).unwrap_or_else(|err| {
            panic!(
                "failed to create native DLL target directory {}: {}",
                target_dir.display(),
                err
            )
        });
        let dest = target_dir.join(source.file_name().expect("DLL should have a file name"));
        std::fs::copy(source, &dest).unwrap_or_else(|err| {
            panic!(
                "failed to copy native DLL {} to {}: {}",
                source.display(),
                dest.display(),
                err
            )
        });
    }

    fn copy_all_matching_dlls(source_dir: &Path, prefix: &str, target_dir: &Path) {
        let entries = std::fs::read_dir(source_dir).unwrap_or_else(|err| {
            panic!(
                "failed to read native DLL directory {}: {}",
                source_dir.display(),
                err
            )
        });

        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.starts_with(prefix) && file_name.ends_with(".dll") {
                copy_dll(&path, target_dir);
            }
        }
    }
}
