fn main() {
    // Compile C++ CUDA profiler (only if CUDA toolkit is present)
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5".to_string());
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

    // Compile llama.cpp as static library (requires git submodule)
    let llama_dir = std::path::Path::new("llama-cpp");
    if llama_dir.exists() {
        cc::Build::new()
            .cpp(true)
            .files(&[
                llama_dir.join("ggml.c"),
                llama_dir.join("ggml-alloc.c"),
                llama_dir.join("ggml-backend.c"),
                llama_dir.join("ggml-quants.c"),
                llama_dir.join("llama.cpp"),
            ])
            .include(llama_dir)
            .flag_if_supported("-std=c++11")
            .flag_if_supported("-pthread")
            .compile("llama");

        println!("cargo:rerun-if-changed=llama-cpp/llama.h");
        println!("cargo:rerun-if-changed=llama-cpp/llama.cpp");
    }

    tauri_build::build()
}
