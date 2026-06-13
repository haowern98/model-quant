#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::c_char;

pub type llama_model = std::ffi::c_void;
pub type llama_context = std::ffi::c_void;

extern "C" {
    pub fn llama_backend_init() -> ();
    pub fn llama_backend_free() -> ();

    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_new_context_with_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub fn llama_free(ctx: *mut llama_context);

    pub fn llama_model_desc(model: *mut llama_model, buf: *mut c_char, buf_size: usize) -> i32;
    pub fn llama_model_n_ctx_train(model: *mut llama_model) -> i32;
    pub fn llama_model_size(model: *mut llama_model) -> u64;

    // Row-level quantize functions from ggml-quants.h
    pub fn quantize_row_q8_0(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q4_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q6_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q5_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q3_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q2_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
    pub fn quantize_row_q8_K(src: *const f32, dst: *mut std::ffi::c_void, k: i32) -> usize;
}

#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: Option<extern "C" fn(f32, *mut std::ffi::c_void)>,
    pub progress_callback_user_data: *mut std::ffi::c_void,
    pub kv_overrides: *mut std::ffi::c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
}

impl Default for llama_model_params {
    fn default() -> Self {
        Self {
            n_gpu_layers: 9999,
            split_mode: 0,
            main_gpu: 0,
            tensor_split: std::ptr::null(),
            progress_callback: None,
            progress_callback_user_data: std::ptr::null_mut(),
            kv_overrides: std::ptr::null_mut(),
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
        }
    }
}

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_threshold: f32,
    pub cb_eval: Option<extern "C" fn(*mut std::ffi::c_void)>,
    pub cb_eval_user_data: *mut std::ffi::c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub offload_kqv: bool,
    pub flash_attn: bool,
    pub no_perf: bool,
    pub abort_callback: Option<extern "C" fn(*mut std::ffi::c_void) -> bool>,
    pub abort_callback_data: *mut std::ffi::c_void,
}

impl Default for llama_context_params {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4) as u32,
            n_threads_batch: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4) as u32,
            rope_scaling_type: 0,
            pooling_type: 0,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_threshold: -1.0,
            cb_eval: None,
            cb_eval_user_data: std::ptr::null_mut(),
            type_k: 1,
            type_v: 1,
            offload_kqv: true,
            flash_attn: false,
            no_perf: false,
            abort_callback: None,
            abort_callback_data: std::ptr::null_mut(),
        }
    }
}

// LlamaBackend is gated behind the `llama` feature until llama.cpp is linked.
// When the submodule is added and build.rs compiles llama.cpp, enable this.

#[cfg(feature = "llama")]
pub struct LlamaBackend;

#[cfg(feature = "llama")]
impl LlamaBackend {
    pub fn init() -> Self {
        unsafe {
            llama_backend_init();
        }
        LlamaBackend
    }
}

#[cfg(feature = "llama")]
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe {
            llama_backend_free();
        }
    }
}
