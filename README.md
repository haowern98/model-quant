# Model Surgery Workbench

A visual mixed-precision quantization editor for GGUF models. Load a GGUF file, inspect layers, assign per-tensor quantization types, test with live profiling, and export optimized GGUF files.

## Features

- **Layer Browser** — Grouped tensor view (embeddings, attention layers, output)
- **Per-Tensor Quantization** — Assign different quantization types to individual tensors
- **Bulk Assignment** — Apply quant types by pattern (all attention, all FFN, etc.)
- **Live Benchmarking** — Test recipes with real inference profiling
- **VRAM Profiling** — CUDA VRAM measurement and visualization
- **Recipe Management** — Save, load, and share quantization recipes

## Tech Stack

- **UI**: React + TypeScript + Tailwind CSS + Motion
- **Desktop Shell**: Tauri v2 (OS WebView, no Electron)
- **Backend**: Rust with llama.cpp linked as a static library
- **Profiling**: Custom C++ CUDA wrappers (with stub fallback)
- **Testing**: Playwright (UI/E2E) + Rust tests (backend)

## Prerequisites

- Node.js 18+ and npm
- Rust toolchain (rustup)
- CUDA Toolkit 12.x (for GPU profiling; builds without it using a stub)
- Git LFS (for test fixtures)

## Development

```bash
npm install          # Install frontend dependencies
npm run dev          # Vite dev server (UI only, uses mock data)
npx tauri dev        # Full Tauri desktop app
```

## Testing

```bash
npx playwright test  # UI and E2E tests
cargo test           # Rust backend tests (from src-tauri/)
```

## Build

```bash
npx tauri build
```

Produces an NSIS installer in `src-tauri/target/release/bundle/nsis/`.

## Architecture

Single-process desktop app. React UI runs in the OS WebView (Edge WebView2 on Windows), communicates via Tauri IPC with a Rust backend. llama.cpp is compiled as a `.lib` and linked directly into the binary. A C++ profiling layer wraps CUDA APIs for VRAM measurement. All native code is statically linked.
