# Native C++ Runtime

This directory contains native C++ dependencies and patches used by the Tauri app.

## llama.cpp

`llama.cpp` is pinned as a Git submodule at:

`c0c7e147e7efa6c5858754b47259ba4880f8a906`

Apply project patches from `native/cpp/patches` before building a runtime that uses
`llama_model_init_from_user`. The current patch makes metadata-only user-model
loading skip missing optional tensors instead of allocating zero-filled tensors.
