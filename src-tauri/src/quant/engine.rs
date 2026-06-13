use std::path::Path;

use crate::quant::recipe::RecipeState;

/// Estimate the quantized size (bytes) for a tensor given shape and quant type.
pub fn estimate_quantized_size(shape: &[u64], bits_per_weight: f32) -> u64 {
    let elements: u64 = shape.iter().product();
    (elements as f64 * bits_per_weight as f64 / 8.0).ceil() as u64
}

/// Compute VRAM estimate for the entire recipe in MB.
pub fn estimate_vram_mb(recipe: &RecipeState, tensor_shapes: &[Vec<u64>]) -> f64 {
    let mut total: u64 = 0;
    for (i, assign) in recipe.assignments.iter().enumerate() {
        let shape = tensor_shapes.get(i).map(|s| s.as_slice()).unwrap_or(&[]);
        total += estimate_quantized_size(shape, assign.quant_type.bits_per_weight());
    }
    total as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_q8_size() {
        let shape = vec![4096, 4096];
        let size = estimate_quantized_size(&shape, 8.0);
        assert_eq!(size, 16_777_216);
    }

    #[test]
    fn test_estimate_q4_size() {
        let shape = vec![4096, 4096];
        let size = estimate_quantized_size(&shape, 4.8);
        assert_eq!(size, 10_066_330);
    }
}

/// Stub: apply recipe to produce a new GGUF. Full implementation requires
/// llama.cpp linked for per-row quantize functions. This stub copies the
/// source file as-is so the command pipeline can be tested end-to-end.
pub fn apply_recipe_stub(source: &Path, dest: &Path, _recipe: &RecipeState) -> Result<(), String> {
    std::fs::copy(source, dest)
        .map_err(|e| format!("Failed to copy GGUF: {}", e))
        .map(|_| ())
}
