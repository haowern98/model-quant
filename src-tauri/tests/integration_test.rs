use model_surgery::gguf::reader::parse_gguf;
use model_surgery::quant::engine::estimate_quantized_size;
use model_surgery::quant::recipe::{QuantType, RecipeState};

#[test]
fn test_gguf_parse_invalid_file() {
    let result = parse_gguf(std::path::Path::new("nonexistent.gguf"));
    assert!(result.is_err());
}

#[test]
fn test_recipe_full_workflow() {
    let tensors = vec![
        "tok_embeddings.weight".to_string(),
        "layers.0.attention.wq.weight".to_string(),
        "layers.0.feed_forward.w1.weight".to_string(),
        "output.weight".to_string(),
    ];

    let mut recipe = RecipeState::new("test.gguf".into(), tensors.clone(), QuantType::Q4_K_M);

    // Assign attention tensors to Q6_K
    recipe.assign_by_pattern("all_attn", QuantType::Q6_K);

    // Assign embeddings to Q8_0
    recipe.assign_by_pattern("all_embeddings", QuantType::Q8_0);

    // Verify
    let emb = recipe
        .assignments
        .iter()
        .find(|a| a.tensor_name == "tok_embeddings.weight")
        .unwrap();
    assert_eq!(emb.quant_type, "Q8_0");

    let attn = recipe
        .assignments
        .iter()
        .find(|a| a.tensor_name.contains("attention"))
        .unwrap();
    assert_eq!(attn.quant_type, "Q6_K");

    let ffn = recipe
        .assignments
        .iter()
        .find(|a| a.tensor_name.contains("feed_forward"))
        .unwrap();
    assert_eq!(ffn.quant_type, "Q4_K_M");

    // JSON roundtrip
    let json = serde_json::to_string(&recipe).unwrap();
    let deserialized: RecipeState = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.assignments.len(), 4);

    // Estimate VRAM
    let shapes: Vec<Vec<u64>> = tensors.iter().map(|_| vec![4096, 4096]).collect();
    let vram = model_surgery::quant::engine::estimate_vram_mb(&recipe, &shapes);
    assert!(vram > 0.0);
}

#[test]
fn test_quant_type_bits() {
    assert_eq!(QuantType::F16.bits_per_weight(), 16.0);
    assert_eq!(QuantType::Q8_0.bits_per_weight(), 8.0);
    assert_eq!(QuantType::Q4_K_M.bits_per_weight(), 4.8);
    assert_eq!(QuantType::Q2_K.bits_per_weight(), 2.6);
}

#[test]
fn test_estimate_quantized_size() {
    let shape = vec![4096, 4096];
    let f16_size = estimate_quantized_size(&shape, 16.0);
    let q8_size = estimate_quantized_size(&shape, 8.0);
    let q4_size = estimate_quantized_size(&shape, 4.8);

    assert_eq!(f16_size, 33_554_432);
    assert_eq!(q8_size, 16_777_216);
    assert!(q4_size < f16_size);
    assert!(q4_size < q8_size);
}

#[test]
fn test_parse_real_gguf_gemma_4b() {
    let path = std::path::Path::new(
        r"C:\Users\Wu Family Computer\.lmstudio\models\lmstudio-community\gemma-3-4b-it-GGUF\gemma-3-4b-it-Q4_K_M.gguf",
    );
    if !path.exists() {
        eprintln!("Skipping: file not found at {}", path.display());
        return;
    }

    let result = parse_gguf(path);
    match result {
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
        Ok(model) => {
            println!("=== METADATA ===");
            println!("  name:         {}", model.metadata.name);
            println!("  architecture: {}", model.metadata.architecture);
            println!("  total params: {}", model.metadata.total_params);
            println!(
                "  FP16 size:    {} MB",
                model.metadata.total_size_fp16 / 1_048_576
            );

            println!("\n=== TENSORS (first 20) ===");
            for t in model.tensors.iter().take(20) {
                println!(
                    "  {:60} layer={:4} group={:15} quant={:6} shape={:?} size={}",
                    t.name, t.layer_index, t.layer_group, t.current_quant, t.shape, t.size_bytes
                );
            }

            println!("\n  ... {} total tensors", model.tensors.len());

            println!("\n=== LAYER DISTRIBUTION ===");
            let mut groups: std::collections::HashMap<i32, usize> =
                std::collections::HashMap::new();
            for t in &model.tensors {
                *groups.entry(t.layer_index).or_default() += 1;
            }
            let mut sorted: Vec<_> = groups.into_iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            for (layer_idx, count) in sorted {
                println!("  layer index {:4}: {} tensors", layer_idx, count);
            }

            assert!(model.tensors.len() > 0, "should have tensors");
            assert!(
                !model.metadata.architecture.is_empty(),
                "should have architecture"
            );
        }
    }
}
