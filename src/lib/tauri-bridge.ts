import type {
  ModelInfo,
  TensorInfo,
  RecipeState,
  BenchmarkResult,
  AssignPattern,
  QuantType,
  RecipeSummary,
  RecipeTestMode,
  RecipeEvalPreset,
  HardwareSnapshot,
  ModelInspectorApiStatus,
  GpqaBenchmarkConfig,
  GpqaDiamondStatus,
  GpqaShotMode,
} from "../types";

let invokeFn: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

try {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  invokeFn = tauriInvoke;
} catch {
  invokeFn = () => {
    throw new Error(
      "Tauri bridge not available. Inject mock via setMockInvoke().",
    );
  };
}

export async function invoke<T>(
  cmd: string,
  args?: Record<string, unknown>,
): Promise<T> {
  return invokeFn(cmd, args) as Promise<T>;
}

export function setMockInvoke(
  fn: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>,
): void {
  invokeFn = fn;
}

// Typed command wrappers

export async function openModel(path: string): Promise<ModelInfo> {
  return invoke<ModelInfo>("open_model", { path });
}

export async function getTensors(): Promise<TensorInfo[]> {
  return invoke<TensorInfo[]>("get_tensors");
}

export async function assignQuant(
  tensorNames: string[],
  quantType: QuantType,
): Promise<RecipeState> {
  return invoke<RecipeState>("assign_quant", { tensorNames, quantType });
}

export async function assignAll(quantType: QuantType): Promise<RecipeState> {
  return invoke<RecipeState>("assign_all", { quantType });
}

export async function assignByPattern(
  pattern: AssignPattern,
  quantType: QuantType,
): Promise<RecipeState> {
  return invoke<RecipeState>("assign_by_pattern", { pattern, quantType });
}

export async function testRecipe(
  recipe: RecipeState,
  promptTokens: number,
  testMode: RecipeTestMode,
  evalPreset: RecipeEvalPreset,
): Promise<BenchmarkResult> {
  return invoke<BenchmarkResult>("test_recipe", {
    recipe,
    promptTokens,
    testMode,
    evalPreset,
  });
}

export async function cancelRecipeTest(): Promise<void> {
  return invoke<void>("cancel_recipe_test");
}

export async function getHardwareSnapshot(): Promise<HardwareSnapshot> {
  return invoke<HardwareSnapshot>("get_hardware_snapshot");
}

export async function startModelInspectorApi(options?: {
  benchmarkLabel?: string;
  contextWindow?: number;
}): Promise<ModelInspectorApiStatus> {
  return invoke<ModelInspectorApiStatus>("start_modelinspector_api", {
    benchmarkLabel: options?.benchmarkLabel ?? null,
    contextWindow: options?.contextWindow ?? null,
  });
}

export async function stopModelInspectorApi(): Promise<ModelInspectorApiStatus> {
  return invoke<ModelInspectorApiStatus>("stop_modelinspector_api");
}

export async function getModelInspectorApiStatus(): Promise<ModelInspectorApiStatus> {
  return invoke<ModelInspectorApiStatus>("get_modelinspector_api_status");
}

export async function getGpqaDiamondStatus(): Promise<GpqaDiamondStatus> {
  return invoke<GpqaDiamondStatus>("get_gpqa_diamond_status");
}

export async function installGpqaDiamondHarness(): Promise<GpqaDiamondStatus> {
  return invoke<GpqaDiamondStatus>("install_gpqa_diamond_harness");
}

export async function downloadGpqaDiamondDataset(): Promise<GpqaDiamondStatus> {
  return invoke<GpqaDiamondStatus>("download_gpqa_diamond_dataset");
}

export async function runGpqaDiamondBenchmark(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  shotMode: GpqaShotMode,
  config: GpqaBenchmarkConfig,
): Promise<BenchmarkResult> {
  return invoke<BenchmarkResult>("run_gpqa_diamond_benchmark", {
    baseUrl,
    apiKey,
    modelId,
    shotMode,
    config,
  });
}

export async function cancelOfficialBenchmark(): Promise<void> {
  return invoke<void>("cancel_official_benchmark");
}

export async function saveRecipe(
  path: string,
  recipe: RecipeState,
): Promise<void> {
  return invoke<void>("save_recipe", { path, recipe });
}

export async function exportGguf(
  path: string,
  recipe: RecipeState,
): Promise<void> {
  return invoke<void>("export_gguf", { path, recipe });
}

export async function loadRecipe(path: string): Promise<RecipeState> {
  return invoke<RecipeState>("load_recipe", { path });
}

export async function listRecipes(): Promise<RecipeSummary[]> {
  return invoke<RecipeSummary[]>("list_recipes");
}
