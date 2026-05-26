import type {
  ModelInfo,
  TensorInfo,
  RecipeState,
  BenchmarkResult,
  AssignPattern,
  QuantType,
  RecipeSummary,
  RecipeTestMode,
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
): Promise<BenchmarkResult> {
  return invoke<BenchmarkResult>("test_recipe", {
    recipe,
    promptTokens,
    testMode,
  });
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
