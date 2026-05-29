// ---- Quantization Types ----

export type QuantType =
  | "F32"
  | "BF16"
  | "F16"
  | "Q8_0"
  | "Q6_K"
  | "Q5_K"
  | "Q5_K_M"
  | "Q4_K"
  | "Q4_K_M"
  | "Q3_K"
  | "Q3_K_M"
  | "Q2_K";

export const QUANT_TYPES: {
  value: QuantType;
  label: string;
  bitsPerWeight: number;
  quality: string;
}[] = [
  {
    value: "F32",
    label: "F32",
    bitsPerWeight: 32.0,
    quality: "Full precision",
  },
  {
    value: "BF16",
    label: "BF16",
    bitsPerWeight: 16.0,
    quality: "Reference (brain float 16)",
  },
  {
    value: "F16",
    label: "F16",
    bitsPerWeight: 16.0,
    quality: "Reference (no quant)",
  },
  {
    value: "Q8_0",
    label: "Q8_0",
    bitsPerWeight: 8.0,
    quality: "Near-lossless",
  },
  {
    value: "Q6_K",
    label: "Q6_K",
    bitsPerWeight: 6.6,
    quality: "Very high quality",
  },
  { value: "Q5_K", label: "Q5_K", bitsPerWeight: 5.5, quality: "High quality" },
  {
    value: "Q5_K_M",
    label: "Q5_K_M",
    bitsPerWeight: 5.3,
    quality: "High quality",
  },
  { value: "Q4_K", label: "Q4_K", bitsPerWeight: 4.5, quality: "Good" },
  {
    value: "Q4_K_M",
    label: "Q4_K_M",
    bitsPerWeight: 4.8,
    quality: "Good (default trade-off)",
  },
  { value: "Q3_K", label: "Q3_K", bitsPerWeight: 3.4, quality: "Passable" },
  { value: "Q3_K_M", label: "Q3_K_M", bitsPerWeight: 3.9, quality: "Passable" },
  {
    value: "Q2_K",
    label: "Q2_K",
    bitsPerWeight: 2.6,
    quality: "Maximum compression",
  },
];

export function isQuantType(value: string): value is QuantType {
  return QUANT_TYPES.some((q) => q.value === value);
}

export function toTargetQuant(
  value: string | null | undefined,
  fallback: QuantType = "Q4_K_M",
): QuantType {
  return value && isQuantType(value) ? value : fallback;
}

// ---- Model ----

export interface ModelMetadata {
  name: string;
  architecture: string;
  totalParams: number;
  totalSizeFp16: number;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  currentQuant: string;
  sizeBytes: number;
  layerIndex: number;
  layerGroup:
    | "embedding"
    | "attention"
    | "norm"
    | "output_norm"
    | "output"
    | "other";
}

export interface ModelInfo {
  metadata: ModelMetadata;
  tensors: TensorInfo[];
  currentUniformQuant: string;
  totalSizeBytes: number;
}

// ---- Recipe ----

export interface QuantAssignment {
  tensorName: string;
  quantType: QuantType;
}

export interface RecipeProfile {
  vramEstimate: number;
  sizeSavedVsQ8: number;
}

export type RecipeStatus = "draft" | "profiled" | "applied" | "saved";

export interface RecipeState {
  id: string;
  baseModel: string;
  assignments: QuantAssignment[];
  profile: RecipeProfile | null;
  status: RecipeStatus;
}

export type RecipeTestMode = "single" | "compare_baseline";
export type RecipeEvalPreset = "quick" | "default";

// ---- Progress ----

export type ProgressStage =
  | "requantizing"
  | "writing"
  | "loading"
  | "benchmarking";

export interface ProgressEvent {
  stage: ProgressStage;
  percent: number;
  message: string;
}

// ---- Benchmark ----

export interface BenchmarkResult {
  promptEvalTps: number;
  tokenGenTps: number;
  ttftMs: number;
  promptEvalMs: number;
  generationMs: number;
  vramPeakMb: number;
  vramAllocatedMb: number;
  diskSizeMb: number;
  elapsedMs: number;
  loadMs: number;
  testMode: string;
  statusMessage: string;
  nativeRuntime: string | null;
  modelTensorCount: number | null;
  modelMetadataCount: number | null;
  copiedTensorCount: number;
  convertedTensorCount: number;
  convertedBytesBefore: number;
  convertedBytesAfter: number;
  baselineBenchmark: RuntimeBenchmark | null;
  qualityEval: RecipeQualityEval | null;
  standardEval: StandardEvalReport | null;
}

export interface RecipeQualityEval {
  baselineNll: number | null;
  baselinePpl: number | null;
  baselineEvalMs: number | null;
  baselineVramPeakMb: number | null;
  baselineVramAllocatedMb: number | null;
  recipeNll: number;
  recipePpl: number;
  recipeEvalMs: number;
  recipeVramPeakMb: number;
  recipeVramAllocatedMb: number;
  pplDelta: number;
  pplDeltaPercent: number;
  evalTokenCount: number;
  evalSampleCount: number;
  skippedSampleCount: number;
}

export interface RuntimeBenchmark {
  promptEvalTps: number;
  tokenGenTps: number;
  ttftMs: number;
  promptEvalMs: number;
  generationMs: number;
  vramPeakMb: number;
  vramAllocatedMb: number;
  loadMs: number;
  elapsedMs: number;
  modelTensorCount: number | null;
}

export interface StandardEvalReport {
  sampleCount: number;
  taskCount: number;
  baselineAccuracy: number | null;
  recipeAccuracy: number;
  accuracyDelta: number | null;
  correctToWrongCount: number;
  wrongToCorrectCount: number;
  baselineAvgMargin: number | null;
  recipeAvgMargin: number;
  marginDelta: number | null;
  tasks: StandardEvalTaskReport[];
}

export interface StandardEvalTaskReport {
  task: string;
  sampleCount: number;
  baselineCorrectCount: number | null;
  recipeCorrectCount: number;
  correctToWrongCount: number;
  wrongToCorrectCount: number;
  samePredictionCount: number;
  baselineAccuracy: number | null;
  recipeAccuracy: number;
  accuracyDelta: number | null;
  baselineAvgMargin: number | null;
  recipeAvgMargin: number;
  marginDelta: number | null;
  baselineAvgCorrectNll: number | null;
  recipeAvgCorrectNll: number;
}

// ---- Bulk Assign ----

export type AssignPattern = "all_attn" | "all_ffn" | "all_embeddings" | "all";

// ---- Recipe Summary ----

export interface RecipeSummary {
  id: string;
  baseModel: string;
  status: RecipeStatus;
  createdAt: string;
}
