// ---- Quantization Types ----

export type QuantType =
  | "F32"
  | "BF16"
  | "F16"
  | "Q8_0"
  | "Q6_K"
  | "Q5_K"
  | "Q5_K_M"
  | "Q5_1"
  | "Q5_0"
  | "Q4_K"
  | "Q4_K_M"
  | "Q4_1"
  | "Q4_0"
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
    value: "Q5_1",
    label: "Q5_1",
    bitsPerWeight: 6.0,
    quality: "Legacy high quality",
  },
  {
    value: "Q5_0",
    label: "Q5_0",
    bitsPerWeight: 5.5,
    quality: "Legacy high quality",
  },
  { value: "Q4_K", label: "Q4_K", bitsPerWeight: 4.5, quality: "Good" },
  {
    value: "Q4_1",
    label: "Q4_1",
    bitsPerWeight: 5.0,
    quality: "Legacy good",
  },
  {
    value: "Q4_0",
    label: "Q4_0",
    bitsPerWeight: 4.5,
    quality: "Legacy good",
  },
  { value: "Q3_K", label: "Q3_K", bitsPerWeight: 3.4, quality: "Passable" },
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
  fallback: QuantType = "Q4_K",
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
  quantPreflight: TensorQuantPreflight;
}

export interface TensorQuantPreflight {
  canQuantize: boolean;
  allowedTargetQuants: QuantType[];
  blockedReason: string | null;
}

export interface ModelInfo {
  metadata: ModelMetadata;
  tensors: TensorInfo[];
  currentUniformQuant: string;
  totalSizeBytes: number;
}

export interface TensorValuesPreview {
  values: number[];
  rows: number;
  cols: number;
  totalRows: number;
  totalCols: number;
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
export type BenchmarkRunId =
  | "ppl_check"
  | "gpqa_diamond"
  | "humaneval"
  | "terminal_bench"
  | "mmmu_pro"
  | "mmlu_pro"
  | "mmlu_redux"
  | "supergpqa"
  | "claw_eval";
export type GpqaShotMode = "zero_shot" | "five_shot_cot";
export type GpqaThinkingMode = "off" | "on";

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

export interface BenchmarkOutputEvent {
  message: string;
}

export interface ApiOutputEvent {
  message: string;
  mode?: "line" | "append";
  stream?: "reasoning" | "visible" | null;
  header?: string | null;
}

export interface BenchmarkOutputLine {
  id: number;
  timestamp: string;
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
  requestedTargetCount: number;
  verifiedTargetCount: number;
  baselineBenchmark: RuntimeBenchmark | null;
  qualityEval: RecipeQualityEval | null;
  standardEval: StandardEvalReport | null;
}

export interface HardwareSnapshot {
  cpuName: string;
  cpuUsagePercent: number;
  ramUsedBytes: number;
  ramTotalBytes: number;
  gpuName: string | null;
  gpuUsagePercent: number | null;
  vramUsedMb: number | null;
  vramTotalMb: number | null;
  gpuTemperatureC: number | null;
  gpuPowerW: number | null;
  cpuTemperatureC: number | null;
  cpuPowerW: number | null;
}

export interface ModelInspectorApiStatus {
  running: boolean;
  baseUrl: string | null;
  apiKey: string | null;
  modelId: string | null;
}

export interface GpqaDiamondStatus {
  ready: boolean;
  statusLabel: string;
  python: string | null;
  evalscope: string | null;
  datasetReady: boolean;
  datasetStatusLabel: string;
  datasetPath: string | null;
  datasetHash: string | null;
  datasetUrl: string;
  expectedDatasetHash: string;
  detail: string;
}

export interface HumanEvalStatus {
  ready: boolean;
  statusLabel: string;
  python: string | null;
  evalscope: string | null;
  dockerReady: boolean;
  docker: string | null;
  detail: string;
}

export interface TerminalBenchStatus {
  ready: boolean;
  statusLabel: string;
  harborReady: boolean;
  harbor: string | null;
  dockerReady: boolean;
  docker: string | null;
  detail: string;
}

export interface MmmuProStatus {
  ready: boolean;
  statusLabel: string;
  detail: string;
}

export interface HumanEvalDatasetStatus {
  datasetReady: boolean;
  datasetStatusLabel: string;
  datasetPath: string | null;
  datasetHash: string | null;
  datasetUrl: string;
  expectedDatasetHash: string;
}

export interface MmmuProDatasetStatus {
  datasetReady: boolean;
  datasetStatusLabel: string;
  datasetPath: string | null;
  datasetHash: string | null;
  datasetUrl: string;
  expectedDatasetHash: string;
}

export interface TerminalBenchDatasetStatus {
  datasetReady: boolean;
  datasetStatusLabel: string;
  datasetPath: string | null;
  datasetHash: string | null;
  datasetUrl: string;
  expectedDatasetHash: string;
}

export interface TerminalBenchDatasetRow {
  index: number;
  taskId: string;
  instruction: string;
  path: string;
}

export interface HumanEvalDatasetRow {
  index: number;
  taskId: string;
  entryPoint: string;
  prompt: string;
  canonicalSolution: string;
}

export interface MmmuProDatasetRow {
  index: number;
  taskId: string;
  subject: string;
  question: string;
  choices: string[];
  imageUrls: string[];
}

export interface GpqaDatasetRow {
  index: number;
  question: string;
  choices: string[];
  answer: string | null;
}

export interface GpqaBenchmarkConfigInput {
  seed: string;
  contextWindow: string;
  sampleLimit: string;
  temperature: string;
  thinking: GpqaThinkingMode;
  topK: string;
  repeatPenalty: string;
  presencePenalty: string;
  topP: string;
  minP: string;
}

export interface GpqaBenchmarkConfig {
  seed?: number;
  contextWindow: number;
  sampleLimit: number;
  temperature: number;
  thinking: GpqaThinkingMode;
  topK?: number;
  repeatPenalty?: number;
  presencePenalty?: number;
  topP?: number;
  minP?: number;
}

export const MMMU_PRO_SUBJECTS = [
  { id: "Accounting", label: "Accounting" },
  { id: "Agriculture", label: "Agriculture" },
  { id: "Architecture_and_Engineering", label: "Architecture and Engineering" },
  { id: "Art", label: "Art" },
  { id: "Art_Theory", label: "Art Theory" },
  { id: "Basic_Medical_Science", label: "Basic Medical Science" },
  { id: "Biology", label: "Biology" },
  { id: "Chemistry", label: "Chemistry" },
  { id: "Clinical_Medicine", label: "Clinical Medicine" },
  { id: "Computer_Science", label: "Computer Science" },
  { id: "Design", label: "Design" },
  { id: "Diagnostics_and_Laboratory_Medicine", label: "Diagnostics and Laboratory Medicine" },
  { id: "Economics", label: "Economics" },
  { id: "Electronics", label: "Electronics" },
  { id: "Energy_and_Power", label: "Energy and Power" },
  { id: "Finance", label: "Finance" },
  { id: "Geography", label: "Geography" },
  { id: "History", label: "History" },
  { id: "Literature", label: "Literature" },
  { id: "Manage", label: "Manage" },
  { id: "Marketing", label: "Marketing" },
  { id: "Materials", label: "Materials" },
  { id: "Math", label: "Math" },
  { id: "Mechanical_Engineering", label: "Mechanical Engineering" },
  { id: "Music", label: "Music" },
  { id: "Pharmacy", label: "Pharmacy" },
  { id: "Physics", label: "Physics" },
  { id: "Psychology", label: "Psychology" },
  { id: "Public_Health", label: "Public Health" },
  { id: "Sociology", label: "Sociology" },
] as const;

export type MmmuProSubjectId = (typeof MMMU_PRO_SUBJECTS)[number]["id"];

export interface MmmuProSubjectConfigInput {
  subject: MmmuProSubjectId;
  included: boolean;
  sampleLimit: string;
}

export interface MmmuProBenchmarkConfigInput extends GpqaBenchmarkConfigInput {
  subjects?: MmmuProSubjectConfigInput[];
}

export interface MmmuProSubjectConfig {
  subject: MmmuProSubjectId;
  sampleLimit: number;
}

export interface MmmuProBenchmarkConfig extends GpqaBenchmarkConfig {
  subjects: MmmuProSubjectConfig[];
}

export interface TerminalBenchBenchmarkConfigInput {
  seed: string;
  contextWindow: string;
  samples: string;
  runsPerTask: string;
  maxTurns: string;
  timeoutMultiplier: string;
  temperature: string;
  thinking: GpqaThinkingMode;
  topK: string;
  repeatPenalty: string;
  presencePenalty: string;
  topP: string;
  minP: string;
}

export interface TerminalBenchBenchmarkConfig {
  seed?: number;
  contextWindow: number;
  samples?: number;
  runsPerTask: number;
  maxTurns: number;
  timeoutMultiplier: number;
  temperature: number;
  thinking: GpqaThinkingMode;
  topK?: number;
  repeatPenalty?: number;
  presencePenalty?: number;
  topP?: number;
  minP?: number;
}

export interface RecipeQualityEval {
  baselineNll: number | null;
  baselinePpl: number | null;
  baselinePplUncertainty: number | null;
  baselineEvalMs: number | null;
  baselineVramPeakMb: number | null;
  baselineVramAllocatedMb: number | null;
  recipeNll: number;
  recipePpl: number;
  recipePplUncertainty: number;
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
  sampleAudits: StandardEvalSampleAuditReport[];
}

export interface StandardEvalTaskReport {
  task: string;
  metric?: string;
  nShot?: number;
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

export interface StandardEvalSampleAuditReport {
  task: string;
  docId: string;
  sampleIndex: number;
  prompt: string;
  targetDelimiter: string;
  goldIndex: number;
  baselinePredictionIndex: number | null;
  recipePredictionIndex: number;
  baselineCorrect: boolean | null;
  recipeCorrect: boolean;
  flipType: string;
  choices: StandardEvalChoiceAuditReport[];
}

export interface StandardEvalChoiceAuditReport {
  index: number;
  choice: string;
  continuation: string;
  denominator: number;
  baselineNll: number | null;
  baselineLoglikelihood: number | null;
  baselineScore: number | null;
  recipeNll: number;
  recipeLoglikelihood: number;
  recipeScore: number;
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
