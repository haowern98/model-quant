import type {
  ModelInfo,
  RecipeState,
  QuantType,
  AssignPattern,
} from "../../src/types";

export function createMockBridge() {
  let cancelRequested = false;
  const mockState = window as Window & {
    __MODEL_SURGERY_LAST_GPQA_ARGS__?: unknown;
  };
  mockState.__MODEL_SURGERY_LAST_GPQA_ARGS__ = null;
  let gpqaReady =
    typeof window === "undefined" ||
    !new URLSearchParams(window.location.search).has("gpqaMissing");
  let gpqaDatasetReady = gpqaReady;
  const allowedTargetQuants: QuantType[] = [
    "F32",
    "BF16",
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K",
    "Q4_K",
    "Q3_K",
    "Q2_K",
  ];
  const bf16AllowedPreflight = {
    canQuantize: true,
    allowedTargetQuants: allowedTargetQuants.filter((q) => q !== "F32"),
    blockedReason: null,
  };
  const q8AllowedPreflight = {
    canQuantize: true,
    allowedTargetQuants: ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"] satisfies QuantType[],
    blockedReason: null,
  };
  const q8OnlyPreflight = {
    canQuantize: true,
    allowedTargetQuants: ["Q8_0"] satisfies QuantType[],
    blockedReason: null,
  };
  const blockedNormPreflight = {
    canQuantize: false,
    allowedTargetQuants: [],
    blockedReason: "1D tensors are not quantizable weight matrices",
  };
  const mockModel: ModelInfo = {
    metadata: {
      name: "Mock-Llama-3-8B",
      architecture: "llama",
      totalParams: 8_030_000_000,
      totalSizeFp16: 14.96 * 1024 * 1024 * 1024,
    },
    tensors: [
      {
        name: "tok_embeddings.weight",
        shape: [128256, 4096],
        currentQuant: "BF16",
        sizeBytes: 262_000_000,
        layerIndex: -1,
        layerGroup: "embedding",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.attention.wq.weight",
        shape: [4096, 4096],
        currentQuant: "Q8_0",
        sizeBytes: 80_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: q8AllowedPreflight,
      },
      {
        name: "layers.0.attention.wk.weight",
        shape: [4096, 1024],
        currentQuant: "BF16",
        sizeBytes: 20_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.attention.wv.weight",
        shape: [4096, 1024],
        currentQuant: "BF16",
        sizeBytes: 20_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.attention.wo.weight",
        shape: [4096, 4096],
        currentQuant: "BF16",
        sizeBytes: 80_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.attention.short.weight",
        shape: [128, 4096],
        currentQuant: "Q8_0",
        sizeBytes: 2_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: q8OnlyPreflight,
      },
      {
        name: "layers.0.feed_forward.w1.weight",
        shape: [14336, 4096],
        currentQuant: "BF16",
        sizeBytes: 117_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.feed_forward.w2.weight",
        shape: [4096, 14336],
        currentQuant: "BF16",
        sizeBytes: 117_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.0.feed_forward.w3.weight",
        shape: [14336, 4096],
        currentQuant: "BF16",
        sizeBytes: 117_000_000,
        layerIndex: 0,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "layers.1.attention.wq.weight",
        shape: [4096, 4096],
        currentQuant: "Q8_0",
        sizeBytes: 80_000_000,
        layerIndex: 1,
        layerGroup: "attention",
        quantPreflight: q8AllowedPreflight,
      },
      {
        name: "layers.1.attention.wk.weight",
        shape: [4096, 1024],
        currentQuant: "BF16",
        sizeBytes: 20_000_000,
        layerIndex: 1,
        layerGroup: "attention",
        quantPreflight: bf16AllowedPreflight,
      },
      {
        name: "output_norm.weight",
        shape: [4096],
        currentQuant: "F32",
        sizeBytes: 16_000,
        layerIndex: -1,
        layerGroup: "output_norm",
        quantPreflight: blockedNormPreflight,
      },
      {
        name: "output.weight",
        shape: [128256, 4096],
        currentQuant: "BF16",
        sizeBytes: 262_000_000,
        layerIndex: -1,
        layerGroup: "output",
        quantPreflight: bf16AllowedPreflight,
      },
    ],
    currentUniformQuant: "BF16",
    totalSizeBytes: 4_920_000_000,
  };

  const recipe: RecipeState = {
    id: "mock-recipe",
    baseModel: "mock-model.gguf",
    assignments: mockModel.tensors.map((t) => ({
      tensorName: t.name,
      quantType: t.currentQuant,
      sourceQuant: t.currentQuant,
    })),
    profile: null,
    status: "draft",
  };

  const handlers: Record<string, (args?: Record<string, unknown>) => unknown> =
    {
      open_model: () => mockModel,
      get_tensors: () => mockModel.tensors,
      assign_quant: (args) => {
        const { tensorNames, quantType } = args!;
        const names = tensorNames as string[];
        recipe.assignments = recipe.assignments.map((a) =>
          names.includes(a.tensorName)
            ? { ...a, quantType: quantType as QuantType }
            : a,
        );
        recipe.status = "draft";
        return recipe;
      },
      assign_all: (args) => {
        const { quantType } = args!;
        recipe.assignments = recipe.assignments.map((a) => ({
          ...a,
          quantType: quantType as QuantType,
        }));
        recipe.status = "draft";
        return recipe;
      },
      assign_by_pattern: (args) => {
        const { pattern, quantType } = args!;
        const isMatch = (name: string, p: AssignPattern): boolean => {
          if (p === "all_attn") return name.includes("attention");
          if (p === "all_ffn") return name.includes("feed_forward");
          if (p === "all_embeddings")
            return name.includes("embedding") || name.includes("output");
          return true;
        };
        recipe.assignments = recipe.assignments.map((a) =>
          isMatch(a.tensorName, pattern as AssignPattern)
            ? { ...a, quantType: quantType as QuantType }
            : a,
        );
        recipe.status = "draft";
        return recipe;
      },
      test_recipe: async (args) => {
        cancelRequested = false;
        for (let interval = 0; interval < 4; interval += 1) {
          await new Promise((resolve) => setTimeout(resolve, 250));
          if (cancelRequested) throw new Error("Recipe test cancelled");
        }
        const isCompare = args?.testMode === "compare_baseline";
        const isDefault = args?.evalPreset !== "quick";
        const standardSampleCount = isDefault ? 300 : 36;
        const changedTargetCount = recipe.assignments.filter((assignment) => {
          const tensor = mockModel.tensors.find(
            (candidate) => candidate.name === assignment.tensorName,
          );
          return tensor?.currentQuant !== assignment.quantType;
        }).length;
        return {
          promptEvalTps: 1247,
          tokenGenTps: 88.3,
          ttftMs: 412,
          promptEvalMs: 18,
          generationMs: 180,
          vramPeakMb: 5820,
          vramAllocatedMb: 5760,
          diskSizeMb: 5780,
          elapsedMs: isCompare ? 6200 : 3200,
          loadMs: 800,
          testMode: isCompare
            ? "native_recipe_eval_v1"
            : "native_recipe_single_v1",
          statusMessage: isCompare
            ? "Mock baseline comparison completed"
            : "Mock standalone recipe test completed",
          nativeRuntime: null,
          modelTensorCount: mockModel.tensors.length,
          modelMetadataCount: 32,
          copiedTensorCount: changedTargetCount > 0 ? 8 : 0,
          convertedTensorCount: changedTargetCount,
          convertedBytesBefore:
            changedTargetCount > 0 ? changedTargetCount * 80_000_000 : 0,
          convertedBytesAfter:
            changedTargetCount > 0 ? changedTargetCount * 40_000_000 : 0,
          requestedTargetCount: changedTargetCount,
          verifiedTargetCount: changedTargetCount,
          baselineBenchmark: isCompare
            ? {
                promptEvalTps: 1180,
                tokenGenTps: 76.5,
                ttftMs: 480,
                promptEvalMs: 21,
                generationMs: 210,
                vramPeakMb: 7140,
                vramAllocatedMb: 7040,
                loadMs: 950,
                elapsedMs: 1181,
                modelTensorCount: mockModel.tensors.length,
              }
            : null,
          qualityEval: {
            baselineNll: isCompare ? 1.92 : null,
            baselinePpl: isCompare ? 6.82 : null,
            baselinePplUncertainty: isCompare ? 0.08 : null,
            baselineEvalMs: isCompare ? 920 : null,
            baselineVramPeakMb: isCompare ? 7140 : null,
            baselineVramAllocatedMb: isCompare ? 7040 : null,
            recipeNll: 1.96,
            recipePpl: 7.1,
            recipePplUncertainty: 0.09,
            recipeEvalMs: 860,
            recipeVramPeakMb: 5820,
            recipeVramAllocatedMb: 5760,
            pplDelta: isCompare ? 0.28 : 0,
            pplDeltaPercent: isCompare ? 4.11 : 0,
            evalTokenCount: 384,
            evalSampleCount: isDefault ? 10 : 4,
            skippedSampleCount: 0,
          },
          standardEval: {
            sampleCount: standardSampleCount,
            taskCount: 6,
            baselineAccuracy: isCompare ? 0.76 : null,
            recipeAccuracy: 0.74,
            accuracyDelta: isCompare ? -0.02 : null,
            correctToWrongCount: isCompare ? 8 : 0,
            wrongToCorrectCount: isCompare ? 2 : 0,
            baselineAvgMargin: isCompare ? 0.42 : null,
            recipeAvgMargin: 0.38,
            marginDelta: isCompare ? -0.04 : null,
            tasks: [
              "arc_challenge",
              "arc_easy",
              "gsm8k",
              "hellaswag",
              "mmlu_mixed",
              "truthfulqa_mc",
            ].map((task) => ({
              task,
              sampleCount: standardSampleCount / 6,
              baselineCorrectCount: isCompare ? 38 : null,
              recipeCorrectCount: 37,
              correctToWrongCount: isCompare ? 2 : 0,
              wrongToCorrectCount: isCompare ? 1 : 0,
              samePredictionCount: isCompare ? 47 : 0,
              baselineAccuracy: isCompare ? 0.76 : null,
              recipeAccuracy: 0.74,
              accuracyDelta: isCompare ? -0.02 : null,
              baselineAvgMargin: isCompare ? 0.42 : null,
              recipeAvgMargin: 0.38,
              marginDelta: isCompare ? -0.04 : null,
              baselineAvgCorrectNll: isCompare ? 1.4 : null,
              recipeAvgCorrectNll: 1.45,
            })),
          },
        };
      },
      cancel_recipe_test: () => {
        cancelRequested = true;
      },
      get_hardware_snapshot: () => ({
        cpuName: "Mock Ryzen CPU",
        cpuUsagePercent: 24,
        ramUsedBytes: 12.6 * 1024 * 1024 * 1024,
        ramTotalBytes: 64 * 1024 * 1024 * 1024,
        gpuName: "Mock NVIDIA GPU",
        gpuUsagePercent: 72,
        vramUsedMb: 8192,
        vramTotalMb: 24576,
        gpuTemperatureC: 62,
        gpuPowerW: 286,
        cpuTemperatureC: null,
        cpuPowerW: null,
      }),
      start_modelinspector_api: () => ({
        running: true,
        baseUrl: "http://127.0.0.1:12345/v1",
        apiKey: "mock-modelinspector-token",
        modelId: "mock-model.gguf",
      }),
      stop_modelinspector_api: () => ({
        running: false,
        baseUrl: null,
        apiKey: null,
        modelId: null,
      }),
      get_modelinspector_api_status: () => ({
        running: false,
        baseUrl: null,
        apiKey: null,
        modelId: null,
      }),
      get_gpqa_diamond_status: () => ({
        ready: gpqaReady && gpqaDatasetReady,
        statusLabel: gpqaReady ? (gpqaDatasetReady ? "Ready" : "Download") : "Install",
        python: gpqaReady ? "3.11.8" : null,
        evalscope: gpqaReady ? "1.8.0" : null,
        datasetReady: gpqaDatasetReady,
        datasetStatusLabel: gpqaDatasetReady ? "Verified" : "Missing",
        datasetPath: gpqaDatasetReady ? "C:\\Users\\mock\\AppData\\Local\\MI\\g\\datasets\\gpqa_diamond_dataset_ready.json" : null,
        datasetHash: gpqaDatasetReady ? "marker-v1" : null,
        datasetUrl: "AI-ModelScope/gpqa_diamond",
        expectedDatasetHash: "EvalScope dataset cache marker",
        detail: gpqaReady && gpqaDatasetReady
          ? "Mock EvalScope GPQA Diamond harness and dataset are ready."
          : gpqaReady
            ? "Mock GPQA dataset is not downloaded."
          : "Mock GPQA harness is not installed.",
      }),
      install_gpqa_diamond_harness: async () => {
        gpqaReady = true;
        return {
          ready: gpqaDatasetReady,
          statusLabel: gpqaDatasetReady ? "Ready" : "Download",
          python: "3.11.8",
          evalscope: "1.8.0",
          datasetReady: gpqaDatasetReady,
          datasetStatusLabel: gpqaDatasetReady ? "Verified" : "Missing",
          datasetPath: gpqaDatasetReady ? "C:\\Users\\mock\\AppData\\Local\\MI\\g\\datasets\\gpqa_diamond_dataset_ready.json" : null,
          datasetHash: gpqaDatasetReady ? "marker-v1" : null,
          datasetUrl: "AI-ModelScope/gpqa_diamond",
          expectedDatasetHash: "EvalScope dataset cache marker",
          detail: gpqaDatasetReady
            ? "Mock EvalScope GPQA Diamond harness and dataset are ready."
            : "Mock GPQA dataset is not downloaded.",
        };
      },
      download_gpqa_diamond_dataset: async () => {
        gpqaReady = true;
        gpqaDatasetReady = true;
        return {
          ready: true,
          statusLabel: "Ready",
          python: "3.11.8",
          evalscope: "1.8.0",
          datasetReady: true,
          datasetStatusLabel: "Verified",
          datasetPath: "C:\\Users\\mock\\AppData\\Local\\MI\\g\\datasets\\gpqa_diamond_dataset_ready.json",
          datasetHash: "marker-v1",
          datasetUrl: "AI-ModelScope/gpqa_diamond",
          expectedDatasetHash: "EvalScope dataset cache marker",
          detail: "Mock EvalScope GPQA Diamond harness and dataset are ready.",
        };
      },
      run_gpqa_diamond_benchmark: async (args) => {
        mockState.__MODEL_SURGERY_LAST_GPQA_ARGS__ = args;
        return {
        promptEvalTps: 0,
        tokenGenTps: 0,
        ttftMs: 0,
        promptEvalMs: 0,
        generationMs: 0,
        vramPeakMb: 0,
        vramAllocatedMb: 0,
        diskSizeMb: 0,
        elapsedMs: 1000,
        loadMs: 0,
        testMode: "official_gpqa_diamond",
        statusMessage: `Mock GPQA Diamond official harness completed with ${args?.shotMode === "zero_shot" ? "0-shot CoT" : "5-shot CoT"}.`,
        nativeRuntime: "EvalScope GPQA Diamond mock report",
        modelTensorCount: mockModel.tensors.length,
        modelMetadataCount: 32,
        copiedTensorCount: 0,
        convertedTensorCount: 0,
        convertedBytesBefore: 0,
        convertedBytesAfter: 0,
        requestedTargetCount: 0,
        verifiedTargetCount: 0,
        baselineBenchmark: null,
        qualityEval: null,
        standardEval: {
          sampleCount: 198,
          taskCount: 1,
          baselineAccuracy: null,
          recipeAccuracy: 0.631,
          accuracyDelta: null,
          correctToWrongCount: 0,
          wrongToCorrectCount: 0,
          baselineAvgMargin: null,
          recipeAvgMargin: 0,
          marginDelta: null,
          tasks: [
            {
              task: "gpqa_diamond",
              sampleCount: 198,
              baselineCorrectCount: null,
              recipeCorrectCount: 125,
              correctToWrongCount: 0,
              wrongToCorrectCount: 0,
              samePredictionCount: 0,
              baselineAccuracy: null,
              recipeAccuracy: 0.631,
              accuracyDelta: null,
              baselineAvgMargin: null,
              recipeAvgMargin: 0,
              marginDelta: null,
              baselineAvgCorrectNll: null,
              recipeAvgCorrectNll: 0,
            },
          ],
          sampleAudits: [],
        },
      };
      },
      cancel_official_benchmark: () => {},
      save_recipe: () => {
        recipe.status = "saved";
      },
      export_gguf: () => {},
      load_recipe: () => recipe,
      list_recipes: () => [
        {
          id: "mock-recipe",
          baseModel: "mock-model.gguf",
          status: "saved",
          createdAt: "2026-05-23",
        },
      ],
    };

  return async (
    cmd: string,
    args?: Record<string, unknown>,
  ): Promise<unknown> => {
    const handler = handlers[cmd];
    if (!handler) throw new Error(`Unknown command: ${cmd}`);
    return handler(args);
  };
}
