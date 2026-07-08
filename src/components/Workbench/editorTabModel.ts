import type { BenchmarkResult } from "../../types";

export const GPQA_DETAILS_TAB_ID = "benchmark:gpqa_diamond:details";
export const GPQA_DATASET_TAB_ID = "benchmark:gpqa_diamond:dataset";
export const HUMANEVAL_DETAILS_TAB_ID = "benchmark:humaneval:details";
export const TERMINAL_BENCH_DETAILS_TAB_ID = "benchmark:terminal_bench_2_1:details";

export type EditorTab =
  | {
      id: `layer:${number}`;
      kind: "layer";
      layerIndex: number;
      label?: string;
    }
  | {
      id: `eval-results:${string}`;
      kind: "eval-results";
      result: BenchmarkResult;
    }
  | {
      id: typeof GPQA_DETAILS_TAB_ID;
      kind: "gpqa-details";
    }
  | {
      id: typeof GPQA_DATASET_TAB_ID;
      kind: "gpqa-dataset";
    }
  | {
      id: typeof HUMANEVAL_DETAILS_TAB_ID;
      kind: "humaneval-details";
    }
  | {
      id: typeof TERMINAL_BENCH_DETAILS_TAB_ID;
      kind: "terminal-bench-details";
    };

export function layerEditorTab(layerIndex: number, label?: string): EditorTab {
  return {
    id: `layer:${layerIndex}`,
    kind: "layer",
    layerIndex,
    label,
  };
}

export function gpqaDetailsEditorTab(): EditorTab {
  return {
    id: GPQA_DETAILS_TAB_ID,
    kind: "gpqa-details",
  };
}

export function gpqaDatasetEditorTab(): EditorTab {
  return {
    id: GPQA_DATASET_TAB_ID,
    kind: "gpqa-dataset",
  };
}

export function humanevalDetailsEditorTab(): EditorTab {
  return {
    id: HUMANEVAL_DETAILS_TAB_ID,
    kind: "humaneval-details",
  };
}

export function terminalBenchDetailsEditorTab(): EditorTab {
  return {
    id: TERMINAL_BENCH_DETAILS_TAB_ID,
    kind: "terminal-bench-details",
  };
}

export function evalResultsEditorTab(result: BenchmarkResult): EditorTab {
  return {
    id: `eval-results:${Date.now()}:${Math.random().toString(36).slice(2)}`,
    kind: "eval-results",
    result,
  };
}

export function editorTabLabel(tab: EditorTab): string {
  if (tab.kind === "eval-results") return "Eval Results";
  if (tab.kind === "gpqa-details") return "GPQA Diamond";
  if (tab.kind === "gpqa-dataset") return "GPQA Diamond Dataset";
  if (tab.kind === "humaneval-details") return "HumanEval";
  if (tab.kind === "terminal-bench-details") return "Terminal-Bench 2.1";
  if (tab.label) return tab.label;
  if (tab.layerIndex < 0) return "Global tensors";
  return `Layer ${tab.layerIndex}`;
}
