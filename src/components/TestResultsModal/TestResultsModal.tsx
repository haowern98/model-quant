import { Fragment } from "react";
import { LatencyTable } from "./LatencyTable";
import { VRAMChart } from "./VRAMChart";
import { SaveOrDiscard } from "./SaveOrDiscard";
import type { BenchmarkResult, RuntimeBenchmark } from "../../types";

interface TestResultsModalProps {
  result: BenchmarkResult | null;
  onSave: () => void;
  onExport: () => void;
  onDiscard: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }
  return `${bytes} B`;
}

function driftLabel(deltaPercent: number): string {
  const drift = Math.abs(deltaPercent);
  if (drift <= 1) return "LOW DRIFT";
  if (drift <= 3) return "MODERATE DRIFT";
  return "HIGH DRIFT";
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTps(value: number, digits = 1): string {
  return `${value.toFixed(digits)} t/s`;
}

function runtimeElapsed(result: BenchmarkResult): number {
  return result.loadMs + result.promptEvalMs + result.generationMs;
}

function recipeAsRuntime(result: BenchmarkResult): RuntimeBenchmark {
  return {
    promptEvalTps: result.promptEvalTps,
    tokenGenTps: result.tokenGenTps,
    ttftMs: result.ttftMs,
    promptEvalMs: result.promptEvalMs,
    generationMs: result.generationMs,
    vramPeakMb: result.vramPeakMb,
    vramAllocatedMb: result.vramAllocatedMb,
    loadMs: result.loadMs,
    elapsedMs: runtimeElapsed(result),
    modelTensorCount: result.modelTensorCount,
  };
}

function RuntimeComparison({ result }: { result: BenchmarkResult }) {
  if (!result.baselineBenchmark) return null;

  const baseline = result.baselineBenchmark;
  const recipe = recipeAsRuntime(result);
  const recipeDiskMb =
    result.convertedBytesAfter > 0
      ? result.convertedBytesAfter / (1024 * 1024)
      : result.diskSizeMb;

  const rows = [
    [
      "Prompt eval",
      formatTps(baseline.promptEvalTps, 0),
      formatTps(recipe.promptEvalTps, 0),
    ],
    [
      "Token gen",
      formatTps(baseline.tokenGenTps),
      formatTps(recipe.tokenGenTps),
    ],
    [
      "TTFT",
      `${baseline.ttftMs.toFixed(0)} ms`,
      `${recipe.ttftMs.toFixed(0)} ms`,
    ],
    ["Load", formatSeconds(baseline.loadMs), formatSeconds(recipe.loadMs)],
    [
      "Total elapsed",
      formatSeconds(baseline.elapsedMs),
      formatSeconds(recipe.elapsedMs),
    ],
    [
      "Tensors",
      baseline.modelTensorCount?.toString() ?? "-",
      recipe.modelTensorCount?.toString() ?? "-",
    ],
    [
      "Peak alloc",
      `${baseline.vramPeakMb.toFixed(0)} MB`,
      `${recipe.vramPeakMb.toFixed(0)} MB`,
    ],
    [
      "Working set",
      `${baseline.vramAllocatedMb.toFixed(0)} MB`,
      `${recipe.vramAllocatedMb.toFixed(0)} MB`,
    ],
    [
      "Disk size",
      `${result.diskSizeMb.toFixed(0)} MB`,
      `${recipeDiskMb.toFixed(0)} MB`,
    ],
  ];

  return (
    <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
      <h3 className="font-semibold text-text-muted uppercase tracking-wider mb-2">
        Runtime Compare
      </h3>
      <div className="grid grid-cols-[1fr_96px_96px] gap-x-4 gap-y-1">
        <span />
        <span className="text-right text-text-muted uppercase tracking-wider">
          Baseline
        </span>
        <span className="text-right text-text-muted uppercase tracking-wider">
          Recipe
        </span>
        {rows.map(([label, baseValue, recipeValue]) => (
          <Fragment key={label}>
            <span className="text-text-muted">{label}</span>
            <span className="text-right font-mono text-text-primary">
              {baseValue}
            </span>
            <span className="text-right font-mono text-text-primary">
              {recipeValue}
            </span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export function TestResultsModal({
  result,
  onSave,
  onExport,
  onDiscard,
}: TestResultsModalProps) {
  if (!result) return null;

  const isNativeSmoke = result.testMode === "native_runtime_smoke";
  const isNativeBaseline = result.testMode === "native_baseline";
  const passed = isNativeSmoke || isNativeBaseline || result.tokenGenTps > 0;
  const hasTensorStats =
    result.copiedTensorCount > 0 ||
    result.convertedTensorCount > 0 ||
    result.convertedBytesBefore > 0 ||
    result.convertedBytesAfter > 0;
  const quality = result.qualityEval;
  const qualityHasBaseline =
    quality?.baselinePpl !== null && quality?.baselinePpl !== undefined;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-bg-surface border border-border-default rounded-lg shadow-2xl w-[720px] max-w-[92vw] max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-default">
          <h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wider">
            Benchmark Results
          </h2>
          <span
            className={`text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
              passed
                ? "bg-accent-signal/20 text-accent-signal"
                : "bg-accent-solder/20 text-accent-solder"
            }`}
          >
            {isNativeSmoke || isNativeBaseline
              ? "NATIVE OK"
              : passed
                ? "PASS"
                : "FAIL"}{" "}
            {result.elapsedMs / 1000}s
          </span>
        </div>

        <div className="px-4 pt-4 text-xs text-text-muted">
          <p>{result.statusMessage}</p>
          {result.nativeRuntime && (
            <p className="mt-2 font-mono break-words">{result.nativeRuntime}</p>
          )}
        </div>

        {hasTensorStats && (
          <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
            <div className="grid grid-cols-2 gap-x-6 gap-y-1">
              <span className="text-text-muted">Copied tensors</span>
              <span className="text-right font-mono text-text-primary">
                {result.copiedTensorCount}
              </span>
              <span className="text-text-muted">Converted tensors</span>
              <span className="text-right font-mono text-text-primary">
                {result.convertedTensorCount}
              </span>
              <span className="text-text-muted">Converted from</span>
              <span className="text-right font-mono text-text-primary">
                {formatBytes(result.convertedBytesBefore)}
              </span>
              <span className="text-text-muted">Converted to</span>
              <span className="text-right font-mono text-text-primary">
                {formatBytes(result.convertedBytesAfter)}
              </span>
            </div>
          </div>
        )}

        {quality && (
          <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-text-muted uppercase tracking-wider">
                Quality
              </h3>
              {qualityHasBaseline && (
                <span className="font-mono text-text-primary">
                  {driftLabel(quality.pplDeltaPercent)}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1">
              {qualityHasBaseline && (
                <>
                  <span className="text-text-muted">Baseline PPL</span>
                  <span className="text-right font-mono text-text-primary">
                    {quality.baselinePpl!.toFixed(3)}
                  </span>
                </>
              )}
              <span className="text-text-muted">Recipe PPL</span>
              <span className="text-right font-mono text-text-primary">
                {quality.recipePpl.toFixed(3)}
              </span>
              <span className="text-text-muted">Recipe NLL</span>
              <span className="text-right font-mono text-text-primary">
                {quality.recipeNll.toFixed(3)}
              </span>
              {qualityHasBaseline && (
                <>
                  <span className="text-text-muted">Delta</span>
                  <span className="text-right font-mono text-text-primary">
                    {quality.pplDelta.toFixed(3)}
                  </span>
                  <span className="text-text-muted">Delta %</span>
                  <span className="text-right font-mono text-text-primary">
                    {quality.pplDeltaPercent.toFixed(2)}%
                  </span>
                </>
              )}
              <span className="text-text-muted">Eval tokens</span>
              <span className="text-right font-mono text-text-primary">
                {quality.evalTokenCount}
              </span>
              <span className="text-text-muted">Eval samples</span>
              <span className="text-right font-mono text-text-primary">
                {quality.evalSampleCount}
              </span>
            </div>
          </div>
        )}

        {result.baselineBenchmark ? (
          <RuntimeComparison result={result} />
        ) : (
          <div className="p-4 grid grid-cols-2 gap-6">
            <LatencyTable result={result} />
            <VRAMChart result={result} />
          </div>
        )}

        <div className="px-4 pb-4">
          <SaveOrDiscard
            onSave={onSave}
            onExport={onExport}
            onDiscard={onDiscard}
          />
        </div>
      </div>
    </div>
  );
}
