import { Fragment } from "react";
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

function formatSignedNumber(value: number, digits = 3): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

function formatMetricValue(value: number): string {
  return value.toFixed(3);
}

function formatMetricDelta(value: number | null): string {
  if (value === null) return "-";
  return formatSignedNumber(value, 3);
}

function formatStdErr(value: number, sampleCount: number): string {
  if (sampleCount <= 0) return "-";
  const variance = Math.max(0, (value * (1 - value)) / sampleCount);
  return Math.sqrt(variance).toFixed(3);
}

function choiceLabel(index: number): string {
  return String.fromCharCode(65 + index);
}

function metricForTask(task: string): string {
  if (task.startsWith("mmlu_") || task === "truthfulqa_mc1") {
    return "acc";
  }
  return "acc_norm";
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

function RuntimeSection({ result }: { result: BenchmarkResult }) {
  const baseline = result.baselineBenchmark;
  const recipe = recipeAsRuntime(result);
  const recipeDiskMb =
    result.convertedBytesAfter > 0
      ? result.convertedBytesAfter / (1024 * 1024)
      : result.diskSizeMb;

  const singleRows = [
    ["Prompt eval", formatTps(recipe.promptEvalTps, 0)],
    ["Token gen", formatTps(recipe.tokenGenTps)],
    ["TTFT", `${recipe.ttftMs.toFixed(0)} ms`],
    ["Load", formatSeconds(recipe.loadMs)],
    ["Total elapsed", formatSeconds(recipe.elapsedMs)],
    ["Tensors", recipe.modelTensorCount?.toString() ?? "-"],
    ["Peak alloc", `${recipe.vramPeakMb.toFixed(0)} MB`],
    ["Working set", `${recipe.vramAllocatedMb.toFixed(0)} MB`],
    ["Disk size", `${recipeDiskMb.toFixed(0)} MB`],
  ];

  const compareRows = baseline
    ? [
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
      ]
    : [];

  return (
    <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
      <h3 className="font-semibold text-text-muted uppercase tracking-wider mb-2">
        {baseline ? "Runtime Compare" : "Runtime"}
      </h3>
      {baseline ? (
        <div className="grid grid-cols-[1fr_96px_96px] gap-x-4 gap-y-1">
          <span />
          <span className="text-right text-text-muted uppercase tracking-wider">
            Baseline
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Recipe
          </span>
          {compareRows.map(([label, baseValue, recipeValue]) => (
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
      ) : (
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          {singleRows.map(([label, value]) => (
            <Fragment key={label}>
              <span className="text-text-muted">{label}</span>
              <span className="text-right font-mono text-text-primary">
                {value}
              </span>
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

function ResultsTable({ result }: { result: BenchmarkResult }) {
  const standard = result.standardEval;
  if (!standard || standard.baselineAccuracy !== null) return null;

  return (
    <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
      <h3 className="font-semibold text-text-muted uppercase tracking-wider mb-2">
        Results
      </h3>

      <div className="overflow-x-auto">
        <div className="grid min-w-[560px] grid-cols-[minmax(220px,1fr)_72px_56px_72px_72px] gap-x-4 gap-y-1">
          <span className="text-text-muted uppercase tracking-wider">Task</span>
          <span className="text-text-muted uppercase tracking-wider">
            Metric
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            N-Shot
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Value
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Stderr
          </span>

          {standard.tasks.map((task) => {
            const metric = metricForTask(task.task);
            return (
              <Fragment key={task.task}>
                <span className="text-text-muted truncate" title={task.task}>
                  {task.task}
                </span>
                <span className="font-mono text-text-primary">{metric}</span>
                <span className="text-right font-mono text-text-primary">
                  0
                </span>
                <span className="text-right font-mono text-text-primary">
                  {formatMetricValue(task.recipeAccuracy)}
                </span>
                <span className="text-right font-mono text-text-primary">
                  {formatStdErr(task.recipeAccuracy, task.sampleCount)}
                </span>
              </Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function PairedCompareTable({ result }: { result: BenchmarkResult }) {
  const standard = result.standardEval;
  if (!standard || standard.baselineAccuracy === null) return null;

  return (
    <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
      <h3 className="font-semibold text-text-muted uppercase tracking-wider mb-2">
        Paired Compare
      </h3>

      <div className="overflow-x-auto">
        <div className="grid min-w-[1220px] grid-cols-[minmax(220px,1fr)_72px_56px_72px_72px_72px_160px_170px_96px_104px] gap-x-4 gap-y-1">
          <span className="text-text-muted uppercase tracking-wider">Task</span>
          <span className="text-text-muted uppercase tracking-wider">
            Metric
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            N-Shot
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Base
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Recipe
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Delta
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Lost Correct Answers
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Gained Correct Answers
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Base Stderr
          </span>
          <span className="text-right text-text-muted uppercase tracking-wider">
            Recipe Stderr
          </span>

          {standard.tasks.map((task) => (
            <Fragment key={task.task}>
              <span className="text-text-muted truncate" title={task.task}>
                {task.task}
              </span>
              <span className="font-mono text-text-primary">
                {metricForTask(task.task)}
              </span>
              <span className="text-right font-mono text-text-primary">
                0
              </span>
              <span className="text-right font-mono text-text-primary">
                {task.baselineAccuracy === null
                  ? "-"
                  : formatMetricValue(task.baselineAccuracy)}
              </span>
              <span className="text-right font-mono text-text-primary">
                {formatMetricValue(task.recipeAccuracy)}
              </span>
              <span className="text-right font-mono text-text-primary">
                {formatMetricDelta(task.accuracyDelta)}
              </span>
              <span className="text-right font-mono text-text-primary">
                {task.correctToWrongCount}
              </span>
              <span className="text-right font-mono text-text-primary">
                {task.wrongToCorrectCount}
              </span>
              <span className="text-right font-mono text-text-primary">
                {task.baselineAccuracy === null
                  ? "-"
                  : formatStdErr(task.baselineAccuracy, task.sampleCount)}
              </span>
              <span className="text-right font-mono text-text-primary">
                {formatStdErr(task.recipeAccuracy, task.sampleCount)}
              </span>
            </Fragment>
          ))}

          <span className="text-text-primary font-semibold uppercase">
            Total
          </span>
          <span className="font-mono text-text-primary">mixed</span>
          <span className="text-right font-mono text-text-primary">0</span>
          <span className="text-right font-mono text-text-primary">
            {formatMetricValue(standard.baselineAccuracy)}
          </span>
          <span className="text-right font-mono text-text-primary">
            {formatMetricValue(standard.recipeAccuracy)}
          </span>
          <span className="text-right font-mono text-text-primary">
            {formatMetricDelta(standard.accuracyDelta)}
          </span>
          <span className="text-right font-mono text-text-primary">
            {standard.correctToWrongCount}
          </span>
          <span className="text-right font-mono text-text-primary">
            {standard.wrongToCorrectCount}
          </span>
          <span className="text-right font-mono text-text-primary">-</span>
          <span className="text-right font-mono text-text-primary">-</span>
        </div>
      </div>
    </div>
  );
}

function StandardEvalSampleAudit({ result }: { result: BenchmarkResult }) {
  const audits = result.standardEval?.sampleAudits ?? [];
  if (audits.length === 0) return null;

  return (
    <div className="mx-4 mt-4 pt-3 border-t border-border-default text-xs">
      <details>
        <summary className="cursor-pointer font-semibold text-text-muted uppercase tracking-wider">
          Sample Audit ({audits.length})
        </summary>
        <div className="mt-3 space-y-4">
          {audits.map((audit) => (
            <div
              key={`${audit.sampleIndex}-${audit.task}-${audit.docId}`}
              className="border-t border-border-default pt-3 first:border-t-0 first:pt-0"
            >
              <div className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1">
                <span className="text-text-muted truncate">
                  {audit.task} / {audit.docId}
                </span>
                <span className="font-mono text-text-primary uppercase">
                  {audit.flipType.replaceAll("_", " ")}
                </span>
                <span className="text-text-muted">
                  Gold {choiceLabel(audit.goldIndex)}
                </span>
                <span className="font-mono text-text-primary">
                  {audit.baselinePredictionIndex !== null
                    ? `Base ${choiceLabel(audit.baselinePredictionIndex)} / `
                    : ""}
                  Recipe {choiceLabel(audit.recipePredictionIndex)}
                </span>
              </div>

              <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap rounded border border-border-default bg-bg-base p-2 font-mono text-[11px] text-text-muted">
                {audit.prompt}
              </pre>

              <div className="mt-2 grid grid-cols-[32px_minmax(120px,1fr)_72px_88px_88px_88px_88px] gap-x-3 gap-y-1">
                <span className="text-text-muted uppercase">#</span>
                <span className="text-text-muted uppercase">Continuation</span>
                <span className="text-right text-text-muted uppercase">
                  Denom
                </span>
                <span className="text-right text-text-muted uppercase">
                  Base NLL
                </span>
                <span className="text-right text-text-muted uppercase">
                  Base Score
                </span>
                <span className="text-right text-text-muted uppercase">
                  Recipe NLL
                </span>
                <span className="text-right text-text-muted uppercase">
                  Recipe Score
                </span>
                {audit.choices.map((choice) => (
                  <Fragment key={choice.index}>
                    <span className="font-mono text-text-primary">
                      {choiceLabel(choice.index)}
                    </span>
                    <span
                      className="font-mono text-text-primary truncate"
                      title={choice.continuation}
                    >
                      {JSON.stringify(choice.continuation)}
                    </span>
                    <span className="text-right font-mono text-text-primary">
                      {choice.denominator.toFixed(0)}
                    </span>
                    <span className="text-right font-mono text-text-primary">
                      {choice.baselineNll === null
                        ? "-"
                        : choice.baselineNll.toFixed(3)}
                    </span>
                    <span className="text-right font-mono text-text-primary">
                      {choice.baselineScore === null
                        ? "-"
                        : choice.baselineScore.toFixed(3)}
                    </span>
                    <span className="text-right font-mono text-text-primary">
                      {choice.recipeNll.toFixed(3)}
                    </span>
                    <span className="text-right font-mono text-text-primary">
                      {choice.recipeScore.toFixed(3)}
                    </span>
                  </Fragment>
                ))}
              </div>
            </div>
          ))}
        </div>
      </details>
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
    result.convertedBytesAfter > 0 ||
    result.requestedTargetCount > 0;
  const quality = result.qualityEval;
  const qualityHasBaseline =
    quality?.baselinePpl !== null && quality?.baselinePpl !== undefined;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-bg-surface border border-border-default rounded-lg shadow-2xl w-[1080px] max-w-[92vw] max-h-[90vh] overflow-y-auto">
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
            <h3 className="font-semibold text-text-muted uppercase tracking-wider mb-2">
              Summary
            </h3>
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
              {result.requestedTargetCount > 0 && (
                <>
                  <span className="text-text-muted">Verified targets</span>
                  <span className="text-right font-mono text-text-primary">
                    {result.verifiedTargetCount}/{result.requestedTargetCount}
                  </span>
                </>
              )}
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
              <span className="text-text-muted">Rolling PPL tokens</span>
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

        <ResultsTable result={result} />
        <PairedCompareTable result={result} />
        <StandardEvalSampleAudit result={result} />
        <RuntimeSection result={result} />

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
