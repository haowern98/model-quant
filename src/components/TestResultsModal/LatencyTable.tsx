import type { BenchmarkResult } from "../../types";

interface LatencyTableProps {
  result: BenchmarkResult;
}

export function LatencyTable({ result }: LatencyTableProps) {
  const runtimeElapsedMs =
    result.loadMs + result.promptEvalMs + result.generationMs;

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
        Inference
      </h3>
      <div className="space-y-1 text-sm">
        <div className="flex justify-between">
          <span className="text-text-muted">Prompt eval</span>
          <span className="font-mono text-text-primary">
            {result.promptEvalTps.toFixed(0)} t/s
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Token gen</span>
          <span className="font-mono text-text-primary">
            {result.tokenGenTps.toFixed(1)} t/s
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">TTFT</span>
          <span className="font-mono text-text-primary">
            {result.ttftMs.toFixed(0)} ms
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Load</span>
          <span className="font-mono text-text-primary">
            {(result.loadMs / 1000).toFixed(1)}s
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Total elapsed</span>
          <span className="font-mono text-text-primary">
            {(runtimeElapsedMs / 1000).toFixed(1)}s
          </span>
        </div>
        {result.modelTensorCount !== null && (
          <div className="flex justify-between">
            <span className="text-text-muted">Tensors</span>
            <span className="font-mono text-text-primary">
              {result.modelTensorCount}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
