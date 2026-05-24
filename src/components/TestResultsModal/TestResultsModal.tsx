import { LatencyTable } from './LatencyTable';
import { VRAMChart } from './VRAMChart';
import { SaveOrDiscard } from './SaveOrDiscard';
import type { BenchmarkResult } from '../../types';

interface TestResultsModalProps {
  result: BenchmarkResult | null;
  onSave: () => void;
  onExport: () => void;
  onDiscard: () => void;
}

export function TestResultsModal({ result, onSave, onExport, onDiscard }: TestResultsModalProps) {
  if (!result) return null;

  const isNativeSmoke = result.testMode === 'native_runtime_smoke';
  const isNativeBaseline = result.testMode === 'native_baseline';
  const passed = isNativeSmoke || isNativeBaseline || result.tokenGenTps > 0;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-bg-surface border border-border-default rounded-lg shadow-2xl w-[480px] overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-default">
          <h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wider">
            Benchmark Results
          </h2>
          <span className={`text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
            passed ? 'bg-accent-signal/20 text-accent-signal' : 'bg-accent-solder/20 text-accent-solder'
          }`}>
            {isNativeSmoke || isNativeBaseline ? 'NATIVE OK' : passed ? 'PASS' : 'FAIL'} {result.elapsedMs / 1000}s
          </span>
        </div>

        <div className="px-4 pt-4 text-xs text-text-muted">
          <p>{result.statusMessage}</p>
          {result.nativeRuntime && (
            <p className="mt-2 font-mono break-words">{result.nativeRuntime}</p>
          )}
        </div>

        <div className="p-4 grid grid-cols-2 gap-6">
          <LatencyTable result={result} />
          <VRAMChart result={result} />
        </div>

        <div className="px-4 pb-4">
          <SaveOrDiscard onSave={onSave} onExport={onExport} onDiscard={onDiscard} />
        </div>
      </div>
    </div>
  );
}
