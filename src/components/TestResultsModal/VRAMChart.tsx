import type { BenchmarkResult } from '../../types';

interface VRAMChartProps {
  result: BenchmarkResult;
}

export function VRAMChart({ result }: VRAMChartProps) {
  const max = Math.max(result.vramPeakMb, result.vramAllocatedMb);
  const peakPct = `${Math.round((result.vramPeakMb / max) * 100)}%`;
  const allocPct = `${Math.round((result.vramAllocatedMb / max) * 100)}%`;

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">VRAM</h3>
      <div className="space-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-text-muted w-20">Peak alloc</span>
          <div className="flex-1 bg-bg-surface-alt rounded-full h-3 overflow-hidden">
            <div className="h-full bg-accent-solder rounded-full" style={{ width: peakPct }} />
          </div>
          <span className="font-mono text-text-primary w-20 text-right">{(result.vramPeakMb / 1024).toFixed(2)} GB</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-text-muted w-20">Working set</span>
          <div className="flex-1 bg-bg-surface-alt rounded-full h-3 overflow-hidden">
            <div className="h-full bg-accent-copper rounded-full" style={{ width: allocPct }} />
          </div>
          <span className="font-mono text-text-primary w-20 text-right">{(result.vramAllocatedMb / 1024).toFixed(2)} GB</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Disk size</span>
          <span className="font-mono text-text-primary">{(result.diskSizeMb / 1024).toFixed(2)} GB</span>
        </div>
      </div>
    </div>
  );
}
