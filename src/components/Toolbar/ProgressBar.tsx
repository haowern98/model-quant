import type { ProgressEvent } from '../../types';

const STAGE_LABELS: Record<string, string> = {
  installing: 'Installing eval backend...',
  requantizing: 'Requantizing tensors...',
  writing: 'Writing GGUF to disk...',
  loading: 'Loading model into VRAM...',
  benchmarking: 'Running benchmark...',
};

interface ProgressBarProps {
  progress: ProgressEvent | null;
}

export function ProgressBar({ progress }: ProgressBarProps) {
  if (!progress) return null;

  return (
    <div className="flex items-center gap-2 min-w-[240px]">
      <div className="flex-1 bg-bg-surface-alt rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-accent-copper rounded-full transition-all duration-300"
          style={{ width: `${Math.round(progress.percent * 100)}%` }}
        />
      </div>
      <span className="text-xs text-text-muted whitespace-nowrap">
        {STAGE_LABELS[progress.stage] ?? progress.message}
      </span>
    </div>
  );
}
