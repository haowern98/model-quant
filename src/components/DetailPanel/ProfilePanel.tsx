import type { QuantType, RecipeProfile } from '../../types';
import { quantBitsPerWeight } from '../../types';
import { formatBytes, estQuantSize } from '../../lib/format';
import type { TensorInfo } from '../../types';

interface ProfilePanelProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
}

export function ProfilePanel({ tensors, assignments, profile }: ProfilePanelProps) {
  const totalTargetBytes = tensors.reduce((sum, t) => {
    const qt = assignments[t.name] ?? t.currentQuant;
    const bpw = quantBitsPerWeight(qt) ?? 4.5;
    return sum + estQuantSize(t.shape, bpw);
  }, 0);

  const f16Size = tensors.reduce((sum, t) => sum + estQuantSize(t.shape, 16), 0);
  const q8Size = tensors.reduce((sum, t) => sum + estQuantSize(t.shape, 8), 0);
  const q4Size = tensors.reduce((sum, t) => sum + estQuantSize(t.shape, 4.8), 0);

  const maxSize = Math.max(f16Size, q8Size, totalTargetBytes, q4Size);
  const barPercent = (size: number) => `${maxSize > 0 ? Math.round((size / maxSize) * 100) : 0}%`;

  return (
    <div className="h-full p-3 space-y-3 overflow-hidden">
      <h3 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wider">Size Profile</h3>

      <div className="space-y-1.5">
        {[
          { label: 'FP16', size: f16Size, color: 'bg-text-muted' },
          { label: 'Q8_0', size: q8Size, color: 'bg-text-secondary' },
          { label: 'Recipe', size: totalTargetBytes, color: 'bg-accent-copper' },
          { label: 'Q4_K_M', size: q4Size, color: 'bg-accent-signal' },
        ].map(({ label, size, color }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="text-xs text-text-muted w-16 text-right font-mono">{label}</span>
            <div className="flex-1 bg-bg-surface-alt rounded-full h-2.5 overflow-hidden">
              <div
                className={`h-full rounded-full ${color} transition-all duration-300`}
                style={{ width: barPercent(size) }}
              />
            </div>
            <span className="text-xs text-text-secondary w-20 font-mono">{formatBytes(size)}</span>
          </div>
        ))}
      </div>

      {profile && (
        <div className="bg-bg-surface-alt rounded p-2 space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-text-muted">VRAM estimate</span>
            <span className="text-text-primary font-mono">{formatBytes(profile.vramEstimate * 1024 * 1024)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted">Saved vs Q8</span>
            <span className="text-accent-signal font-mono">{formatBytes(profile.sizeSavedVsQ8 * 1024 * 1024)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
