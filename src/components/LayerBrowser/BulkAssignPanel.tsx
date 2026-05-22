import { type AssignPattern, type QuantType, QUANT_TYPES } from '../../types';

const PATTERNS: { value: AssignPattern; label: string }[] = [
  { value: 'all_attn', label: 'All Attention' },
  { value: 'all_ffn', label: 'All FFN' },
  { value: 'all_embeddings', label: 'All Embeddings' },
  { value: 'all', label: 'Entire Model' },
];

interface BulkAssignPanelProps {
  onAssign: (pattern: AssignPattern, quantType: QuantType) => void;
  disabled: boolean;
}

export function BulkAssignPanel({ onAssign, disabled }: BulkAssignPanelProps) {
  const handleClick = (pattern: AssignPattern, quantType: QuantType) => {
    onAssign(pattern, quantType);
  };

  return (
    <div className="border-t border-border-default p-3 space-y-2">
      <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">Bulk Assign</h3>
      <div className="space-y-1">
        {PATTERNS.map(p => (
          <div key={p.value} className="flex items-center gap-1">
            <span className="text-xs text-text-secondary w-24 flex-shrink-0">{p.label}</span>
            <select
              onChange={e => { if (e.target.value) handleClick(p.value, e.target.value as QuantType); e.target.value = ''; }}
              disabled={disabled}
              className="flex-1 bg-bg-surface-alt border border-border-default rounded px-1 py-0.5 text-xs
                         text-text-primary focus:outline-none focus:border-accent-copper disabled:opacity-40 font-mono"
            >
              <option value="">Apply...</option>
              {QUANT_TYPES.map(q => (
                <option key={q.value} value={q.value}>{q.label}</option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}
