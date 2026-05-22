import { QUANT_TYPES, type QuantType } from '../../types';

interface QuantPresetMenuProps {
  onSetAll: (qt: QuantType) => void;
  disabled: boolean;
}

export function QuantPresetMenu({ onSetAll, disabled }: QuantPresetMenuProps) {
  return (
    <select
      onChange={e => { if (e.target.value) onSetAll(e.target.value as QuantType); e.target.value = ''; }}
      disabled={disabled}
      className="bg-bg-surface-alt border border-border-default rounded px-2 py-1 text-sm text-text-primary
                 focus:outline-none focus:border-accent-copper disabled:opacity-40"
    >
      <option value="">Set all to...</option>
      {QUANT_TYPES.map(q => (
        <option key={q.value} value={q.value}>{q.label}</option>
      ))}
    </select>
  );
}
