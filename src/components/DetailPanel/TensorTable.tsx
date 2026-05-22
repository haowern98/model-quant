import { type QuantType, QUANT_TYPES } from '../../types';
import { formatBytes, estQuantSize } from '../../lib/format';
import type { TensorInfo } from '../../types';

interface TensorTableProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
}

export function TensorTable({ tensors, assignments, onAssignQuant }: TensorTableProps) {
  if (tensors.length === 0) {
    return <div className="text-text-muted text-sm p-4">Select a layer to view tensors</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border-default text-text-muted text-xs uppercase tracking-wider">
            <th className="text-left px-4 py-2 font-medium">Tensor</th>
            <th className="text-left px-4 py-2 font-medium">Shape</th>
            <th className="text-left px-4 py-2 font-medium">Quant</th>
            <th className="text-right px-4 py-2 font-medium">Current</th>
            <th className="text-right px-4 py-2 font-medium">Target</th>
          </tr>
        </thead>
        <tbody>
          {tensors.map(t => {
            const assignedQuant = assignments[t.name] ?? t.currentQuant;
            const currentSize = t.sizeBytes;
            const targetSize = estQuantSize(t.shape, QUANT_TYPES.find(q => q.value === assignedQuant)!.bitsPerWeight);
            return (
              <tr key={t.name} className="border-b border-border-default/50 hover:bg-bg-surface-alt/50">
                <td className="px-4 py-2 font-mono text-text-primary text-xs">{t.name.split('.').pop()}</td>
                <td className="px-4 py-2 font-mono text-text-muted text-xs">[{t.shape.join(', ')}]</td>
                <td className="px-4 py-2">
                  <select
                    value={assignedQuant}
                    onChange={e => onAssignQuant(t.name, e.target.value as QuantType)}
                    className="bg-bg-surface-alt border border-border-default rounded px-1 py-0.5 text-xs text-text-primary
                               focus:outline-none focus:border-accent-copper font-mono"
                  >
                    {QUANT_TYPES.map(q => (
                      <option key={q.value} value={q.value}>{q.label}</option>
                    ))}
                  </select>
                </td>
                <td className="px-4 py-2 text-right font-mono text-text-muted text-xs">{formatBytes(currentSize)}</td>
                <td className="px-4 py-2 text-right font-mono text-xs text-accent-solder">{formatBytes(targetSize)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
