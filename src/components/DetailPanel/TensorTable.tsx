import { type QuantType, QUANT_TYPES, quantBitsPerWeight } from '../../types';
import { formatBytes, estQuantSize, formatTensorName } from '../../lib/format';
import type { TensorInfo } from '../../types';

interface TensorTableProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
}

export function TensorTable({ tensors, assignments, onAssignQuant }: TensorTableProps) {
  if (tensors.length === 0) {
    return <div className="tensor-empty-state">Select a layer to view tensors</div>;
  }

  return (
    <div className="tensor-table-scroll">
      <table className="tensor-table">
        <thead>
          <tr>
            <th className="row-number" aria-label="Row number"></th>
            <th>Tensor</th>
            <th>Shape</th>
            <th>Current Quant</th>
            <th>Target Quant</th>
            <th className="numeric">Current</th>
            <th className="numeric">Target</th>
          </tr>
        </thead>
        <tbody>
          {tensors.map((t, index) => {
            const assignedQuant = assignments[t.name] ?? t.currentQuant;
            const currentSize = t.sizeBytes;
            const targetBits = quantBitsPerWeight(assignedQuant) ?? 4.5;
            const targetSize = estQuantSize(t.shape, targetBits);
            const canAssignTarget = t.quantPreflight?.canQuantize ?? true;
            const disabledReason = t.quantPreflight?.blockedReason ?? 'Tensor cannot be quantized';
            const allowedTargetQuants = t.quantPreflight?.allowedTargetQuants;
            const editableOptions = allowedTargetQuants && allowedTargetQuants.length > 0
              ? QUANT_TYPES.filter(q => allowedTargetQuants.includes(q.value))
              : QUANT_TYPES;
            const hasAssignedOption = editableOptions.some(q => q.value === assignedQuant);
            const quantOptions = hasAssignedOption
              ? editableOptions
              : [
                  {
                    value: assignedQuant,
                    label: assignedQuant,
                    bitsPerWeight: targetBits,
                    quality: 'Source as-is',
                  },
                  ...editableOptions,
                ];
            const hasAlternateTarget = quantOptions.some(q => q.value !== assignedQuant);
            const targetDisabled = !canAssignTarget || !hasAlternateTarget;
            const targetDisabledReason = !canAssignTarget
              ? disabledReason
              : 'No compatible smaller target quant is available';
            const changed = assignedQuant !== t.currentQuant;
            return (
              <tr key={t.name} className={changed ? "changed-row" : undefined}>
                <td className="row-number">{index + 1}</td>
                <td className="tensor-name" title={t.name}>
                  {formatTensorName(t.name)}
                </td>
                <td className="shape">[{t.shape.join(', ')}]</td>
                <td className="quant">{t.currentQuant}</td>
                <td>
                  <select
                    value={assignedQuant}
                    onChange={e => onAssignQuant(t.name, e.target.value as QuantType)}
                    disabled={targetDisabled}
                    title={targetDisabled ? targetDisabledReason : undefined}
                    className="target-control"
                  >
                    {quantOptions.map(q => (
                      <option key={q.value} value={q.value}>{q.label}</option>
                    ))}
                  </select>
                  {targetDisabled && <span className="lock" aria-hidden="true" />}
                </td>
                <td className="numeric">{formatBytes(currentSize)}</td>
                <td className="numeric">{formatBytes(targetSize)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
