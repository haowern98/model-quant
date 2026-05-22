import { TensorTable } from './TensorTable';
import { ProfilePanel } from './ProfilePanel';
import type { QuantType, RecipeProfile, TensorInfo } from '../../types';

interface DetailPanelProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
}

export function DetailPanel({ tensors, assignments, profile, onAssignQuant }: DetailPanelProps) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1">
        <div className="px-4 py-3 border-b border-border-default">
          <h2 className="font-heading text-base font-semibold text-text-primary uppercase tracking-wider">
            {tensors.length > 0
              ? `Layer ${tensors[0].layerIndex} — ${tensors.length} tensors`
              : 'No layer selected'}
          </h2>
        </div>
        <TensorTable tensors={tensors} assignments={assignments} onAssignQuant={onAssignQuant} />
      </div>
      <div className="border-t border-border-default">
        <ProfilePanel tensors={tensors} assignments={assignments} profile={profile} />
      </div>
    </div>
  );
}
