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
  const title = tensors.length > 0
    ? `${tensors[0].layerIndex < 0 ? 'Global tensors' : `Layer ${tensors[0].layerIndex}`} - ${tensors.length} tensors`
    : 'No layer selected';

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="shrink-0 px-4 py-3 border-b border-border-default">
          <h2 className="font-heading text-base font-semibold text-text-primary uppercase tracking-wider">
            {title}
          </h2>
        </div>
        <div className="flex-1 min-h-0 overflow-auto">
          <TensorTable tensors={tensors} assignments={assignments} onAssignQuant={onAssignQuant} />
        </div>
      </div>
      <div className="shrink-0 h-44 border-t border-border-default bg-bg-primary">
        <ProfilePanel tensors={tensors} assignments={assignments} profile={profile} />
      </div>
    </div>
  );
}
