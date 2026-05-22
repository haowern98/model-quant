import type { TensorInfo } from '../../types';

interface LayerTreeProps {
  tensors: TensorInfo[];
  selectedLayerIndex: number | null;
  onSelectLayer: (index: number) => void;
  filterText: string;
}

export function LayerTree({ tensors, selectedLayerIndex, onSelectLayer, filterText }: LayerTreeProps) {
  const groups = new Map<number, TensorInfo[]>();
  for (const t of tensors) {
    const key = t.layerIndex;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(t);
  }

  const sorted = [...groups.entries()].sort(([a], [b]) => a - b);

  const groupLabel = (layerIndex: number): string => {
    if (layerIndex === -2) return 'Output';
    if (layerIndex === -1) return 'Embedding';
    return `Layer ${layerIndex}`;
  };

  const filtered = sorted.filter(([idx, ts]) => {
    if (!filterText) return true;
    const label = groupLabel(idx).toLowerCase();
    return label.includes(filterText.toLowerCase()) ||
      ts.some(t => t.name.toLowerCase().includes(filterText.toLowerCase()));
  });

  return (
    <div className="flex-1 overflow-y-auto px-2 py-1">
      {filtered.map(([layerIndex, ts]) => (
        <div key={layerIndex} className="mb-0.5">
          <button
            onClick={() => onSelectLayer(layerIndex)}
            className={`w-full text-left px-2 py-1 rounded text-sm font-medium transition-colors
              ${selectedLayerIndex === layerIndex
                ? 'bg-accent-copper/20 text-accent-copper-bright'
                : 'text-text-secondary hover:bg-bg-surface-alt hover:text-text-primary'
              }`}
          >
            {groupLabel(layerIndex)}
            <span className="text-text-muted text-xs ml-2">({ts.length} tensors)</span>
          </button>
          {selectedLayerIndex === layerIndex && ts.map(t => (
            <div key={t.name} className="ml-4 px-2 py-0.5 text-xs text-text-muted font-mono truncate">
              {t.name.split('.').pop()}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
