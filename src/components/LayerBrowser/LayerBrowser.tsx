import { useState } from 'react';
import { SearchBar } from './SearchBar';
import { LayerTree } from './LayerTree';
import type { TensorInfo } from '../../types';

interface LayerBrowserProps {
  tensors: TensorInfo[];
  selectedLayerIndex: number | null;
  onSelectLayer: (index: number) => void;
}

export function LayerBrowser({ tensors, selectedLayerIndex, onSelectLayer }: LayerBrowserProps) {
  const [filterText, setFilterText] = useState('');

  return (
    <div className="flex flex-col h-full min-h-0">
      <SearchBar value={filterText} onChange={setFilterText} />
      <LayerTree
        tensors={tensors}
        selectedLayerIndex={selectedLayerIndex}
        onSelectLayer={onSelectLayer}
        filterText={filterText}
      />
    </div>
  );
}
