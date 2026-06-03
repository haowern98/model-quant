interface LayerTabsProps {
  openLayers: number[];
  activeLayerIndex: number | null;
  onSelectLayer: (layerIndex: number) => void;
  onCloseLayer: (layerIndex: number) => void;
}

export function LayerTabs({
  openLayers,
  activeLayerIndex,
  onSelectLayer,
  onCloseLayer,
}: LayerTabsProps) {
  const labelFor = (layerIndex: number) =>
    layerIndex < 0 ? "Global tensors" : `Layer ${layerIndex}`;

  return (
    <div className="layer-tabs" role="tablist" aria-label="Open layers">
      {openLayers.map((layerIndex) => (
        <button
          key={layerIndex}
          type="button"
          role="tab"
          aria-label={labelFor(layerIndex)}
          aria-selected={activeLayerIndex === layerIndex}
          className={`layer-tab ${activeLayerIndex === layerIndex ? "active" : ""}`}
          onClick={() => onSelectLayer(layerIndex)}
        >
          <span className="tab-name">{labelFor(layerIndex)}</span>
          <span
            className="tab-close"
            aria-hidden="true"
            onClick={(event) => {
              event.stopPropagation();
              onCloseLayer(layerIndex);
            }}
          />
        </button>
      ))}
    </div>
  );
}
