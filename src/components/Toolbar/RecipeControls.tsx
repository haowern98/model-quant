interface RecipeControlsProps {
  onSave: () => void;
  onLoad: () => void;
  onExport: () => void;
  disabled: boolean;
}

export function RecipeControls({ onSave, onLoad, onExport, disabled }: RecipeControlsProps) {
  const btnClass = `px-2 py-1 text-xs font-medium rounded border border-border-default text-text-secondary
    hover:text-text-primary hover:border-text-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed`;

  return (
    <div className="flex items-center gap-1">
      <button onClick={onSave} disabled={disabled} className={btnClass}>Save Recipe</button>
      <button onClick={onLoad} disabled={disabled} className={btnClass}>Load Recipe</button>
      <button onClick={onExport} disabled={disabled} className={`${btnClass} text-accent-copper border-accent-copper/30`}>
        Export GGUF
      </button>
    </div>
  );
}
