interface SaveOrDiscardProps {
  onSave: () => void;
  onExport: () => void;
  onDiscard: () => void;
}

export function SaveOrDiscard({ onSave, onExport, onDiscard }: SaveOrDiscardProps) {
  return (
    <div className="flex items-center justify-end gap-2 pt-3 border-t border-border-default">
      <button onClick={onDiscard}
        className="px-3 py-1 text-sm text-text-muted hover:text-text-primary transition-colors">
        Discard
      </button>
      <button onClick={onSave}
        className="px-3 py-1 text-sm rounded border border-border-default text-text-secondary
                   hover:text-text-primary hover:border-text-muted transition-colors">
        Save Recipe
      </button>
      <button onClick={onExport}
        className="px-3 py-1 text-sm font-medium rounded bg-accent-copper/10 text-accent-copper-bright
                   border border-accent-copper/30 hover:bg-accent-copper/20 transition-colors">
        Export GGUF
      </button>
    </div>
  );
}
