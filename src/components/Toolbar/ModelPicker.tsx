interface ModelPickerProps {
  modelPath: string | null;
  onOpen: () => void;
  disabled: boolean;
}

export function ModelPicker({ modelPath, onOpen, disabled }: ModelPickerProps) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onOpen}
        disabled={disabled}
        className="px-3 py-1 text-sm font-medium rounded bg-accent-copper/10 text-accent-copper-bright
                   border border-accent-copper/30 hover:bg-accent-copper/20 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Open GGUF...
      </button>
      {modelPath && (
        <span className="text-xs text-text-muted font-mono truncate max-w-[200px]">
          {modelPath.split(/[\\/]/).pop()}
        </span>
      )}
    </div>
  );
}
