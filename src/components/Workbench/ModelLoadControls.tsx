export function ModelLoadControls() {
  return (
    <div className="editor-run-controls model-load-controls">
      <div className="run-split-action" role="group" aria-label="Model load controls">
        <button
          type="button"
          className="run-split-primary"
          aria-label="Load model"
          title="Load Model is not available yet"
          disabled
        >
          <span className="codicon codicon-arrow-circle-up" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="run-split-chevron"
          aria-label="Load model options"
          title="Load configuration is not available yet"
          disabled
        >
          <span className="codicon codicon-chevron-down" aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
