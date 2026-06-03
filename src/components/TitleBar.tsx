import {
  closeWindow,
  minimizeWindow,
  toggleMaximizeWindow,
} from "../lib/window-controls";

interface TitleBarProps {
  modelPath: string | null;
  onOpenModel: () => void;
}

const MENU_ITEMS = [
  "File",
  "Edit",
  "Selection",
  "View",
  "Go",
  "Run",
  "Terminal",
  "Help",
];

function modelLabel(modelPath: string | null): string {
  if (!modelPath) return "Open GGUF...";
  return modelPath.split(/[\\/]/).pop() ?? modelPath;
}

export function TitleBar({ modelPath, onOpenModel }: TitleBarProps) {
  return (
    <header className="titlebar">
      <div className="app-mark" aria-label="Model Surgery">
        <span className="vscode-mark" aria-hidden="true" />
      </div>

      <nav className="menu" aria-label="Application menu">
        {MENU_ITEMS.map((item) => (
          <button key={item} type="button" aria-label={item}>
            {item}
          </button>
        ))}
      </nav>

      <div className="nav-buttons" aria-label="Navigation">
        <button type="button" aria-label="Back">
          <span className="codicon arrow-left" aria-hidden="true" />
        </button>
        <button type="button" aria-label="Forward">
          <span className="codicon arrow-right" aria-hidden="true" />
        </button>
      </div>

      <button
        type="button"
        className="command-center"
        aria-label="Model Surgery command center"
        onClick={onOpenModel}
      >
        <span className="command-title">{modelLabel(modelPath)}</span>
      </button>

      <div className="title-drag-fill" data-tauri-drag-region />

      <div className="title-actions" aria-label="Window controls">
        <button type="button" aria-label="Split layout">
          <span className="codicon split" aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label="Minimize"
          onClick={() => void minimizeWindow()}
        >
          <span className="window-min" aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label="Maximize"
          onClick={() => void toggleMaximizeWindow()}
        >
          <span className="window-max" aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label="Close"
          onClick={() => void closeWindow()}
        >
          <span className="window-close" aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
