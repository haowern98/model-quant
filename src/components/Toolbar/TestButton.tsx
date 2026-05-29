import type { RecipeTestMode } from "../../types";

interface TestButtonProps {
  mode: RecipeTestMode;
  onClick: () => void;
  disabled: boolean;
  running: boolean;
}

export function TestButton({
  mode,
  onClick,
  disabled,
  running,
}: TestButtonProps) {
  const label = mode === "compare_baseline" ? "Compare Recipe" : "Test Recipe";

  return (
    <button
      onClick={onClick}
      disabled={disabled || running}
      className={`px-3 py-1 text-sm font-semibold rounded uppercase tracking-wider transition-all
        ${
          running
            ? "bg-accent-solder/20 text-accent-solder border border-accent-solder/30 cursor-wait"
            : "bg-accent-signal/20 text-accent-signal border border-accent-signal/30 hover:bg-accent-signal/30"
        }
        disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {running ? "Testing..." : label}
    </button>
  );
}
