import type { QuantType, RecipeProfile, TensorInfo } from "../../types";
import { QUANT_TYPES, toTargetQuant } from "../../types";
import { estQuantSize, formatBytes } from "../../lib/format";

interface BottomPanelProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
}

export function BottomPanel({ tensors, assignments, profile }: BottomPanelProps) {
  const totalTargetBytes = tensors.reduce((sum, tensor) => {
    const quant = assignments[tensor.name] ?? toTargetQuant(tensor.currentQuant);
    const bits = QUANT_TYPES.find((item) => item.value === quant)?.bitsPerWeight ?? 4.5;
    return sum + estQuantSize(tensor.shape, bits);
  }, 0);
  const f16Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 16), 0);
  const q8Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 8), 0);
  const q4Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 4.8), 0);

  return (
    <section className="bottom-panel" aria-label="Bottom panel">
      <div className="bottom-tabs" role="tablist">
        <button type="button" role="tab" className="active" aria-label="SIZE PROFILE">
          SIZE PROFILE
        </button>
        <button type="button" role="tab" aria-label="EVAL RESULTS">
          EVAL RESULTS
        </button>
        <button type="button" role="tab" aria-label="SAMPLE AUDIT">
          SAMPLE AUDIT
        </button>
        <button type="button" role="tab" aria-label="OUTPUT">
          OUTPUT
        </button>
      </div>
      <div className="bottom-content">
        <Metric label="FP16" value={formatBytes(f16Size)} />
        <Metric label="Q8_0" value={formatBytes(q8Size)} />
        <Metric label="Recipe" value={formatBytes(totalTargetBytes)} accent />
        <Metric label="Q4_K_M" value={formatBytes(q4Size)} />
        <div className="bottom-note">
          {profile
            ? `Profiled VRAM estimate ${formatBytes(profile.vramEstimate * 1024 * 1024)}.`
            : "Ready. Quick/Default and Single/Compare are run configuration controls for the current recipe."}
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`bottom-metric ${accent ? "accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
