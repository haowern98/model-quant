import { useEffect, useState } from "react";
import { getHardwareSnapshot } from "../../lib/tauri-bridge";
import { formatBytes } from "../../lib/format";
import type { HardwareSnapshot } from "../../types";

const unavailable = "Unavailable";

function percent(value: number | null): string {
  return value === null ? unavailable : `${Math.round(value)}%`;
}

function metric(value: number | null, unit: string): string {
  return value === null ? unavailable : `${Math.round(value)} ${unit}`;
}

function memory(usedBytes: number | null, totalBytes: number | null): string {
  if (usedBytes === null || totalBytes === null || totalBytes <= 0) return unavailable;
  return `${formatBytes(usedBytes)} / ${formatBytes(totalBytes)}`;
}

export function HardwarePanel() {
  const [snapshot, setSnapshot] = useState<HardwareSnapshot | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const next = await getHardwareSnapshot();
        if (active) setSnapshot(next);
      } catch {
        if (active) setSnapshot(null);
      }
    };

    void refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const gpuUsage = snapshot?.gpuUsagePercent ?? null;
  const cpuUsage = snapshot?.cpuUsagePercent ?? null;
  const ramUsed = snapshot?.ramUsedBytes ?? null;
  const ramTotal = snapshot?.ramTotalBytes ?? null;
  const ramUsage = ramUsed !== null && ramTotal ? (ramUsed / ramTotal) * 100 : null;
  const vramUsed = snapshot?.vramUsedMb ?? null;
  const vramTotal = snapshot?.vramTotalMb ?? null;
  const vramUsage = vramUsed !== null && vramTotal ? (vramUsed / vramTotal) * 100 : null;

  return (
    <div className="hardware-panel">
      <HardwareSection title="LIVE RESOURCES">
        <ResourceRow
          icon="chip"
          label="GPU"
          device={snapshot?.gpuName ?? unavailable}
          usage={gpuUsage}
          value={percent(gpuUsage)}
        />
        <ResourceRow
          icon="database"
          label="VRAM"
          device={memory(
            vramUsed === null ? null : vramUsed * 1024 * 1024,
            vramTotal === null ? null : vramTotal * 1024 * 1024,
          )}
          usage={vramUsage}
          value={percent(vramUsage)}
        />
        <ResourceRow
          icon="server-process"
          label="CPU"
          device={snapshot?.cpuName ?? unavailable}
          usage={cpuUsage}
          value={percent(cpuUsage)}
        />
        <ResourceRow
          icon="vm"
          label="RAM"
          device={memory(ramUsed, ramTotal)}
          usage={ramUsage}
          value={percent(ramUsage)}
        />
      </HardwareSection>

      <HardwareSection title="THERMALS & POWER" secondary>
        <DetailRow icon="flame" label="GPU temperature" value={metric(snapshot?.gpuTemperatureC ?? null, "C")} />
        <DetailRow icon="zap" label="GPU board power" value={metric(snapshot?.gpuPowerW ?? null, "W")} />
        <DetailRow icon="flame" label="CPU temperature" value={metric(snapshot?.cpuTemperatureC ?? null, "C")} />
        <DetailRow icon="zap" label="CPU package power" value={metric(snapshot?.cpuPowerW ?? null, "W")} />
      </HardwareSection>
    </div>
  );
}

function HardwareSection({
  title,
  secondary = false,
  children,
}: {
  title: string;
  secondary?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`hardware-section ${secondary ? "secondary" : ""}`}>
      <div className="hardware-section-heading">{title}</div>
      <div className="hardware-list">{children}</div>
    </section>
  );
}

function ResourceRow({
  icon,
  label,
  device,
  usage,
  value,
}: {
  icon: string;
  label: string;
  device: string;
  usage: number | null;
  value: string;
}) {
  const width = Math.min(100, Math.max(0, usage ?? 0));
  return (
    <div className="hardware-resource">
      <span className={`codicon codicon-${icon}`} aria-hidden="true" />
      <span className="hardware-label">{label}</span>
      <span className="hardware-device" title={device}>{device}</span>
      <span className="hardware-bar" aria-hidden="true"><span style={{ width: `${width}%` }} /></span>
      <strong>{value}</strong>
    </div>
  );
}

function DetailRow({ icon, label, value }: { icon: string; label: string; value: string }) {
  return (
    <div className="hardware-detail">
      <span className={`codicon codicon-${icon}`} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
