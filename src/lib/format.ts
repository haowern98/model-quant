export function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(0)} MB`;
  if (bytes >= 1_024) return `${(bytes / 1_024).toFixed(0)} KB`;
  return `${bytes} B`;
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

export function estQuantSize(shape: number[], bitsPerWeight: number): number {
  const elements = shape.reduce((a, b) => a * b, 1);
  return Math.round(elements * bitsPerWeight / 8);
}

export function formatTensorName(name: string): string {
  return name
    .replace(/^blk\.\d+\./, '')
    .replace(/^layers\.\d+\./, '');
}
