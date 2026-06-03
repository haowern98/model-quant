import { isTauri } from "@tauri-apps/api/core";

async function currentWindow() {
  if (!isTauri()) return null;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow();
}

export async function minimizeWindow(): Promise<void> {
  const win = await currentWindow();
  await win?.minimize();
}

export async function toggleMaximizeWindow(): Promise<void> {
  const win = await currentWindow();
  if (!win) return;

  if (await win.isMaximized()) {
    await win.unmaximize();
  } else {
    await win.maximize();
  }
}

export async function closeWindow(): Promise<void> {
  const win = await currentWindow();
  await win?.close();
}
