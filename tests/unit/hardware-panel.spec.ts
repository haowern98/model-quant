import { expect, test } from "@playwright/test";

test("shows live hardware telemetry in the bottom panel", async ({ page }) => {
  await page.goto("/");

  const hardwareTab = page.getByRole("tab", { name: "HARDWARE" });
  await expect(hardwareTab.locator(".codicon-pulse")).toBeVisible();
  await hardwareTab.click();

  await expect(page.getByText("LIVE RESOURCES")).toBeVisible();
  await expect(page.getByText("THERMALS & POWER")).toBeVisible();
  await expect(page.getByText("Mock NVIDIA GPU")).toBeVisible();
  await expect(page.getByText("8.00 GB / 24.00 GB")).toBeVisible();
  await expect(page.getByText("GPU temperature")).toBeVisible();
  await expect(page.getByText("62 C")).toBeVisible();
  await expect(page.getByText("CPU package power")).toBeVisible();
  await expect(page.getByText("Unavailable")).toHaveCount(2);

  const panel = await page.locator(".hardware-panel").evaluate((element) => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
  }));
  expect(panel.scrollHeight).toBeLessThanOrEqual(panel.clientHeight);
});
