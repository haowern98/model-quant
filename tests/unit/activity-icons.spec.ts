import { expect, test } from "@playwright/test";

test("uses VS Code codicons in the activity bar", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByLabel("Explorer view").locator(".codicon-files")).toBeVisible();
  await expect(page.getByLabel("Chat with model").locator(".codicon-comment-discussion")).toBeVisible();
  await expect(page.getByLabel("Testing").locator(".codicon-beaker")).toBeVisible();
  await expect(page.getByLabel("Server mode").locator(".codicon-server-process")).toBeVisible();
  await expect(page.getByLabel("Settings").locator(".codicon-settings-gear")).toBeVisible();
});

test("switches the side bar to the Testing view", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Testing" }).click();

  const testingPanel = page.getByRole("complementary", { name: "Testing" });
  await expect(testingPanel).toBeVisible();
  await expect(testingPanel.getByText("MODEL EVALUATION", { exact: true })).toBeVisible();

  await expect(testingPanel.locator(".explorer-section-header", { hasText: "LOCAL CHECKS" })).toBeVisible();
  await expect(testingPanel.locator(".explorer-section-header", { hasText: "BENCHMARKS" })).toBeVisible();
  await expect(testingPanel.locator(".explorer-section-header", { hasText: "ENVIRONMENT" })).toBeVisible();
  await expect(testingPanel.locator(".explorer-section-header", { hasText: "LATEST RUNS" })).toBeVisible();

  const gpqaRow = testingPanel.getByRole("button", { name: /GPQA Diamond/ }).first();
  await expect(gpqaRow).toBeVisible();
  await expect(gpqaRow.locator(".tree-chevron")).toBeVisible();
  await expect(gpqaRow.locator(".tree-folder-icon")).toBeVisible();
  await expect(gpqaRow.locator(".tree-count")).toHaveText(/Ready|Needs harness|Download|Install/);
  await expect(testingPanel.getByRole("button", { name: "GPQA Diamond Details" })).toBeVisible();
  await expect(testingPanel.getByRole("button", { name: "GPQA Diamond Dataset" })).toBeVisible();
  await expect(testingPanel.getByText("Samples", { exact: true })).toHaveCount(0);
});
