import { expect, test } from "@playwright/test";

test("uses VS Code codicons in the activity bar", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByLabel("Explorer view").locator(".codicon-files")).toBeVisible();
  await expect(page.getByLabel("Chat with model").locator(".codicon-comment-discussion")).toBeVisible();
  await expect(page.getByLabel("Server mode").locator(".codicon-server-process")).toBeVisible();
  await expect(page.getByLabel("Settings").locator(".codicon-settings-gear")).toBeVisible();
});
