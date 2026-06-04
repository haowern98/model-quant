import { expect, test } from "@playwright/test";

async function loadModel(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Model Surgery command center" }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "mock.gguf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
}

test.describe("Eval Results editor", () => {
  test("opens completed test results in one reusable editor tab", async ({ page }) => {
    await page.goto("/");
    await loadModel(page);
    await page.getByRole("button", { name: /^Layer 0 / }).click();

    await page.getByRole("button", { name: "Run recipe test" }).click();

    const editorTabs = page.getByRole("tablist", { name: "Open layers" });
    const evalResultsTab = editorTabs.getByRole("tab", { name: "Eval Results" });

    await expect(evalResultsTab).toBeVisible();
    await expect(evalResultsTab).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(editorTabs.getByRole("tab", { name: "Layer 0" })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Benchmark Results" }),
    ).toBeVisible();
    await expect(page.locator(".eval-results-editor")).toHaveCSS(
      "overflow-y",
      "auto",
    );
    await expect(page.getByText("Verified targets")).toBeVisible();
    await expect(page.getByText("0/0", { exact: true })).toBeVisible();
    await expect(page.locator(".fixed.inset-0")).toHaveCount(0);

    await page.getByRole("button", { name: "Run recipe test" }).click();
    await expect(evalResultsTab).toHaveCount(1);
  });
});
