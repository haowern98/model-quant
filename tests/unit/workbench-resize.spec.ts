import { expect, test } from "@playwright/test";

async function loadModel(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Model Surgery command center" }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "Qwen3.5-9B-Q4_K_M-with-a-very-long-model-name.gguf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
}

test("resizes the Explorer while keeping counts and model actions visible", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  await loadModel(page);

  const explorer = page.getByRole("complementary", { name: "Explorer" });
  const handle = page.getByRole("separator", { name: "Resize Explorer" });
  const before = await explorer.boundingBox();
  const handleBox = await handle.boundingBox();

  expect(before).not.toBeNull();
  expect(handleBox).not.toBeNull();

  await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(205, handleBox!.y + 120);
  await page.mouse.up();

  const after = await explorer.boundingBox();
  expect(after).not.toBeNull();
  expect(after!.width).toBeLessThan(before!.width);
  await expect(page.getByRole("button", { name: "Model actions" })).toBeVisible();
  await expect(page.locator(".layer-row .tree-count").first()).toBeVisible();

  const filename = page.locator(".explorer-section-toggle span:last-child");
  await expect(filename).toHaveCSS("text-overflow", "ellipsis");
});

test("resizes the bottom panel vertically", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");

  const panel = page.getByRole("region", { name: "Bottom panel" });
  const handle = page.getByRole("separator", { name: "Resize bottom panel" });
  const before = await panel.boundingBox();
  const handleBox = await handle.boundingBox();

  expect(before).not.toBeNull();
  expect(handleBox).not.toBeNull();

  await page.mouse.move(handleBox!.x + 200, handleBox!.y + handleBox!.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox!.x + 200, handleBox!.y - 100);
  await page.mouse.up();

  const after = await panel.boundingBox();
  expect(after).not.toBeNull();
  expect(after!.height).toBeGreaterThan(before!.height);
});
