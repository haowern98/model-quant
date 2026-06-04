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

test("overlays the Explorer resize handle without reserving a layout gap", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");

  const explorer = page.getByRole("complementary", { name: "Explorer" });
  const handle = page.getByRole("separator", { name: "Resize Explorer" });
  const editor = page.locator(".editor-pane");
  const explorerBox = await explorer.boundingBox();
  const handleBox = await handle.boundingBox();
  const editorBox = await editor.boundingBox();
  const indicatorWidth = await handle.evaluate(
    (element) => getComputedStyle(element, "::after").width,
  );

  expect(explorerBox).not.toBeNull();
  expect(handleBox).not.toBeNull();
  expect(editorBox).not.toBeNull();

  const explorerRight = explorerBox!.x + explorerBox!.width;
  const handleCenter = handleBox!.x + handleBox!.width / 2;

  expect(editorBox!.x).toBe(explorerRight);
  expect(handleBox!.width).toBe(4);
  expect(handleCenter).toBe(explorerRight);
  expect(indicatorWidth).toBe("2px");
});

test("collapses Explorer past its minimum and restores it by dragging the divider", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");

  const explorer = page.getByRole("complementary", { name: "Explorer" });
  const handle = page.getByRole("separator", { name: "Resize Explorer" });
  const handleBox = await handle.boundingBox();

  expect(handleBox).not.toBeNull();

  await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(120, handleBox!.y + 120);
  await page.mouse.up();

  await expect(explorer).not.toBeVisible();
  await expect(handle).toBeVisible();
  await expect(handle).toHaveAttribute("aria-valuenow", "0");

  const collapsedHandleBox = await handle.boundingBox();
  expect(collapsedHandleBox).not.toBeNull();

  await page.mouse.move(
    collapsedHandleBox!.x + collapsedHandleBox!.width / 2,
    collapsedHandleBox!.y + 120,
  );
  await page.mouse.down();
  await page.mouse.move(collapsedHandleBox!.x + 40, collapsedHandleBox!.y + 120);
  await page.mouse.up();

  await expect(explorer).toBeVisible();
  await expect(handle).toHaveAttribute("aria-valuenow", "150");
});

test("Explorer activity icon toggles the panel and restores its previous width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");

  const explorer = page.getByRole("complementary", { name: "Explorer" });
  const explorerButton = page.getByRole("button", { name: "Explorer view" });
  const handle = page.getByRole("separator", { name: "Resize Explorer" });
  const handleBox = await handle.boundingBox();

  expect(handleBox).not.toBeNull();

  await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(280, handleBox!.y + 120);
  await page.mouse.up();

  const before = await explorer.boundingBox();

  expect(before).not.toBeNull();

  await explorerButton.click();
  await expect(explorer).not.toBeVisible();
  await expect(explorerButton).toHaveAttribute("aria-pressed", "false");

  await explorerButton.click();
  await expect(explorer).toBeVisible();
  await expect(explorerButton).toHaveAttribute("aria-pressed", "true");

  const restored = await explorer.boundingBox();
  expect(restored).not.toBeNull();
  expect(restored!.width).toBe(before!.width);
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
