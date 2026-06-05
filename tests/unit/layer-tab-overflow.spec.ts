import { expect, test } from "@playwright/test";

async function loadModel(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Model Surgery command center" }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "mock.gguf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
}

test("scrolls overflowing layer tabs with the mouse wheel and shows an overlay thumb", async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 700 });
  await page.goto("/");
  await loadModel(page);

  await page.getByRole("button", { name: /^Global tensors / }).click();
  await page.getByRole("button", { name: /^Layer 0 / }).click();
  await page.getByRole("button", { name: /^Layer 1 / }).click();

  const tabs = page.getByRole("tablist", { name: "Open layers" });
  const thumb = page.locator(".layer-tabs-scroll-thumb");
  await tabs.evaluate((element) => {
    element.scrollLeft = 0;
  });
  const before = await tabs.evaluate((element) => element.scrollLeft);

  await tabs.hover();
  await page.mouse.wheel(0, 400);

  await expect.poll(() => tabs.evaluate((element) => element.scrollLeft)).toBeGreaterThan(before);
  await expect(thumb).toHaveClass(/visible/);
  await expect(thumb).toHaveCSS("height", "2px");

  await page.mouse.move(10, 200);
  await expect(thumb).not.toHaveClass(/visible/, { timeout: 2000 });
});

test("automatically reveals a selected hidden layer tab", async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 700 });
  await page.goto("/");
  await loadModel(page);

  await page.getByRole("button", { name: /^Global tensors / }).click();
  await page.getByRole("button", { name: /^Layer 0 / }).click();
  await page.getByRole("button", { name: /^Layer 1 / }).click();

  const tabs = page.getByRole("tablist", { name: "Open layers" });
  await tabs.evaluate((element) => {
    element.scrollLeft = element.scrollWidth;
  });
  const before = await tabs.evaluate((element) => element.scrollLeft);

  await page.getByRole("button", { name: /^Global tensors / }).click();

  await expect.poll(() => tabs.evaluate((element) => element.scrollLeft)).toBeLessThan(before);
  await expect(page.getByRole("tab", { name: "Global tensors" })).toBeInViewport();
});

test("reorders editor tabs by dragging them horizontally", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 700 });
  await page.goto("/");
  await loadModel(page);

  await page.getByRole("button", { name: /^Global tensors / }).click();
  await page.getByRole("button", { name: /^Layer 0 / }).click();
  await page.getByRole("button", { name: /^Layer 1 / }).click();

  const tabNames = () =>
    page
      .getByRole("tablist", { name: "Open layers" })
      .getByRole("tab")
      .evaluateAll((tabs) => tabs.map((tab) => tab.textContent?.trim() ?? ""));

  await expect.poll(tabNames).toEqual(["Global tensors", "Layer 0", "Layer 1"]);

  const source = await page.getByRole("tab", { name: "Layer 1" }).boundingBox();
  const target = await page.getByRole("tab", { name: "Global tensors" }).boundingBox();
  expect(source).not.toBeNull();
  expect(target).not.toBeNull();

  await page.mouse.move(source!.x + source!.width / 2, source!.y + source!.height / 2);
  await page.mouse.down();
  await page.mouse.move(target!.x + 4, target!.y + target!.height / 2);
  await page.mouse.up();

  await expect.poll(tabNames).toEqual(["Layer 1", "Global tensors", "Layer 0"]);
});
