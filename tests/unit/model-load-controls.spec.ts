import { expect, test } from "@playwright/test";

test("shows a disabled Load Model split control matching the benchmark control dimensions", async ({ page }) => {
  await page.goto("/");

  const loadButton = page.getByRole("button", { name: "Load model", exact: true });
  const loadChevron = page.getByRole("button", { name: "Load model options" });
  const runButton = page.getByRole("button", { name: "Run recipe test" });
  const runChevron = page.getByRole("button", { name: "Test run options" });

  await expect(loadButton).toBeDisabled();
  await expect(loadChevron).toBeDisabled();

  const dimensions = await Promise.all(
    [loadButton, loadChevron, runButton, runChevron].map((control) =>
      control.evaluate((element) => {
        const { width, height } = element.getBoundingClientRect();
        return { width, height };
      }),
    ),
  );

  expect(dimensions[0]).toEqual(dimensions[2]);
  expect(dimensions[1]).toEqual(dimensions[3]);
});

test("keeps the Load and Run split controls close together", async ({ page }) => {
  await page.goto("/");

  const loadChevron = page.getByRole("button", { name: "Load model options" });
  const runButton = page.getByRole("button", { name: "Run recipe test" });
  const gap = await Promise.all([loadChevron, runButton].map((control) => control.boundingBox()));

  expect(gap[0]).not.toBeNull();
  expect(gap[1]).not.toBeNull();
  expect(gap[1]!.x - (gap[0]!.x + gap[0]!.width)).toBeLessThan(20);
});
