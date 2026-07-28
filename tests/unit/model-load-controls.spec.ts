import { expect, test } from "@playwright/test";

test("keeps Load Model enabled and explains when no GGUF is open", async ({ page }) => {
  await page.goto("/");

  const loadButton = page.getByRole("button", { name: "Load model", exact: true });
  const loadChevron = page.getByRole("button", { name: "Load model options" });
  const runButton = page.getByRole("button", { name: "Run recipe test" });
  const runChevron = page.getByRole("button", { name: "Test run options" });

  await expect(loadButton).toBeEnabled();
  await expect(loadChevron).toBeEnabled();

  await loadButton.click();
  await expect(page.getByRole("alert")).toContainText("Open a GGUF model first.");

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
  expect(gap[1]!.x - (gap[0]!.x + gap[0]!.width)).toBeLessThan(10);
});

test("opens a persisted Model Load configuration dropdown", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Load model options" }).click();

  const configuration = page.getByRole("dialog", { name: "Model load configuration" });
  await expect(configuration.getByText("CONFIGURATION", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Load model seed")).toHaveValue("");
  await expect(page.getByRole("button", { name: "Load model thinking Off" })).toBeVisible();
  await expect(page.getByLabel("Load model temperature")).toHaveValue("0");
  await expect(page.getByLabel("Load model top K sampling")).toHaveValue("40");
  await expect(page.getByLabel("Load model repeat penalty")).toHaveValue("1.1");
  await expect(page.getByLabel("Load model presence penalty")).toHaveValue("0");
  await expect(page.getByLabel("Load model top P sampling")).toHaveValue("0.95");
  await expect(page.getByLabel("Load model min P sampling")).toHaveValue("0.05");
  await expect(page.getByLabel("Load model context window")).toHaveValue("20000");

  await page.getByLabel("Load model temperature").fill("0.2");
  await page.keyboard.press("Escape");
  await expect(configuration).toBeHidden();

  await page.getByRole("button", { name: "Load model options" }).click();
  await expect(page.getByLabel("Load model temperature")).toHaveValue("0.2");
});

test("matches the Official Benchmarks heading and has no configuration row separators", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Load model options" }).click();
  const configuration = page.getByRole("dialog", { name: "Model load configuration" });
  const configurationHeading = configuration.getByText("CONFIGURATION", { exact: true });
  const configurationStyles = await configurationHeading.evaluate((element) => {
    const styles = getComputedStyle(element);
    return {
      color: styles.color,
      fontSize: styles.fontSize,
      fontWeight: styles.fontWeight,
      height: styles.height,
      paddingLeft: styles.paddingLeft,
    };
  });
  const rowBorders = await configuration.locator(".benchmark-info-row").evaluateAll((rows) =>
    rows.map((row) => getComputedStyle(row).borderTopWidth),
  );

  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Test run options" }).click();
  const officialHeading = page.getByText("OFFICIAL BENCHMARKS", { exact: true });
  const officialStyles = await officialHeading.evaluate((element) => {
    const styles = getComputedStyle(element);
    return {
      color: styles.color,
      fontSize: styles.fontSize,
      fontWeight: styles.fontWeight,
      height: styles.height,
      paddingLeft: styles.paddingLeft,
    };
  });

  expect(configurationStyles).toEqual(officialStyles);
  expect(rowBorders).toEqual(Array(rowBorders.length).fill("0px"));
});
