import { expect, test } from "@playwright/test";

test("keeps trace-pane visibility independent for each open Chat tab", async ({ page }) => {
  await page.goto("/", { waitUntil: "domcontentloaded" });

  await page.getByRole("button", { name: "Chat with model" }).click();
  await page.getByRole("button", { name: "New chat", exact: true }).click();

  const traceToggle = page.getByRole("button", { name: /Trace: (Off|On)/ });
  const tracePane = page.getByRole("complementary", { name: "Trace & Mechanistic Analysis" });

  await expect(traceToggle).toBeVisible();
  await expect(tracePane).toBeHidden();

  await traceToggle.click();
  await expect(traceToggle).toHaveAccessibleName("Trace: On");
  await expect(tracePane).toBeVisible();

  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(traceToggle).toHaveAccessibleName("Trace: Off");
  await expect(tracePane).toBeHidden();

  await page.getByRole("tab", { name: "New chat" }).first().click();
  await expect(traceToggle).toHaveAccessibleName("Trace: On");
  await expect(tracePane).toBeVisible();
});
