import { expect, test } from "@playwright/test";

test("renders a connected command center without changing the titlebar height", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");

  const titlebar = page.locator(".titlebar");
  const commandCenter = page.getByRole("button", { name: "Model Surgery command center" });
  const commandMenu = page.getByRole("button", { name: "Command center menu" });
  const closeButton = page.getByRole("button", { name: "Close" });

  await expect(titlebar).toHaveCSS("height", "35px");
  await expect(commandCenter).toHaveCSS("height", "24px");
  await expect(commandMenu).toBeVisible();
  await expect(closeButton).toBeInViewport();

  const spacing = await page.evaluate(() => {
    const titlebarBox = document.querySelector(".titlebar")!.getBoundingClientRect();
    const commandBox = document.querySelector(".command-center-group")!.getBoundingClientRect();
    return {
      overflow: titlebarBox.right - window.innerWidth,
      top: commandBox.top - titlebarBox.top,
      bottom: titlebarBox.bottom - commandBox.bottom,
      horizontalOffset:
        commandBox.left + commandBox.width / 2 -
        (titlebarBox.left + titlebarBox.width / 2),
    };
  });

  expect(spacing.overflow).toBeLessThanOrEqual(0);
  expect(Math.abs(spacing.top - spacing.bottom)).toBeLessThanOrEqual(1);
  expect(Math.abs(spacing.horizontalOffset)).toBeLessThanOrEqual(1);

  await commandMenu.click();
  await expect(page.getByRole("menuitem", { name: "Open model GGUF..." })).toBeVisible();
});
