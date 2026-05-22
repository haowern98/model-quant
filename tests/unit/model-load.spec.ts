import { test, expect } from '@playwright/test';

test.describe('Model Load', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders app shell on load', async ({ page }) => {
    await expect(page.locator('text=Model Surgery')).toBeVisible();
    await expect(page.locator('text=Open GGUF...')).toBeVisible();
  });
});
