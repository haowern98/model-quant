import { test, expect } from '@playwright/test';

test.describe('Export', () => {
  test('Export GGUF button is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('text=Export GGUF')).toBeVisible();
  });
});
