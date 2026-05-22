import { test, expect } from '@playwright/test';

test.describe('Test Modal', () => {
  test('Test Recipe button is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('text=Test Recipe')).toBeVisible();
  });
});
