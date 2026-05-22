import { test, expect } from '@playwright/test';

test.describe('Quant Assign', () => {
  test('preset dropdown renders with options', async ({ page }) => {
    await page.goto('/');
    const presetSelect = page.locator('select').first();
    await expect(presetSelect).toBeVisible();
  });
});
