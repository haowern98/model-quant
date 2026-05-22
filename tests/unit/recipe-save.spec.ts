import { test, expect } from '@playwright/test';

test.describe('Recipe Save/Load', () => {
  test('Save and Load Recipe buttons are visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('text=Save Recipe')).toBeVisible();
    await expect(page.locator('text=Load Recipe')).toBeVisible();
  });
});
