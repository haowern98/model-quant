import { test, expect } from '@playwright/test';

test.describe('Layer Select', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('shows sidebar with filter and bulk assign when mock model loads', async ({ page }) => {
    await expect(page.getByPlaceholder('Filter layers...')).toBeVisible();
    await expect(page.locator('text=Bulk Assign')).toBeVisible();
    await expect(page.locator('text=No layer selected')).toBeVisible();
  });
});
