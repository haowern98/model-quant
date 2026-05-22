import { test, expect } from '@playwright/test';

test.describe('Bulk Assign', () => {
  test('renders bulk assign panel with patterns', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('text=Bulk Assign')).toBeVisible();
    await expect(page.locator('text=All Attention')).toBeVisible();
    await expect(page.locator('text=All FFN')).toBeVisible();
    await expect(page.locator('text=All Embeddings')).toBeVisible();
    await expect(page.locator('text=Entire Model')).toBeVisible();
  });
});
