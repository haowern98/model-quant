import { test, expect } from '@playwright/test';

test.describe('Layout containment', () => {
  test('keeps size profile and bulk assign visible in the app viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/');

    await page.getByText('Open GGUF...').click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });
    await page.getByText('Layer 0', { exact: false }).click();

    await expect(page.getByText('Size Profile')).toBeVisible();
    await expect(page.getByText('Bulk Assign')).toBeVisible();
  });
});
