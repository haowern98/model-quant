import { test, expect } from '@playwright/test';

test.describe('Layout containment', () => {
  test('keeps size profile and editor run controls visible in the app viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/');

    await page.getByRole('button', { name: 'Model Surgery command center' }).click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });
    await page.getByText('Layer 0', { exact: false }).click();

    await expect(page.getByRole('tab', { name: 'SIZE PROFILE' })).toBeVisible();
    await expect(page.getByLabel('Eval preset')).toBeVisible();
    await expect(page.getByLabel('Test mode')).toBeVisible();
  });
});
