import { test, expect } from '@playwright/test';

test.describe('Layer Select', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('shows VS Code workbench explorer before a model loads', async ({ page }) => {
    await expect(page.getByText('EXPLORER')).toBeVisible();
    await expect(page.getByRole('button', { name: 'GGUF', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'MMPROJ', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'LORA ADAPTER', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'No layer selected' })).toBeVisible();
  });

  test('opens clicked layers as editor tabs', async ({ page }) => {
    await page.getByRole('button', { name: 'Model Surgery command center' }).click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });

    await page.getByRole('button', { name: /^Layer 0 / }).click();
    await expect(page.getByRole('tab', { name: 'Layer 0' })).toBeVisible();

    await page.getByRole('button', { name: /^Layer 1 / }).click();
    await expect(page.getByRole('tab', { name: 'Layer 0' })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'Layer 1' })).toBeVisible();
  });
});
