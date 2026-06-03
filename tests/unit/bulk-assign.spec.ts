import { test, expect } from '@playwright/test';

test.describe('Bulk Assign', () => {
  test('renders bulk assign patterns from the model actions menu', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model Surgery command center' }).click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });

    await page.getByRole('button', { name: 'Model actions' }).click();

    await expect(page.getByText('All Attention')).toBeVisible();
    await expect(page.getByText('All FFN')).toBeVisible();
    await expect(page.getByText('All Embeddings')).toBeVisible();
    await expect(page.getByText('Entire Model')).toBeVisible();
  });
});
