import { test, expect } from '@playwright/test';

test.describe('E2E Smoke', () => {
  test('full app shell renders correctly', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('button', { name: 'Model Surgery command center' })).toBeVisible();
    await expect(page.getByText('EXPLORER')).toBeVisible();
    await expect(page.getByRole('button', { name: 'GGUF', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'MMPROJ', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'LORA ADAPTER', exact: true })).toBeVisible();
    await expect(page.getByLabel('Eval preset')).toBeVisible();
    await expect(page.getByLabel('Test mode')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run recipe test' })).toBeVisible();
  });

  test('preset menu has quant type options', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model Surgery command center' }).click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });
    await page.getByRole('button', { name: 'Model actions' }).click();
    const presetSelect = page.getByLabel('Entire Model target');
    const options = await presetSelect.locator('option').allTextContents();
    expect(options).toContain('Q4_K');
    expect(options).toContain('Q3_K');
    expect(options).toContain('Q2_K');
    expect(options).toContain('Q8_0');
    expect(options).toContain('F16');
    expect(options).not.toContain('Q5_K_M');
    expect(options).not.toContain('Q4_K_M');
    expect(options).not.toContain('Q3_K_M');
  });

  test('model actions menu shows all bulk assign patterns', async ({ page }) => {
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
