import { test, expect } from '@playwright/test';

test.describe('Quant Assign', () => {
  test('preset dropdown renders with options', async ({ page }) => {
    await page.goto('/');
    const presetSelect = page.locator('select').first();
    await expect(presetSelect).toBeVisible();
  });

  test('disables target quant dropdown for tensors blocked by preflight', async ({ page }) => {
    await page.goto('/');

    await page.getByText('Open GGUF...').click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });
    await page.getByText('Global tensors', { exact: false }).click();

    const normRow = page.locator('tr').filter({ hasText: 'output_norm.weight' });
    await expect(normRow.locator('select')).toBeDisabled();
  });

  test('omits larger target quant options for already quantized tensors', async ({ page }) => {
    await page.goto('/');

    await page.getByText('Open GGUF...').click();
    await page.locator('input[type="file"]').setInputFiles({
      name: 'mock.gguf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('mock'),
    });
    await page.getByText('Layer 0', { exact: false }).click();

    const q8Row = page.locator('tr').filter({ hasText: 'attention.wq.weight' });
    const options = await q8Row.locator('select option').allTextContents();

    expect(options).toEqual(['Q8_0', 'Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'Q2_K']);
    expect(options).not.toContain('F32');
    expect(options).not.toContain('BF16');
    expect(options).not.toContain('F16');
  });
});
