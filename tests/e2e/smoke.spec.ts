import { test, expect } from '@playwright/test';

test.describe('E2E Smoke', () => {
  test('full app shell renders correctly', async ({ page }) => {
    await page.goto('/');

    await expect(page.locator('text=Model Surgery')).toBeVisible();
    await expect(page.locator('text=Open GGUF...')).toBeVisible();
    await expect(page.locator('text=Bulk Assign')).toBeVisible();
    await expect(page.locator('text=Test Recipe')).toBeVisible();
    await expect(page.locator('text=Save Recipe')).toBeVisible();
    await expect(page.locator('text=Load Recipe')).toBeVisible();
    await expect(page.locator('text=Export GGUF')).toBeVisible();
  });

  test('preset menu has quant type options', async ({ page }) => {
    await page.goto('/');
    const presetSelect = page.locator('select').first();
    const options = await presetSelect.locator('option').allTextContents();
    expect(options).toContain('Q4_K_M');
    expect(options).toContain('Q8_0');
    expect(options).toContain('F16');
  });

  test('bulk assign panel shows all patterns', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('text=All Attention')).toBeVisible();
    await expect(page.locator('text=All FFN')).toBeVisible();
    await expect(page.locator('text=All Embeddings')).toBeVisible();
    await expect(page.locator('text=Entire Model')).toBeVisible();
  });
});
