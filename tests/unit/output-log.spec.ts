import { expect, test } from "@playwright/test";

test("keeps benchmark output logs in the OUTPUT bottom tab without auto-switching", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("tab", { name: "HARDWARE" }).click();
  await expect(page.getByRole("tab", { name: "HARDWARE" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.waitForTimeout(50);

  await page.evaluate(() => {
    window.dispatchEvent(
      new CustomEvent("benchmark-output", {
        detail: { message: "GPQA Diamond started" },
      }),
    );
    window.dispatchEvent(
      new CustomEvent("benchmark-output", {
        detail: { message: "EvalScope stdout: Started task gpqa_diamond" },
      }),
    );
  });

  await expect(page.getByRole("tab", { name: "HARDWARE" })).toHaveAttribute(
    "aria-selected",
    "true",
  );

  await page.getByRole("tab", { name: "OUTPUT" }).click();
  await expect(page.getByRole("tab", { name: "OUTPUT" })).toHaveAttribute(
    "aria-selected",
    "true",
  );

  const output = page.getByRole("log", { name: "Benchmark output" });
  await expect(output).toBeVisible();
  await expect(output.getByText(/\[\d{2}:\d{2}:\d{2}\]GPQA Diamond started/)).toBeVisible();
  await expect(
    output.getByText(/\[\d{2}:\d{2}:\d{2}\]EvalScope stdout: Started task gpqa_diamond/),
  ).toBeVisible();
});

test("opens OUTPUT at the newest log and follows new lines while at the bottom", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "HARDWARE" }).click();
  await page.waitForTimeout(50);

  await page.evaluate(() => {
    for (let index = 1; index <= 80; index += 1) {
      window.dispatchEvent(
        new CustomEvent("benchmark-output", {
          detail: { message: `ModelInspector API: chat completion request ${index} completed` },
        }),
      );
    }
  });

  await page.getByRole("tab", { name: "OUTPUT" }).click();
  const output = page.getByRole("log", { name: "Benchmark output" });
  await expect(
    output.getByText("ModelInspector API: chat completion request 80 completed"),
  ).toBeVisible();

  const scrollState = await output.evaluate((node) => ({
    scrollTop: node.scrollTop,
    clientHeight: node.clientHeight,
    scrollHeight: node.scrollHeight,
  }));
  expect(scrollState.scrollTop + scrollState.clientHeight).toBeGreaterThanOrEqual(
    scrollState.scrollHeight - 2,
  );

  await page.evaluate(() => {
    window.dispatchEvent(
      new CustomEvent("benchmark-output", {
        detail: { message: "ModelInspector API: chat completion request 81 completed" },
      }),
    );
  });

  await expect(
    output.getByText("ModelInspector API: chat completion request 81 completed"),
  ).toBeVisible();
  const updatedScrollState = await output.evaluate((node) => ({
    scrollTop: node.scrollTop,
    clientHeight: node.clientHeight,
    scrollHeight: node.scrollHeight,
  }));
  expect(updatedScrollState.scrollTop + updatedScrollState.clientHeight).toBeGreaterThanOrEqual(
    updatedScrollState.scrollHeight - 2,
  );
});
