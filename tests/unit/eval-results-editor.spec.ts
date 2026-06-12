import { expect, test } from "@playwright/test";

async function loadModel(page: import("@playwright/test").Page) {
  await page.waitForFunction(
    () =>
      (window as Window & { __MODEL_SURGERY_MOCK_READY__?: boolean })
        .__MODEL_SURGERY_MOCK_READY__ === true,
  );
  const fileChooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Model Surgery command center" }).click();
  await (await fileChooser).setFiles({
    name: "mock.gguf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
  await expect(page.locator(".command-title")).toHaveText("mock.gguf");
}

test.describe("Eval Results editor", () => {
  test.describe.configure({ timeout: 60_000 });

  test("opens the run dropdown before a model is loaded", async ({ page }) => {
    await page.goto("/", { waitUntil: "domcontentloaded" });

    const runButton = page.getByRole("button", { name: "Run recipe test" });
    const menuButton = page.getByRole("button", { name: "Test run options" });

    await expect(runButton).toBeDisabled();
    await expect(menuButton).toBeEnabled();
    await expect(runButton.locator(".codicon")).toHaveCSS(
      "color",
      "rgb(204, 204, 204)",
    );
    await expect(runButton).toHaveCSS("width", "19px");
    await expect(runButton.locator(".codicon-run-all")).toHaveCSS("width", "16px");
    await expect(runButton.locator(".codicon-run-all")).toHaveCSS("height", "16px");
    await expect(menuButton).toHaveCSS("width", "18px");
    await expect(menuButton).toHaveCSS("margin-left", "0px");
    await expect(menuButton).toHaveCSS("opacity", "1");
    await expect(menuButton.locator(".codicon-chevron-down")).toHaveCSS(
      "opacity",
      "0.55",
    );
    await expect(menuButton.locator(".codicon-chevron-down")).toHaveCSS(
      "color",
      "rgb(204, 204, 204)",
    );
    await expect(menuButton.locator(".codicon-chevron-down")).toHaveCSS(
      "font-size",
      "12px",
    );
    await expect(menuButton.locator(".codicon-chevron-down")).toHaveCSS(
      "transform",
      "matrix(1, 0, 0, 1, -1, 0)",
    );

    const runBox = await runButton.boundingBox();
    if (!runBox) throw new Error("Run button missing");
    await page.mouse.move(runBox.x + runBox.width / 2, runBox.y + runBox.height / 2);
    await expect(runButton).toHaveCSS("background-color", "rgb(74, 74, 74)");
    await expect(menuButton).toHaveCSS("background-color", "rgb(51, 51, 51)");

    await menuButton.click();
    await expect(page.getByRole("menu", { name: "Test run options" })).toBeVisible();
    await expect(menuButton).toHaveCSS("outline-style", "none");
  });

  test("uses a split run control and keeps the dropdown available while testing", async ({
    page,
  }) => {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await loadModel(page);

    const runGroup = page.getByRole("group", { name: "Recipe test controls" });
    const runButton = page.getByRole("button", { name: "Run recipe test" });
    const menuButton = page.getByRole("button", { name: "Test run options" });
    await expect(runButton).toBeEnabled();
    await expect(runGroup.locator(".codicon-run-all")).toBeVisible();
    await expect(menuButton.locator(".codicon-chevron-down")).toBeVisible();

    await runButton.hover();
    await expect(runGroup).toHaveCSS("background-color", "rgb(51, 51, 51)");
    await expect(runButton).toHaveCSS("background-color", "rgb(74, 74, 74)");
    await expect(menuButton).toHaveCSS("background-color", "rgb(51, 51, 51)");

    await menuButton.click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    await expect(runMenu).toBeVisible();
    await expect(runMenu).toHaveCSS("font-size", "13px");
    await expect(runMenu).toHaveCSS("font-family", /Segoe UI/);
    await expect(runMenu).toHaveCSS("border-radius", "8px");
    await expect(runMenu.getByText("LOCAL CHECKS")).toBeVisible();
    await expect(runMenu.getByText("OFFICIAL BENCHMARKS")).toBeVisible();
    await expect(
      runMenu.getByRole("menuitemcheckbox", { name: /PPL Check Ready/ }),
    ).toHaveAttribute("aria-checked", "true");
    const checkedBox = runMenu
      .getByRole("menuitemcheckbox", { name: /PPL Check Ready/ })
      .locator(".run-menu-check");
    await expect(checkedBox).toHaveCSS("border-style", "solid");
    await expect(checkedBox).toHaveCSS("border-color", "rgb(204, 204, 204)");
    await expect(checkedBox.locator(".codicon-check")).toHaveCSS(
      "color",
      "rgb(204, 204, 204)",
    );
    await expect(
      runMenu.getByRole("menuitemcheckbox", { name: /GPQA Diamond Ready/ }),
    ).toHaveAttribute("aria-checked", "false");
    await expect(
      runMenu
        .getByRole("menuitemcheckbox", { name: /GPQA Diamond Ready/ })
        .locator(".run-menu-check"),
    ).toHaveCSS("border-color", "rgb(204, 204, 204)");
    await expect(runMenu.getByText("MMLU-Pro")).toBeVisible();
    await expect(runMenu.getByText("Download").first()).toBeVisible();
    await expect(runMenu.getByText("Claw-Eval")).toBeVisible();
    await expect(runMenu.getByText("Needs harness")).toBeVisible();
    await expect(runGroup).toHaveCSS("background-color", "rgb(51, 51, 51)");
    await expect(menuButton).toHaveCSS("background-color", "rgb(74, 74, 74)");
    await runButton.hover();
    await expect(runButton).toHaveCSS("background-color", "rgb(74, 74, 74)");
    await expect(menuButton).toHaveCSS("background-color", "rgb(51, 51, 51)");
    const menuPlacement = await page.evaluate(() => {
      const group = document.querySelector(".run-split-action")!.getBoundingClientRect();
      const menu = document.querySelector(".run-action-menu")!.getBoundingClientRect();
      return {
        menuRight: Math.round(menu.right),
        groupRight: Math.round(group.right),
      };
    });
    expect(menuPlacement.menuRight).toBeLessThanOrEqual(menuPlacement.groupRight + 1);

    await runButton.click();

    const cancelButton = page.getByRole("button", { name: "Cancel recipe test" });
    await expect(cancelButton).toBeVisible();
    await expect(runGroup.locator(".codicon-stop-circle")).toBeVisible();
    await expect(menuButton.locator(".codicon-chevron-down")).toBeVisible();

    await cancelButton.click();
    await expect(page.getByText("Cancelling test...")).toBeVisible();
    await expect(cancelButton).toBeDisabled();

    await expect(runButton).toBeVisible();
    await expect(
      page.getByRole("tablist", { name: "Open layers" }).getByRole("tab", {
        name: "Eval Results",
      }),
    ).toHaveCount(0);
  });

  test("allows only PPL Check and GPQA Diamond to be selected for a run", async ({
    page,
  }) => {
    await page.goto("/");
    await loadModel(page);

    await page.getByRole("button", { name: "Test run options" }).click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    const ppl = runMenu.getByRole("menuitemcheckbox", {
      name: /PPL Check Ready/,
    });
    const gpqa = runMenu.getByRole("menuitemcheckbox", {
      name: /GPQA Diamond Ready/,
    });
    const mmluPro = runMenu.getByRole("menuitemcheckbox", {
      name: /MMLU-Pro Download/,
    });
    const clawEval = runMenu.getByRole("menuitemcheckbox", {
      name: /Claw-Eval Needs harness/,
    });

    await expect(ppl).toHaveAttribute("aria-checked", "true");
    await expect(gpqa).toHaveAttribute("aria-checked", "false");

    await gpqa.click();
    await expect(gpqa).toHaveAttribute("aria-checked", "true");

    await ppl.click();
    await expect(ppl).toHaveAttribute("aria-checked", "false");

    await expect(mmluPro).toBeDisabled();
    await expect(clawEval).toBeDisabled();
  });

  test("keeps PPL and GPQA checkboxes toggleable even when unavailable", async ({
    page,
  }) => {
    await page.goto("/");

    await page.getByRole("button", { name: "Test run options" }).click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    const ppl = runMenu.getByRole("menuitemcheckbox", {
      name: /PPL Check Open model/,
    });
    const gpqa = runMenu.getByRole("menuitemcheckbox", {
      name: /GPQA Diamond Needs harness/,
    });
    const mmluPro = runMenu.getByRole("menuitemcheckbox", {
      name: /MMLU-Pro Download/,
    });

    await expect(ppl).toBeEnabled();
    await expect(gpqa).toBeEnabled();
    await expect(mmluPro).toBeDisabled();

    await expect(ppl).toHaveAttribute("aria-checked", "true");
    await ppl.click();
    await expect(ppl).toHaveAttribute("aria-checked", "false");

    await expect(gpqa).toHaveAttribute("aria-checked", "false");
    await gpqa.click();
    await expect(gpqa).toHaveAttribute("aria-checked", "true");
  });

  test("shows a clear error when GPQA is selected but not ready", async ({
    page,
  }) => {
    await page.goto("/?gpqaMissing=1");
    await loadModel(page);

    await page.getByRole("button", { name: "Test run options" }).click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    await runMenu.getByRole("menuitemcheckbox", { name: /PPL Check Ready/ }).click();
    await runMenu
      .getByRole("menuitemcheckbox", { name: /GPQA Diamond Install/ })
      .click();

    await page.getByRole("button", { name: "Run recipe test" }).click();

    await expect(page.getByRole("alert")).toContainText("GPQA Diamond is not ready");
  });

  test("shows a clear error when GPQA is selected before opening a model", async ({
    page,
  }) => {
    await page.goto("/");

    await page.getByRole("button", { name: "Test run options" }).click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    await expect(runMenu).toBeVisible();
    await runMenu.getByRole("menuitemcheckbox", { name: /GPQA Diamond/ }).click();
    await runMenu.getByRole("menuitemcheckbox", { name: /PPL Check Open model/ }).click();

    const runButton = page.getByRole("button", { name: "Run recipe test" });
    await expect(runButton).toBeEnabled();
    await runButton.click();

    await expect(page.getByRole("alert")).toContainText(
      "Open a GGUF model before running GPQA Diamond.",
    );
  });

  test("opens GPQA detail and dataset pages from the model evaluation panel", async ({
    page,
  }) => {
    await page.goto("/?gpqaMissing=1");
    await loadModel(page);
    await page.getByRole("button", { name: "Testing" }).click();

    const testingPanel = page.getByRole("complementary", { name: "Testing" });
    await expect(
      testingPanel.getByRole("button", { name: /GPQA Diamond Install/ }),
    ).toBeVisible();

    await testingPanel.getByRole("button", { name: "GPQA Diamond Details" }).click();
    await expect(
      page.getByRole("tab", { name: "GPQA Diamond Details" }),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Harness" })).toBeVisible();
    await expect(
      page.locator(".benchmark-editor-content .benchmark-info-row strong", {
        hasText: /^EvalScope$/,
      }),
    ).toBeVisible();
    await expect(page.getByText("gpqa_diamond")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Configuration" })).toBeVisible();
    const shotsButton = page.getByRole("button", { name: "GPQA Diamond shots 5-shot CoT" });
    await expect(shotsButton).toBeVisible();
    await expect(shotsButton).toHaveCSS("font-size", "13px");
    await expect(shotsButton).toHaveCSS("color", "rgb(204, 204, 204)");
    await shotsButton.click();
    const shotsMenu = page.getByRole("listbox", { name: "GPQA Diamond shots" });
    await expect(shotsMenu).toBeVisible();
    await expect(shotsMenu).toHaveCSS("background-color", "rgb(32, 32, 32)");
    await expect(shotsMenu).toHaveCSS("border-radius", "8px");
    await expect(shotsMenu).toHaveCSS("font-size", "13px");
    await expect(shotsMenu.getByRole("option", { name: "5-shot CoT" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await shotsMenu.getByRole("option", { name: "0-shot CoT" }).hover();
    await expect(shotsMenu.getByRole("option", { name: "0-shot CoT" })).toHaveCSS(
      "background-color",
      "rgb(42, 45, 46)",
    );
    await shotsMenu.getByRole("option", { name: "0-shot CoT" }).click();
    await expect(page.getByRole("button", { name: "GPQA Diamond shots 0-shot CoT" })).toBeVisible();
    await expect(
      page.locator(".benchmark-info-row strong", { hasText: /^0-shot CoT$/ }),
    ).toBeVisible();
    await expect(page.getByText("CoT", { exact: true })).toBeVisible();
    await expect(page.getByText("Batch size")).toBeVisible();
    await page.getByRole("button", { name: "Install harness" }).click();

    await expect(
      testingPanel.getByRole("button", { name: /GPQA Diamond Download/ }),
    ).toBeVisible();
    await expect(testingPanel.getByText("1.8.0").first()).toBeVisible();

    await testingPanel.getByRole("button", { name: "GPQA Diamond Dataset" }).click();
    await expect(
      page.getByRole("tab", { name: "GPQA Diamond Dataset" }),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Source" })).toBeVisible();
    await page.getByRole("button", { name: "Download dataset" }).click();
    await expect(
      testingPanel.getByRole("button", { name: /GPQA Diamond Ready/ }),
    ).toBeVisible();
  });

  test("passes GPQA numeric configuration with blank-field defaults", async ({
    page,
  }) => {
    await page.goto("/");
    await loadModel(page);
    await page.getByRole("button", { name: "Testing" }).click();

    await page
      .getByRole("complementary", { name: "Testing" })
      .getByRole("button", { name: "GPQA Diamond Details" })
      .click();

    const temperature = page.getByLabel("GPQA Diamond temperature");
    const maxTokens = page.getByLabel("GPQA Diamond max tokens");
    const samples = page.getByLabel("GPQA Diamond samples");

    await expect(temperature).toHaveValue("0");
    await expect(maxTokens).toHaveValue("");
    await expect(maxTokens).toHaveAttribute("placeholder", "1024");
    await expect(samples).toHaveValue("");
    await expect(samples).toHaveAttribute("placeholder", "198");

    await maxTokens.fill("4096");
    await samples.fill("12");
    await temperature.fill("0.2");
    await expect(maxTokens).toHaveValue("4096");
    await expect(samples).toHaveValue("12");
    await expect(temperature).toHaveValue("0.2");

    await maxTokens.fill("");
    await samples.fill("");
    await temperature.fill("");

    await page.getByRole("button", { name: "Test run options" }).click();
    const runMenu = page.getByRole("menu", { name: "Test run options" });
    await runMenu.getByRole("menuitemcheckbox", { name: /PPL Check/ }).click();
    await runMenu.getByRole("menuitemcheckbox", { name: /GPQA Diamond Ready/ }).click();
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Run recipe test" }).click();

    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (
              window as Window & {
                __MODEL_SURGERY_LAST_GPQA_ARGS__?: {
                  maxTokens?: number;
                  sampleLimit?: number;
                  temperature?: number;
                } | null;
              }
            ).__MODEL_SURGERY_LAST_GPQA_ARGS__,
        ),
      )
      .toMatchObject({
        config: {
          maxTokens: 1024,
          sampleLimit: 198,
          temperature: 0,
        },
      });
  });

  test("opens completed test results in one reusable editor tab", async ({ page }) => {
    await page.goto("/");
    await loadModel(page);
    await page.getByRole("button", { name: /^Layer 0 / }).click();

    await page.getByRole("button", { name: "Run recipe test" }).click();

    const editorTabs = page.getByRole("tablist", { name: "Open layers" });
    const evalResultsTab = editorTabs.getByRole("tab", { name: "Eval Results" });

    await expect(evalResultsTab).toBeVisible();
    await expect(evalResultsTab).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(editorTabs.getByRole("tab", { name: "Layer 0" })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Benchmark Results" }),
    ).toBeVisible();
    await expect(page.locator(".eval-results-editor")).toHaveCSS(
      "overflow-y",
      "auto",
    );
    await expect(page.getByText("Verified targets")).toBeVisible();
    await expect(page.getByText("0/0", { exact: true })).toBeVisible();
    await expect(page.locator(".fixed.inset-0")).toHaveCount(0);

    await page.getByRole("button", { name: "Run recipe test" }).click();
    await expect(evalResultsTab).toHaveCount(1);
  });
});
