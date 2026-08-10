import { expect, test } from "@playwright/test";

test("opens the inspecting-position control as a stable-gutter trace popover", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    document.body.append(host);
    const candidates = Array.from({ length: 12 }, (_, index) => ({
      tokenId: index,
      tokenText: `candidate-${index}`,
      logit: index,
      probability: 1 / 12,
    }));
    const tokens = Array.from({ length: 20 }, (_, index) => ({
      index: index + 1,
      tokenId: index,
      tokenText: `token-${index + 1}`,
      logit: index,
      rank: index + 1,
      logitNormalizer: 0,
      candidates,
      layers: [{ layer: 0, candidates }],
    }));

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: { supported: true, tokens },
      selectedTokenIndex: 0,
      onSelectTokenIndex: () => undefined,
      loading: false,
      error: null,
    }));
  });

  const trigger = page.getByRole("button", { name: "Token #1: token-1" });
  await trigger.click();
  const listbox = page.getByRole("listbox", { name: "Inspecting position" });
  await expect(listbox).toBeVisible();
  await expect(listbox.getByRole("option")).toHaveCount(20);
  await expect(listbox).toHaveCSS("scrollbar-gutter", "stable");
  const triggerBox = await trigger.boundingBox();
  const listboxBox = await listbox.boundingBox();
  expect(listboxBox?.y).toBeGreaterThan(triggerBox!.y + triggerBox!.height);
  expect(listboxBox?.y).toBeLessThan(triggerBox!.y + triggerBox!.height + 20);
});

test("keeps the Layer lane aligned while Logit Lens ranks scroll horizontally", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    host.style.width = "240px";
    document.body.append(host);
    const candidates = Array.from({ length: 12 }, (_, index) => ({
      tokenId: index,
      tokenText: `candidate-${index}`,
      logit: index,
      probability: 1 / 12,
    }));

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: {
        supported: true,
        tokens: [{
          index: 1,
          tokenId: 0,
          tokenText: "token",
          logit: 0,
          rank: 1,
          logitNormalizer: 0,
          candidates,
          layers: Array.from({ length: 6 }, (_, layer) => ({ layer, candidates })),
        }],
      },
      selectedTokenIndex: 0,
      onSelectTokenIndex: () => undefined,
      loading: false,
      error: null,
    }));
  });

  const gridWrap = page.locator(".chat-trace-grid-wrap").last();
  const grid = page.locator(".chat-trace-grid").last();
  const tableScrollbar = page.locator(".chat-trace-grid-scrollbar").last();
  await expect(grid).toBeVisible();
  await expect(tableScrollbar).toBeVisible();
  await expect(tableScrollbar).toHaveCSS("scrollbar-gutter", "stable");

  const title = page.locator(".chat-trace-title");
  const layerHeader = grid.getByRole("columnheader", { name: "Layer" });
  const firstLayer = grid.getByRole("button", { name: "L0" });
  const firstRank = grid.getByRole("button", { name: /candidate-0/ }).first();
  const before = await firstLayer.boundingBox();
  const firstRankBox = await firstRank.boundingBox();
  const textLeft = async (locator: typeof title) => locator.evaluate((element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    return range.getBoundingClientRect().x;
  });
  expect(await textLeft(layerHeader)).toBeCloseTo(await textLeft(title), 1);
  expect(before?.y).toBeCloseTo(firstRankBox!.y, 1);
  expect(firstRankBox?.width).toBeLessThanOrEqual(80);
  expect(firstRankBox?.height).toBeLessThanOrEqual(44);

  await gridWrap.evaluate((element) => { element.scrollLeft = 160; });
  await expect.poll(() => gridWrap.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
  await expect.poll(() => tableScrollbar.evaluate((element) => element.scrollLeft)).toBe(160);

  await tableScrollbar.evaluate((element) => {
    element.scrollLeft = 80;
    element.dispatchEvent(new Event("scroll"));
  });
  await expect.poll(() => gridWrap.evaluate((element) => element.scrollLeft)).toBe(80);

  const wrapBox = await gridWrap.boundingBox();
  const after = await firstLayer.boundingBox();
  const layerHeaderBox = await layerHeader.boundingBox();
  expect(after?.x).toBeCloseTo(before!.x, 1);
  expect(layerHeaderBox?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
  expect(after?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
});

test("uses the generated-token outline without outlining the entire selected row", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    document.body.append(host);
    const candidates = [
      { tokenId: 0, tokenText: "generated", logit: 1, probability: 0.75 },
      { tokenId: 1, tokenText: "other", logit: 0, probability: 0.25 },
    ];

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: {
        supported: true,
        tokens: [{
          index: 1,
          tokenId: 0,
          tokenText: "generated",
          logit: 1,
          rank: 1,
          logitNormalizer: 0,
          candidates,
          layers: [{ layer: 0, candidates }],
        }],
      },
      selectedTokenIndex: 0,
      onSelectTokenIndex: () => undefined,
      loading: false,
      error: null,
    }));
  });

  const selectedRow = page.locator(".chat-trace-selected-row");
  const otherCandidate = selectedRow.getByRole("button", { name: /other/ });
  const generatedCandidate = selectedRow.getByRole("button", { name: /generated/ });
  await expect(otherCandidate).toHaveCSS("box-shadow", "none");
  const generatedHighlight = await generatedCandidate.evaluate((element) => {
    const style = getComputedStyle(element);
    return { boxShadow: style.boxShadow, outlineStyle: style.outlineStyle };
  });
  expect(generatedHighlight.outlineStyle).toBe("none");
  expect(generatedHighlight.boxShadow).toContain("rgb(14, 99, 156)");
});

test("arms tracing independently for each Chat tab without opening the trace pane", async ({ page }) => {
  await page.goto("/", { waitUntil: "domcontentloaded" });

  await page.getByRole("button", { name: "Chat with model" }).click();
  await page.getByRole("button", { name: "New chat", exact: true }).click();

  const traceToggle = page.getByRole("button", { name: /Trace: (Off|On)/ });
  const tracePane = page.getByRole("complementary", { name: "Trace & Mechanistic Analysis" });

  await expect(traceToggle).toBeVisible();
  await expect(tracePane).toBeHidden();

  await traceToggle.click();
  await expect(traceToggle).toHaveAccessibleName("Trace: On");
  await expect(tracePane).toBeHidden();

  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(traceToggle).toHaveAccessibleName("Trace: Off");
  await expect(tracePane).toBeHidden();

  await page.getByRole("tab", { name: "New chat" }).first().click();
  await expect(traceToggle).toHaveAccessibleName("Trace: On");
  await expect(tracePane).toBeHidden();
});
