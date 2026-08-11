import { expect, test, type Page } from "@playwright/test";

async function mountResizableTraceChat(page: Page, withSupportedTrace = false) {
  await page.goto("/");

  await page.evaluate(async (supportedTrace) => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatEditor } = await import("/src/components/Workbench/ChatEditor.tsx");
    const host = document.createElement("div");
    host.style.width = "1000px";
    host.style.height = "640px";
    host.style.display = "grid";
    document.body.append(host);

    function Harness() {
      const [traceOpen, setTraceOpen] = React.useState(true);
      return React.createElement(ChatEditor, {
        messages: [{
          id: "assistant-1",
          role: "assistant",
          content: "Saved trace response",
          model: "Test model",
          trace: {
            conversationId: "conversation-1",
            assistantMessageId: "assistant-1",
            modelFingerprint: "model",
            tokenCount: 1,
          },
        }],
        draft: "",
        modelReady: true,
        sending: false,
        disabled: false,
        error: null,
        traceArmed: false,
        tracePanelOpen: traceOpen,
        traceMessageId: "assistant-1",
        tracePayload: supportedTrace ? {
          supported: true,
          tokens: [{
            index: 1,
            tokenId: 0,
            tokenText: "token",
            logit: 0,
            rank: 1,
            logitNormalizer: 0,
            candidates: [{ tokenId: 0, tokenText: "token", logit: 0, probability: 1 }],
            layers: [{ layer: 0, candidates: [{ tokenId: 0, tokenText: "token", logit: 0, probability: 1 }] }],
          }],
        } : { supported: false, tokens: [] },
        traceTokenIndex: 0,
        traceLoading: false,
        traceError: null,
        onDraftChange: () => undefined,
        onTraceArmedChange: () => undefined,
        onOpenTrace: () => setTraceOpen(true),
        onCloseTrace: () => setTraceOpen(false),
        onTraceTokenChange: () => undefined,
        onSend: () => undefined,
      });
    }

    ReactDomClient.createRoot(host).render(React.createElement(Harness));
  }, withSupportedTrace);
}

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
    host.style.height = "360px";
    host.style.display = "grid";
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
  const tracePanel = page.locator(".chat-trace-panel").last();
  await expect(grid).toBeVisible();
  await expect(tableScrollbar).toBeVisible();
  await expect(tableScrollbar).toHaveCSS("scrollbar-gutter", "stable");
  const scrollbarBox = await tableScrollbar.boundingBox();
  const panelBox = await tracePanel.boundingBox();
  expect(scrollbarBox!.y + scrollbarBox!.height).toBeCloseTo(panelBox!.y + panelBox!.height, 1);

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

  await tracePanel.evaluate((element) => { element.scrollTop = 80; });
  await expect.poll(() => tracePanel.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
  const scrolledScrollbarBox = await tableScrollbar.boundingBox();
  const scrolledPanelBox = await tracePanel.boundingBox();
  expect(scrolledScrollbarBox!.y + scrolledScrollbarBox!.height).toBeCloseTo(scrolledPanelBox!.y + scrolledPanelBox!.height, 1);

  const wrapBox = await gridWrap.boundingBox();
  const after = await firstLayer.boundingBox();
  const layerHeaderBox = await layerHeader.boundingBox();
  expect(after?.x).toBeCloseTo(before!.x, 1);
  expect(layerHeaderBox?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
  expect(after?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
});

test("uses a red selected-cell outline while preserving generated-token blue", async ({ page }) => {
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

  await generatedCandidate.click();
  await expect(generatedCandidate).toHaveCSS("box-shadow", "rgb(244, 135, 113) 0px 0px 0px 2px inset");

  await otherCandidate.click();
  await expect(otherCandidate).toHaveCSS("box-shadow", "rgb(244, 135, 113) 0px 0px 0px 2px inset");
  await expect(generatedCandidate).toHaveCSS("box-shadow", "rgb(14, 99, 156) 0px 0px 0px 2px inset");
});

test("shows the clicked candidate token ID in the selection details", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    document.body.append(host);
    const candidates = [
      { tokenId: 2536, tokenText: "generated", logit: 1, probability: 0.75 },
      { tokenId: 4242, tokenText: "selected", logit: 0, probability: 0.25 },
    ];

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: {
        supported: true,
        tokens: [{
          index: 1,
          tokenId: 2536,
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

  const details = page.locator(".chat-trace-details").last();
  await expect(details).toContainText("Token ID");
  await expect(details).toContainText("2536");

  await page.getByRole("button", { name: /selected/ }).click();
  await expect(details).toContainText("Selected token ID");
  await expect(details).toContainText("4242");
});

test("uses a table-wide red-yellow-blue heatmap while preserving cell outlines", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    document.body.append(host);
    const upperCandidates = [
      { tokenId: 0, tokenText: "highest", logit: 30, probability: 0.75 },
      { tokenId: 4, tokenText: "between", logit: 26.25, probability: 0.5 },
      { tokenId: 1, tokenText: "upper", logit: 20, probability: 0.25 },
    ];
    const lowerCandidates = [
      { tokenId: 2, tokenText: "middle", logit: 15, probability: 0.75 },
      { tokenId: 3, tokenText: "lowest", logit: 0, probability: 0.25 },
    ];

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: {
        supported: true,
        tokens: [{
          index: 1,
          tokenId: 0,
          tokenText: "highest",
          logit: 30,
          rank: 1,
          logitNormalizer: 0,
          candidates: upperCandidates,
          layers: [{ layer: 0, candidates: upperCandidates }, { layer: 1, candidates: lowerCandidates }],
        }],
      },
      selectedTokenIndex: 0,
      onSelectTokenIndex: () => undefined,
      loading: false,
      error: null,
    }));
  });

  const grid = page.getByRole("grid", { name: "Logit Lens predictions" });
  const highestCandidate = grid.getByRole("button", { name: /highest/ });
  const betweenCandidate = grid.getByRole("button", { name: /between/ });
  const middleCandidate = grid.getByRole("button", { name: /middle/ });
  const lowestCandidate = grid.getByRole("button", { name: /lowest/ });
  const heatmapToggle = page.getByRole("button", { name: "Heatmap: Off" });

  await heatmapToggle.click();
  await expect(page.getByRole("button", { name: "Heatmap: On" })).toBeVisible();
  await expect(highestCandidate).toHaveCSS("background-color", "rgb(165, 0, 38)");
  await expect(betweenCandidate).toHaveCSS("background-color", "rgb(205, 55, 53)");
  await expect(middleCandidate).toHaveCSS("background-color", "rgb(255, 241, 184)");
  await expect(lowestCandidate).toHaveCSS("background-color", "rgb(49, 54, 149)");
  await expect(highestCandidate).toHaveCSS("box-shadow", "rgb(14, 99, 156) 0px 0px 0px 2px inset");

  await lowestCandidate.click();
  await expect(lowestCandidate).toHaveCSS("box-shadow", "rgb(244, 135, 113) 0px 0px 0px 2px inset");
});

test("renders very small probabilities in scientific notation", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import("/src/components/Workbench/ChatTracePanel.tsx");
    const host = document.createElement("div");
    document.body.append(host);
    const candidates = [
      { tokenId: 0, tokenText: "tiny", logit: 1, probability: 0.0000032 },
      { tokenId: 1, tokenText: "other", logit: 0, probability: 0.0000008 },
    ];

    ReactDomClient.createRoot(host).render(React.createElement(ChatTracePanel, {
      trace: {
        supported: true,
        tokens: [{
          index: 1,
          tokenId: 0,
          tokenText: "tiny",
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

  await page.getByRole("button", { name: "Logit Lens (Logit)" }).click();
  await page.getByRole("option", { name: "Logit Lens (Probability)" }).click();
  await expect(page.getByRole("grid", { name: "Logit Lens predictions" }).getByRole("button", { name: /tiny/ })).toContainText("3.20e-6");
});

test("closes the trace below its minimum width and reopens it from Trace saved", async ({ page }) => {
  await mountResizableTraceChat(page);

  const tracePanel = page.getByRole("complementary", { name: "Logit Lens" });
  const resizer = page.getByRole("separator", { name: "Resize Logit Lens" });
  await expect(tracePanel).toBeVisible();
  await expect(resizer).toBeVisible();
  await expect(resizer).toHaveCSS("cursor", "col-resize");
  await resizer.scrollIntoViewIfNeeded();

  const resizerBox = await resizer.boundingBox();
  await page.mouse.move(resizerBox!.x + 2, resizerBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(resizerBox!.x + 8, resizerBox!.y + 120);
  await page.mouse.up();

  await expect(tracePanel).toBeHidden();
  await page.getByRole("button", { name: "Trace saved" }).click();
  await expect(tracePanel).toBeVisible();
});

test("expands Logit Lens to the editor width beyond the split limit", async ({ page }) => {
  await mountResizableTraceChat(page);

  const resizer = page.getByRole("separator", { name: "Resize Logit Lens" });
  await resizer.scrollIntoViewIfNeeded();
  const resizerBox = await resizer.boundingBox();
  await page.mouse.move(resizerBox!.x + 2, resizerBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(resizerBox!.x - 400, resizerBox!.y + 120);
  await page.mouse.up();

  await expect(page.locator(".chat-editor")).toHaveClass(/chat-editor-trace-fullscreen/);
  await expect(page.locator(".chat-editor-main")).toBeHidden();
});

test("maximizes, restores, and closes Logit Lens from its header actions", async ({ page }) => {
  await mountResizableTraceChat(page, true);

  const tracePanel = page.getByRole("complementary", { name: "Logit Lens" });
  const resizer = page.getByRole("separator", { name: "Resize Logit Lens" });
  const maximize = page.getByRole("button", { name: "Maximize trace panel" });
  const close = page.getByRole("button", { name: "Close trace panel" });
  await resizer.scrollIntoViewIfNeeded();
  const resizerBox = await resizer.boundingBox();
  await page.mouse.move(resizerBox!.x + 2, resizerBox!.y + 120);
  await page.mouse.down();
  await page.mouse.move(resizerBox!.x - 80, resizerBox!.y + 120);
  await page.mouse.up();
  const restoredWidth = (await tracePanel.boundingBox())!.width;

  await expect(maximize).toHaveCSS("width", "18px");
  await expect(maximize).toHaveCSS("height", "24px");
  await expect(close).toHaveCSS("width", "18px");
  await expect(close).toHaveCSS("height", "24px");
  await maximize.click();
  const restore = page.getByRole("button", { name: "Restore trace panel" });
  await expect(restore).toBeVisible();
  await expect(page.locator(".chat-editor-main")).toBeHidden();
  await expect(resizer).toBeHidden();

  await restore.click();
  await expect(page.locator(".chat-editor-main")).toBeVisible();
  expect((await tracePanel.boundingBox())!.width).toBeCloseTo(restoredWidth, 1);

  await close.click();
  await expect(tracePanel).toBeHidden();
  await page.getByRole("button", { name: "Trace saved" }).click();
  await expect(tracePanel).toBeVisible();
});

test("styles the Logit Lens header like a Bottom Panel tab", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(async () => {
    const fresh = `trace-header-test=${Date.now()}`;
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDomClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { ChatTracePanel } = await import(`/src/components/Workbench/ChatTracePanel.tsx?${fresh}`);
    await import(`/src/index.css?${fresh}`);
    const host = document.createElement("div");
    document.body.append(host);
    const candidates = [{ tokenId: 0, tokenText: "token", logit: 0, probability: 1 }];

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
          layers: [{ layer: 0, candidates }],
        }],
      },
      selectedTokenIndex: 0,
      onSelectTokenIndex: () => undefined,
      loading: false,
      error: null,
    }));
  });

  const tab = page.locator(".chat-trace-tab-active");
  await expect(tab).toHaveText("LOGIT LENS");
  await expect(page.locator(".chat-trace-tabs")).toHaveCSS("height", "32px");
  await expect(tab).toHaveCSS("font-size", "11px");
  await expect(tab).toHaveCSS("border-bottom-width", "1px");
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
