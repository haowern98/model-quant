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

test("keeps layer headers visible while the Logit Lens table scrolls horizontally", async ({ page }) => {
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

  const tableWrap = page.locator(".chat-trace-table-wrap").last();
  await tableWrap.evaluate((element) => { element.scrollLeft = 160; });
  await expect.poll(() => tableWrap.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);

  const wrapBox = await tableWrap.boundingBox();
  const layerHeaderBox = await tableWrap.getByRole("columnheader", { name: "Layer" }).boundingBox();
  const firstLayerBox = await tableWrap.getByRole("button", { name: "L0" }).boundingBox();
  expect(layerHeaderBox?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
  expect(firstLayerBox?.x).toBeGreaterThanOrEqual(wrapBox!.x - 1);
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
