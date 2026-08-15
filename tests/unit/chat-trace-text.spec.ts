import { expect, test } from "@playwright/test";

test("maps displayed reasoning and answer text to their original raw trace tokens", async ({ page }) => {
  await page.goto("/");

  const mapped = await page.evaluate(async () => {
    const { mapTraceText } = await import("/src/components/Workbench/chat/traceText.ts");
    const tokens = ["<think>", "Plan", " carefully", "</think>", "Final", " answer"];

    return {
      reasoning: mapTraceText(tokens, "Plan carefully"),
      answer: mapTraceText(tokens, "Final answer", "Plan carefully".length + "<think></think>".length),
    };
  });

  expect(mapped).toEqual({
    reasoning: [
      { tokenIndex: 1, text: "Plan" },
      { tokenIndex: 2, text: " carefully" },
    ],
    answer: [
      { tokenIndex: 4, text: "Final" },
      { tokenIndex: 5, text: " answer" },
    ],
  });
});

test("keeps a captured trace prefix clickable when a cancelled reply has an untraced tail", async ({ page }) => {
  await page.goto("/");

  const mapped = await page.evaluate(async () => {
    const { mapTraceText } = await import("/src/components/Workbench/chat/traceText.ts");
    return mapTraceText(
      ["Yes, I am familiar with Genshin Impact. It is an open-world action role-playing game developed by mi", "Ho"],
      "Yes, I am familiar with Genshin Impact. It is an open-world action role-playing game developed by miHoHo",
    );
  });

  expect(mapped).toEqual([
    { tokenIndex: 0, text: "Yes, I am familiar with Genshin Impact. It is an open-world action role-playing game developed by mi" },
    { tokenIndex: 1, text: "Ho" },
    { tokenIndex: null, text: "Ho" },
  ]);
});
