export type TraceTextPiece = {
  tokenIndex: number;
  text: string;
};

export function mapTraceText(
  tokenTexts: string[],
  displayedText: string,
  startAt = 0,
): TraceTextPiece[] | undefined {
  if (displayedText.length === 0) return [];

  const rawText = tokenTexts.join("");
  const textStart = rawText.indexOf(displayedText, startAt);
  if (textStart < 0) return undefined;

  const textEnd = textStart + displayedText.length;
  const pieces: TraceTextPiece[] = [];
  let tokenStart = 0;

  for (const [tokenIndex, tokenText] of tokenTexts.entries()) {
    const tokenEnd = tokenStart + tokenText.length;
    const overlapStart = Math.max(tokenStart, textStart);
    const overlapEnd = Math.min(tokenEnd, textEnd);
    if (overlapStart < overlapEnd) {
      pieces.push({
        tokenIndex,
        text: tokenText.slice(overlapStart - tokenStart, overlapEnd - tokenStart),
      });
    }
    tokenStart = tokenEnd;
  }

  return pieces.map((piece) => piece.text).join("") === displayedText ? pieces : undefined;
}
