export type TraceTextPiece = {
  tokenIndex: number | null;
  text: string;
};

export function mapTraceText(
  tokenTexts: string[],
  displayedText: string,
  startAt = 0,
): TraceTextPiece[] | undefined {
  if (displayedText.length === 0) return [];

  const rawText = tokenTexts.join("");
  let textStart = rawText.indexOf(displayedText, startAt);
  let tracedLength = displayedText.length;
  if (textStart < 0) {
    let bestStart = -1;
    let bestLength = 0;
    let candidateStart = rawText.indexOf(displayedText[0], startAt);
    while (candidateStart >= 0) {
      const candidateLimit = Math.min(rawText.length - candidateStart, displayedText.length);
      let candidateLength = 0;
      while (
        candidateLength < candidateLimit
        && rawText[candidateStart + candidateLength] === displayedText[candidateLength]
      ) {
        candidateLength += 1;
      }
      if (candidateLength > bestLength) {
        bestStart = candidateStart;
        bestLength = candidateLength;
      }
      candidateStart = rawText.indexOf(displayedText[0], candidateStart + 1);
    }
    if (bestLength === 0) return undefined;
    textStart = bestStart;
    tracedLength = bestLength;
  }

  const textEnd = textStart + tracedLength;
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

  if (pieces.map((piece) => piece.text).join("") !== displayedText.slice(0, tracedLength)) {
    return undefined;
  }
  if (tracedLength < displayedText.length) {
    pieces.push({ tokenIndex: null, text: displayedText.slice(tracedLength) });
  }
  return pieces;
}
