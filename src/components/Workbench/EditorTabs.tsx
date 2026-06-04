import { useEffect, useLayoutEffect, useRef, useState, type WheelEvent } from "react";
import { editorTabLabel, type EditorTab } from "./editorTabModel";

interface EditorTabsProps {
  openEditors: EditorTab[];
  activeEditorId: string | null;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
}

export function EditorTabs({
  openEditors,
  activeEditorId,
  onSelectEditor,
  onCloseEditor,
}: EditorTabsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const hideTimerRef = useRef<number | null>(null);
  const hoveringRef = useRef(false);
  const [scrollbar, setScrollbar] = useState({
    overflow: false,
    visible: false,
    width: 0,
    left: 0,
  });

  const updateScrollbar = () => {
    const element = scrollRef.current;
    if (!element) return;
    const overflow = element.scrollWidth > element.clientWidth + 1;
    const width = overflow
      ? Math.max(24, (element.clientWidth / element.scrollWidth) * element.clientWidth)
      : 0;
    const maxScroll = Math.max(1, element.scrollWidth - element.clientWidth);
    const maxLeft = Math.max(0, element.clientWidth - width);
    const left = overflow ? (element.scrollLeft / maxScroll) * maxLeft : 0;

    setScrollbar((current) => ({
      ...current,
      overflow,
      width,
      left,
      visible: overflow && current.visible,
    }));
  };

  const showScrollbar = (temporary = false) => {
    updateScrollbar();
    setScrollbar((current) => ({
      ...current,
      visible: current.overflow || (scrollRef.current?.scrollWidth ?? 0) > (scrollRef.current?.clientWidth ?? 0),
    }));

    if (hideTimerRef.current !== null) {
      window.clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
    if (temporary && !hoveringRef.current) {
      hideTimerRef.current = window.setTimeout(() => {
        setScrollbar((current) => ({ ...current, visible: false }));
      }, 700);
    }
  };

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(updateScrollbar);
    observer.observe(element);
    updateScrollbar();

    return () => {
      observer.disconnect();
      if (hideTimerRef.current !== null) window.clearTimeout(hideTimerRef.current);
    };
  }, [openEditors]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    const activeTab = element?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!element || !activeTab) return;

    const before = element.scrollLeft;
    activeTab.scrollIntoView({ block: "nearest", inline: "nearest" });
    updateScrollbar();
    if (element.scrollLeft !== before) showScrollbar(true);
  }, [activeEditorId, openEditors]);

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    const element = scrollRef.current;
    if (!element || element.scrollWidth <= element.clientWidth) return;
    event.preventDefault();
    element.scrollLeft += event.deltaX !== 0 ? event.deltaX : event.deltaY;
    showScrollbar(true);
  };

  return (
    <div
      className="layer-tabs-shell"
      onMouseEnter={() => {
        hoveringRef.current = true;
        showScrollbar();
      }}
      onMouseLeave={() => {
        hoveringRef.current = false;
        showScrollbar(true);
      }}
    >
      <div
        ref={scrollRef}
        className="layer-tabs"
        role="tablist"
        aria-label="Open layers"
        onWheel={handleWheel}
        onScroll={() => {
          updateScrollbar();
          showScrollbar(true);
        }}
      >
        {openEditors.map((editor) => (
          <button
            key={editor.id}
            type="button"
            role="tab"
            aria-label={editorTabLabel(editor)}
            aria-selected={activeEditorId === editor.id}
            className={`layer-tab ${activeEditorId === editor.id ? "active" : ""}`}
            onClick={() => onSelectEditor(editor.id)}
          >
            <span className="tab-name">{editorTabLabel(editor)}</span>
            <span
              className="tab-close"
              aria-hidden="true"
              onClick={(event) => {
                event.stopPropagation();
                onCloseEditor(editor.id);
              }}
            />
          </button>
        ))}
      </div>
      {scrollbar.overflow && (
        <div
          className={`layer-tabs-scroll-thumb ${scrollbar.visible ? "visible" : ""}`}
          aria-hidden="true"
          style={{
            width: `${scrollbar.width}px`,
            transform: `translateX(${scrollbar.left}px)`,
          }}
        />
      )}
    </div>
  );
}
