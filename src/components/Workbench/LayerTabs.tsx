import { useEffect, useLayoutEffect, useRef, useState, type WheelEvent } from "react";

interface LayerTabsProps {
  openLayers: number[];
  activeLayerIndex: number | null;
  onSelectLayer: (layerIndex: number) => void;
  onCloseLayer: (layerIndex: number) => void;
}

export function LayerTabs({
  openLayers,
  activeLayerIndex,
  onSelectLayer,
  onCloseLayer,
}: LayerTabsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const hideTimerRef = useRef<number | null>(null);
  const hoveringRef = useRef(false);
  const [scrollbar, setScrollbar] = useState({
    overflow: false,
    visible: false,
    width: 0,
    left: 0,
  });

  const labelFor = (layerIndex: number) =>
    layerIndex < 0 ? "Global tensors" : `Layer ${layerIndex}`;

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
  }, [openLayers]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    const activeTab = element?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!element || !activeTab) return;

    const before = element.scrollLeft;
    activeTab.scrollIntoView({ block: "nearest", inline: "nearest" });
    updateScrollbar();
    if (element.scrollLeft !== before) showScrollbar(true);
  }, [activeLayerIndex, openLayers]);

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
        {openLayers.map((layerIndex) => (
          <button
            key={layerIndex}
            type="button"
            role="tab"
            aria-label={labelFor(layerIndex)}
            aria-selected={activeLayerIndex === layerIndex}
            className={`layer-tab ${activeLayerIndex === layerIndex ? "active" : ""}`}
            onClick={() => onSelectLayer(layerIndex)}
          >
            <span className="tab-name">{labelFor(layerIndex)}</span>
            <span
              className="tab-close"
              aria-hidden="true"
              onClick={(event) => {
                event.stopPropagation();
                onCloseLayer(layerIndex);
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
