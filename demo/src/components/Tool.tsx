import React, { useMemo, useRef, useState } from "react";
import { ToolProps } from "./helpers/Interfaces";

const Tool = ({
  image,
  savedMasks,
  selectedMaskId,
  currentMask,
  clicks,
  interactionMode,
  brushSize,
  disabled = false,
  onImageClick,
  onCreateBoxPrompt,
  onSelectSavedMask,
  onPromoteSavedMask,
  onRefineMask,
}: ToolProps) => {
  const isPaintingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const strokeIdRef = useRef(0);
  const boxStartRef = useRef<{ x: number; y: number } | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const editingWithBrush = Boolean(
    image && currentMask && (interactionMode === "brush" || interactionMode === "eraser"),
  );
  const [draftBox, setDraftBox] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);

  const paintingCursor = useMemo(() => {
    if (!editingWithBrush) {
      return undefined;
    }

    const radius = Math.max(2, Math.min(brushSize, 48));
    const padding = 8;
    const size = radius * 2 + padding * 2;
    const center = size / 2;
    const stroke = interactionMode === "brush" ? "%23ffffff" : "%23ffffff";
    const fill = interactionMode === "brush" ? "rgba(42,157,143,0.16)" : "rgba(209,73,91,0.14)";
    const dash = interactionMode === "eraser" ? " stroke-dasharray='6 4'" : "";
    const svg = `
      <svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}' viewBox='0 0 ${size} ${size}'>
        <circle cx='${center}' cy='${center}' r='${radius}' fill='${fill}' stroke='${stroke}' stroke-width='2'${dash}/>
        <circle cx='${center}' cy='${center}' r='2.2' fill='white' stroke='%231e2a2f' stroke-width='0.8'/>
      </svg>
    `;
    const encoded = `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}") ${center} ${center}, none`;
    return encoded;
  }, [brushSize, editingWithBrush, interactionMode]);

  const getImagePoint = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!image || !frameRef.current) {
      return null;
    }
    const rect = frameRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }
    return {
      x: ((event.clientX - rect.left) / rect.width) * image.naturalWidth,
      y: ((event.clientY - rect.top) / rect.height) * image.naturalHeight,
    };
  };

  const stopPainting = () => {
    isPaintingRef.current = false;
    lastPointRef.current = null;
  };

  const clearDraftBox = () => {
    boxStartRef.current = null;
    setDraftBox(null);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!image || disabled) {
      return;
    }

    const point = getImagePoint(event);
    if (!point) {
      return;
    }

    if (interactionMode === "box") {
      if (event.button !== 0) {
        return;
      }
      boxStartRef.current = point;
      setDraftBox({
        x: point.x,
        y: point.y,
        width: 0,
        height: 0,
      });
      return;
    }

    if (interactionMode === "brush" || interactionMode === "eraser") {
      if (!currentMask || event.button !== 0) {
        return;
      }
      const paintTool = interactionMode;
      strokeIdRef.current += 1;
      isPaintingRef.current = true;
      lastPointRef.current = point;
      onRefineMask({
        fromX: point.x,
        fromY: point.y,
        toX: point.x,
        toY: point.y,
        tool: paintTool,
        radius: brushSize,
        strokeId: strokeIdRef.current,
      });
      return;
    }

    const clickType = event.button === 2 ? 0 : interactionMode === "positive" ? 1 : 0;
    onImageClick({ x: point.x, y: point.y, clickType }, event.button);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const point = getImagePoint(event);
    if (interactionMode === "box" && boxStartRef.current && point) {
      const x0 = Math.min(boxStartRef.current.x, point.x);
      const y0 = Math.min(boxStartRef.current.y, point.y);
      const x1 = Math.max(boxStartRef.current.x, point.x);
      const y1 = Math.max(boxStartRef.current.y, point.y);
      setDraftBox({
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
      });
      return;
    }
    if (!isPaintingRef.current || !currentMask || disabled) {
      return;
    }
    const lastPoint = lastPointRef.current;
    if (!point || !lastPoint) {
      return;
    }
    if (interactionMode !== "brush" && interactionMode !== "eraser") {
      return;
    }
    onRefineMask({
      fromX: lastPoint.x,
      fromY: lastPoint.y,
      toX: point.x,
      toY: point.y,
      tool: interactionMode,
      radius: brushSize,
      strokeId: strokeIdRef.current,
    });
    lastPointRef.current = point;
  };

  const handleMouseLeave = () => {
    stopPainting();
    clearDraftBox();
  };

  const handleMouseUp = (event: React.MouseEvent<HTMLDivElement>) => {
    if (interactionMode === "box" && boxStartRef.current) {
      const point = getImagePoint(event);
      const start = boxStartRef.current;
      clearDraftBox();
      if (!point) {
        return;
      }
      const x0 = Math.min(start.x, point.x);
      const y0 = Math.min(start.y, point.y);
      const x1 = Math.max(start.x, point.x);
      const y1 = Math.max(start.y, point.y);
      if (x1 - x0 < 4 || y1 - y0 < 4) {
        return;
      }
      onCreateBoxPrompt([
        { x: x0, y: y0, clickType: 2 },
        { x: x1, y: y1, clickType: 3 },
      ]);
      return;
    }

    stopPainting();
  };

  const boxPrompt = clicks.length >= 2
    ? (() => {
        const topLeft = clicks.find((click) => click.clickType === 2);
        const bottomRight = clicks.find((click) => click.clickType === 3);
        if (!topLeft || !bottomRight || !image) {
          return null;
        }
        return {
          left: `${(topLeft.x / image.naturalWidth) * 100}%`,
          top: `${(topLeft.y / image.naturalHeight) * 100}%`,
          width: `${((bottomRight.x - topLeft.x) / image.naturalWidth) * 100}%`,
          height: `${((bottomRight.y - topLeft.y) / image.naturalHeight) * 100}%`,
        };
      })()
    : null;

  return (
    <div
      className={`tool-stage ${interactionMode === "brush" || interactionMode === "eraser" ? "painting" : interactionMode === "box" ? "boxing" : ""}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onContextMenu={(event) => event.preventDefault()}
      style={{ cursor: paintingCursor }}
      role="presentation"
    >
      {image ? (
        <div ref={frameRef} className="tool-frame" style={{ cursor: paintingCursor }}>
          <img src={image.src} alt={image.alt || "annotation target"} className="tool-image" />
          {savedMasks.map((mask) => (
            <img
              key={mask.id}
              src={mask.overlayUrl}
              alt="saved mask"
              className={`tool-mask saved ${selectedMaskId === mask.id ? "selected" : ""}`}
            />
          ))}
          <div className="tool-mask-annotations">
            {savedMasks.map((mask, index) => {
              if (!mask.bbox || mask.bbox[2] <= 0 || mask.bbox[3] <= 0) {
                return null;
              }
              const [x, y, width, height] = mask.bbox;
              const left = `${(x / image.naturalWidth) * 100}%`;
              const top = `${(y / image.naturalHeight) * 100}%`;
              const boxWidth = `${(width / image.naturalWidth) * 100}%`;
              const boxHeight = `${(height / image.naturalHeight) * 100}%`;
              return (
                <React.Fragment key={`${mask.id}-annotation`}>
                  <div
                    className={`mask-bbox ${selectedMaskId === mask.id ? "selected" : ""}`}
                    style={{
                      left,
                      top,
                      width: boxWidth,
                      height: boxHeight,
                      borderColor: mask.color,
                    }}
                  />
                  <button
                    type="button"
                    className={`mask-badge ${selectedMaskId === mask.id ? "selected" : ""}`}
                    style={{
                      left,
                      top,
                      backgroundColor: mask.color,
                    }}
                    title={`Mask ${index + 1}${mask.area ? ` • area ${mask.area}` : ""}`}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onSelectSavedMask(mask.id);
                    }}
                    onDoubleClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onPromoteSavedMask(mask.id);
                    }}
                  >
                    {index + 1}
                  </button>
                </React.Fragment>
              );
            })}
          </div>
          {currentMask ? (
            <img src={currentMask.overlayUrl} alt="current mask" className="tool-mask current" />
          ) : null}
          {boxPrompt ? <div className="prompt-box committed" style={boxPrompt} /> : null}
          {draftBox && image ? (
            <div
              className="prompt-box draft"
              style={{
                left: `${(draftBox.x / image.naturalWidth) * 100}%`,
                top: `${(draftBox.y / image.naturalHeight) * 100}%`,
                width: `${(draftBox.width / image.naturalWidth) * 100}%`,
                height: `${(draftBox.height / image.naturalHeight) * 100}%`,
              }}
            />
          ) : null}
          <div className="tool-points">
            {clicks.map((click, index) => {
              if (click.clickType === 2 || click.clickType === 3) {
                return null;
              }
              const left = `${(click.x / image.naturalWidth) * 100}%`;
              const top = `${(click.y / image.naturalHeight) * 100}%`;
              const className = click.clickType === 1 ? "point positive" : "point negative";
              return (
                <span
                  key={`${click.x}-${click.y}-${index}`}
                  className={className}
                  style={{ left, top }}
                  title={click.clickType === 1 ? "positive click" : "negative click"}
                >
                  {index + 1}
                </span>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="tool-empty">Loading image...</div>
      )}
    </div>
  );
};

export default Tool;
