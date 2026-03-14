import React from "react";
import { ToolProps } from "./helpers/Interfaces";

const Tool = ({
  image,
  savedMasks,
  currentMask,
  clicks,
  clickMode,
  disabled = false,
  onAddClick,
}: ToolProps) => {
  const handlePointer = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!image || disabled) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * image.naturalWidth;
    const y = ((event.clientY - rect.top) / rect.height) * image.naturalHeight;
    const clickType = event.button === 2 ? 0 : clickMode === "positive" ? 1 : 0;
    onAddClick({ x, y, clickType });
  };

  return (
    <div
      className="tool-stage"
      onClick={handlePointer}
      onContextMenu={handlePointer}
      role="presentation"
    >
      {image ? (
        <div className="tool-frame">
          <img src={image.src} alt={image.alt || "annotation target"} className="tool-image" />
          {savedMasks.map((mask) => (
            <img key={mask.id} src={mask.overlayUrl} alt="saved mask" className="tool-mask saved" />
          ))}
          {currentMask ? (
            <img src={currentMask.overlayUrl} alt="current mask" className="tool-mask current" />
          ) : null}
          <div className="tool-points">
            {clicks.map((click, index) => {
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
