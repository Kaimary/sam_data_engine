import React from "react";
import Tool from "./Tool";
import {
  CurrentMask,
  DatasetItem,
  InteractionMode,
  MaskEditStroke,
  ProgressSummary,
  RuntimeInfo,
  RuntimeMode,
  SavedMask,
  modelInputProps,
} from "./helpers/Interfaces";

interface StageProps {
  dataset: string;
  runtimeInfo: RuntimeInfo;
  runtimeMode: RuntimeMode;
  currentItem: DatasetItem | null;
  image: HTMLImageElement | null;
  savedMasks: SavedMask[];
  selectedMaskId: string | null;
  currentMask: CurrentMask | null;
  clicks: modelInputProps[];
  interactionMode: InteractionMode;
  brushSize: number;
  canUndo: boolean;
  progress: ProgressSummary;
  busyMessage: string | null;
  error: string | null;
  finished: boolean;
  onDatasetChange: (dataset: string) => void;
  onRuntimeChange: (runtime: RuntimeMode) => void;
  onModeChange: (mode: InteractionMode) => void;
  onBrushSizeChange: (size: number) => void;
  onImageClick: (click: modelInputProps, button: number) => void;
  onCreateBoxPrompt: (boxClicks: [modelInputProps, modelInputProps]) => void;
  onSelectSavedMask: (maskId: string) => void;
  onPromoteSavedMask: (maskId: string) => void;
  onRefineMask: (stroke: MaskEditStroke) => void;
  onUndo: () => void;
  onResetObject: () => void;
  onSaveMask: () => void;
  onRemoveLastMask: () => void;
  onClearSavedMasks: () => void;
  onCompleteImage: () => void;
  onSkipImage: () => void;
}

const Stage = ({
  dataset,
  runtimeInfo,
  runtimeMode,
  currentItem,
  image,
  savedMasks,
  selectedMaskId,
  currentMask,
  clicks,
  interactionMode,
  brushSize,
  canUndo,
  progress,
  busyMessage,
  error,
  finished,
  onDatasetChange,
  onRuntimeChange,
  onModeChange,
  onBrushSizeChange,
  onImageClick,
  onCreateBoxPrompt,
  onSelectSavedMask,
  onPromoteSavedMask,
  onRefineMask,
  onUndo,
  onResetObject,
  onSaveMask,
  onRemoveLastMask,
  onClearSavedMasks,
  onCompleteImage,
  onSkipImage,
}: StageProps) => {
  const annotationCount = savedMasks.length;

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <div className="panel-card hero-card">
          <div className="eyebrow">Assisted-manual data engine</div>
          <h1>SAM assisted annotation</h1>
          <p>
            Each image first gets full-image automatic masks from SAM, then the annotator can refine,
            add, or discard masks before moving to the next sample.
          </p>
        </div>

        <div className="panel-card runtime-card">
          <div className="section-header">
            <h2>Runtime</h2>
            <span>{runtimeMode}</span>
          </div>
          <div className="segmented-control">
            <button
              type="button"
              className={runtimeMode === "server" ? "active" : ""}
              onClick={() => onRuntimeChange("server")}
            >
              Server GPU
            </button>
            <button
              type="button"
              className={runtimeMode === "browser" ? "active" : ""}
              onClick={() => onRuntimeChange("browser")}
            >
              In-browser
            </button>
          </div>
          <div className="runtime-chip-row">
            <span className={`runtime-chip ${runtimeMode === "browser" ? "webgpu" : "wasm"}`}>
              {runtimeMode === "browser" ? "BROWSER" : "BACKEND"}
            </span>
            <p className="helper-copy">
              Interactive prompts run via {runtimeInfo.interactiveLabel}. Automatic masks still come from{" "}
              {runtimeInfo.automaticMaskLabel}.
            </p>
          </div>
        </div>

        <div className="panel-card">
          <div className="section-header">
            <h2>Dataset</h2>
            <span>{progress.pending} pending</span>
          </div>
          <div className="segmented-control">
            <button
              type="button"
              className={dataset === "diagram" ? "active" : ""}
              onClick={() => onDatasetChange("diagram")}
            >
              Diagram
            </button>
            <button
              type="button"
              className={dataset === "plot" ? "active" : ""}
              onClick={() => onDatasetChange("plot")}
            >
              Plot
            </button>
          </div>
          <div className="progress-grid">
            <div>
              <strong>{progress.total}</strong>
              <span>Total</span>
            </div>
            <div>
              <strong>{progress.completed}</strong>
              <span>Completed</span>
            </div>
            <div>
              <strong>{progress.skipped}</strong>
              <span>Skipped</span>
            </div>
            <div>
              <strong>{annotationCount}</strong>
              <span>Saved masks</span>
            </div>
          </div>
        </div>

        <div className="panel-card">
          <div className="section-header">
            <h2>Tools</h2>
            <span>{clicks.length} clicks</span>
          </div>
          <div className="segmented-control compact">
            <button
              type="button"
              className={interactionMode === "positive" ? "active positive" : "positive"}
              onClick={() => onModeChange("positive")}
            >
              Positive
            </button>
            <button
              type="button"
              className={interactionMode === "negative" ? "active negative" : "negative"}
              onClick={() => onModeChange("negative")}
            >
              Negative
            </button>
            <button
              type="button"
              className={interactionMode === "box" ? "active brush" : "brush"}
              onClick={() => onModeChange("box")}
            >
              Box
            </button>
            <button
              type="button"
              className={interactionMode === "brush" ? "active brush" : "brush"}
              onClick={() => onModeChange("brush")}
              disabled={!currentMask}
            >
              Brush
            </button>
            <button
              type="button"
              className={interactionMode === "eraser" ? "active eraser" : "eraser"}
              onClick={() => onModeChange("eraser")}
              disabled={!currentMask}
            >
              Eraser
            </button>
          </div>
          <p className="helper-copy">
            Point modes add click prompts. Box mode drags a rectangle to generate a mask. Double-click a编号
            or press Enter to switch the selected mask into brush editing. `Ctrl+Z` / `Cmd+Z` undoes the
            last edit or prompt action.
          </p>
          {interactionMode === "brush" || interactionMode === "eraser" ? (
            <label className="brush-size-control">
              <span>Brush size</span>
              <input
                type="range"
                min={2}
                max={48}
                step={1}
                value={brushSize}
                onChange={(event) => onBrushSizeChange(Number(event.target.value))}
              />
              <strong>{brushSize}px</strong>
            </label>
          ) : null}
          <div className="button-row">
            <button type="button" onClick={onUndo} disabled={!canUndo}>
              Undo
            </button>
            <button type="button" onClick={onResetObject} disabled={clicks.length === 0 && !currentMask}>
              Reset object
            </button>
          </div>
          <div className="button-row vertical">
            <button type="button" className="primary" onClick={onSaveMask} disabled={!currentMask}>
              Save current mask
            </button>
            <button
              type="button"
              className="ghost"
              onClick={onRemoveLastMask}
              disabled={savedMasks.length === 0}
            >
              Remove last saved mask
            </button>
            <button
              type="button"
              className="ghost"
              onClick={onClearSavedMasks}
              disabled={savedMasks.length === 0}
            >
              Clear loaded masks
            </button>
          </div>
        </div>

        <div className="panel-card">
          <div className="section-header">
            <h2>Image actions</h2>
            <span>{currentItem ? `${currentItem.index + 1}/${currentItem.total}` : "-"}</span>
          </div>
          <div className="button-row vertical">
            <button type="button" className="primary warm" onClick={onCompleteImage} disabled={!currentItem}>
              Finish image and next
            </button>
            <button type="button" className="ghost" onClick={onSkipImage} disabled={!currentItem}>
              Skip image
            </button>
          </div>
        </div>

        <div className="panel-card meta-card">
          <div className="section-header">
            <h2>Current item</h2>
            <span>{currentItem?.split || "unspecified"}</span>
          </div>
          {finished ? (
            <p>All items in this dataset have been processed.</p>
          ) : currentItem ? (
            <>
              <h3>{currentItem.title}</h3>
              <ul>
                <li>{currentItem.filename}</li>
                <li>
                  {currentItem.width} x {currentItem.height}
                </li>
                {currentItem.methodSectionTitle ? <li>{currentItem.methodSectionTitle}</li> : null}
                {currentItem.visualIntent ? <li>{currentItem.visualIntent}</li> : null}
              </ul>
            </>
          ) : (
            <p>Loading next sample...</p>
          )}
        </div>

        {busyMessage ? <div className="status-banner busy">{busyMessage}</div> : null}
        {error ? <div className="status-banner error">{error}</div> : null}
      </aside>

      <main className="workspace">
        <div className="workspace-header">
          <div>
            <div className="eyebrow">Interactive canvas</div>
            <h2>{currentItem ? currentItem.title : "Preparing item"}</h2>
          </div>
          <div className="workspace-legend">
            <span className="legend-chip positive">Positive</span>
            <span className="legend-chip negative">Negative</span>
            <span className="legend-chip saved">Saved masks</span>
          </div>
        </div>
        <div className="workspace-stage">
          <Tool
            image={image}
            savedMasks={savedMasks}
            selectedMaskId={selectedMaskId}
            currentMask={currentMask}
            clicks={clicks}
            interactionMode={interactionMode}
            brushSize={brushSize}
            disabled={Boolean(busyMessage) || finished}
            onImageClick={onImageClick}
            onCreateBoxPrompt={onCreateBoxPrompt}
            onSelectSavedMask={onSelectSavedMask}
            onPromoteSavedMask={onPromoteSavedMask}
            onRefineMask={onRefineMask}
          />
        </div>
      </main>
    </div>
  );
};

export default Stage;
