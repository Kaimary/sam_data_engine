import React from "react";
import Tool from "./Tool";
import {
  CurrentMask,
  DatasetItem,
  ProgressSummary,
  SavedMask,
  modelInputProps,
} from "./helpers/Interfaces";

interface StageProps {
  dataset: string;
  currentItem: DatasetItem | null;
  image: HTMLImageElement | null;
  savedMasks: SavedMask[];
  currentMask: CurrentMask | null;
  clicks: modelInputProps[];
  clickMode: "positive" | "negative";
  progress: ProgressSummary;
  busyMessage: string | null;
  error: string | null;
  finished: boolean;
  onDatasetChange: (dataset: string) => void;
  onModeChange: (mode: "positive" | "negative") => void;
  onAddClick: (click: modelInputProps) => void;
  onUndo: () => void;
  onResetObject: () => void;
  onSaveMask: () => void;
  onRemoveLastMask: () => void;
  onCompleteImage: () => void;
  onSkipImage: () => void;
}

const Stage = ({
  dataset,
  currentItem,
  image,
  savedMasks,
  currentMask,
  clicks,
  clickMode,
  progress,
  busyMessage,
  error,
  finished,
  onDatasetChange,
  onModeChange,
  onAddClick,
  onUndo,
  onResetObject,
  onSaveMask,
  onRemoveLastMask,
  onCompleteImage,
  onSkipImage,
}: StageProps) => {
  const annotationCount = savedMasks.length + (currentMask ? 1 : 0);

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <div className="panel-card hero-card">
          <div className="eyebrow">Assisted-manual data engine</div>
          <h1>SAM assisted annotation</h1>
          <p>
            Follow the paper&apos;s first-stage workflow: click to propose a mask, save object masks,
            then finish the image and move to the next sample automatically.
          </p>
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
              <span>Current masks</span>
            </div>
          </div>
        </div>

        <div className="panel-card">
          <div className="section-header">
            <h2>Prompt mode</h2>
            <span>{clicks.length} clicks</span>
          </div>
          <div className="segmented-control compact">
            <button
              type="button"
              className={clickMode === "positive" ? "active positive" : "positive"}
              onClick={() => onModeChange("positive")}
            >
              Positive
            </button>
            <button
              type="button"
              className={clickMode === "negative" ? "active negative" : "negative"}
              onClick={() => onModeChange("negative")}
            >
              Negative
            </button>
          </div>
          <p className="helper-copy">
            Left click uses the selected mode. Right click always places a negative point.
          </p>
          <div className="button-row">
            <button type="button" onClick={onUndo} disabled={clicks.length === 0}>
              Undo click
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
            currentMask={currentMask}
            clicks={clicks}
            clickMode={clickMode}
            disabled={Boolean(busyMessage) || finished}
            onAddClick={onAddClick}
          />
        </div>
      </main>
    </div>
  );
};

export default Stage;
