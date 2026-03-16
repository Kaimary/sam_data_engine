import React, { useEffect, useMemo, useRef, useState } from "react";
import "./assets/scss/App.scss";
import Stage from "./components/Stage";
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
} from "./components/helpers/Interfaces";
import {
  getBrowserRuntimeLabel,
  runBrowserPrediction,
} from "./components/helpers/browserPredictor";
import { applyBrushStroke, createMaskLayer, maskDataUrlToBinaryMask } from "./components/helpers/maskUtils";

const DEFAULT_DATASET = "diagram";
const MASK_PALETTE = ["#ff7a59", "#5abf90", "#4fa3ff", "#f6bd60", "#9d7bff", "#f28482"];
const PREVIEW_COLOR = "#2e86ab";

const emptyProgress: ProgressSummary = {
  total: 0,
  completed: 0,
  skipped: 0,
  pending: 0,
};

const DEFAULT_RUNTIME_INFO: RuntimeInfo = {
  mode: "server",
  defaultMode: "server",
  location: "server",
  device: "cpu",
  label: "Server",
  interactiveLabel: "Server",
  automaticMaskLabel: "Server",
  modelUrl: "/api/runtime/browser-model.onnx",
};

type UndoActionType = "prompt" | "edit" | "delete" | "save";

interface DeletedMaskSnapshot {
  index: number;
  mask: SavedMask;
}

interface SavedMaskSnapshot {
  index: number;
  mask: SavedMask;
  restoredCurrentMask: CurrentMask;
  restoredClicks: modelInputProps[];
  restoredInteractionMode: InteractionMode;
}

const getRequestedRuntimeMode = (): RuntimeMode | null => {
  if (typeof window === "undefined") {
    return null;
  }
  const runtime = new URLSearchParams(window.location.search).get("runtime");
  return runtime === "browser" || runtime === "server" ? runtime : null;
};

const syncRuntimeSearchParam = (mode: RuntimeMode) => {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("runtime", mode);
  window.history.replaceState({}, "", url.toString());
};

const App = () => {
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo>(DEFAULT_RUNTIME_INFO);
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("server");
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [progress, setProgress] = useState<ProgressSummary>(emptyProgress);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("positive");
  const [brushSize, setBrushSize] = useState(14);
  const [clicks, setClicks] = useState<modelInputProps[]>([]);
  const [currentMask, setCurrentMask] = useState<CurrentMask | null>(null);
  const [editHistory, setEditHistory] = useState<CurrentMask[]>([]);
  const [deleteHistory, setDeleteHistory] = useState<DeletedMaskSnapshot[]>([]);
  const [saveHistory, setSaveHistory] = useState<SavedMaskSnapshot[]>([]);
  const [undoOrder, setUndoOrder] = useState<UndoActionType[]>([]);
  const [savedMasks, setSavedMasks] = useState<SavedMask[]>([]);
  const [selectedMaskId, setSelectedMaskId] = useState<string | null>(null);
  const [busyMessage, setBusyMessage] = useState<string | null>("Loading dataset...");
  const [error, setError] = useState<string | null>(null);

  const loadCounterRef = useRef(0);
  const predictionCounterRef = useRef(0);
  const lastStrokeIdRef = useRef<number | null>(null);
  const imageCacheRef = useRef<Map<string, Promise<HTMLImageElement>>>(new Map());
  const startupRuntimeModeRef = useRef<RuntimeMode | null>(getRequestedRuntimeMode());

  const finished = useMemo(() => !currentItem && progress.pending === 0 && progress.total > 0, [currentItem, progress]);

  const cloneMask = (mask: CurrentMask): CurrentMask => ({
    ...mask,
    clicks: mask.clicks.map((click) => ({ ...click })),
    bbox: mask.bbox ? [...mask.bbox] as [number, number, number, number] : undefined,
    binaryMask: new Uint8ClampedArray(mask.binaryMask),
  });

  const cloneSavedMask = (mask: SavedMask): SavedMask => ({
    ...mask,
    clicks: mask.clicks.map((click) => ({ ...click })),
    bbox: mask.bbox ? ([...mask.bbox] as [number, number, number, number]) : undefined,
    binaryMask: mask.binaryMask ? new Uint8ClampedArray(mask.binaryMask) : undefined,
  });

  const keepPersistentUndoOnly = () => {
    setEditHistory([]);
    setUndoOrder((previous) => previous.filter((action) => action === "delete" || action === "save"));
  };

  const promoteSavedMaskToCurrent = async (maskId: string) => {
    const savedMask = savedMasks.find((mask) => mask.id === maskId);
    if (!savedMask) {
      return;
    }

    let width = savedMask.width;
    let height = savedMask.height;
    let binaryMask = savedMask.binaryMask;

    if (!width || !height || !binaryMask) {
      setBusyMessage("Preparing editable mask...");
      const decodedMask = await maskDataUrlToBinaryMask(savedMask.maskPngDataUrl);
      width = decodedMask.width;
      height = decodedMask.height;
      binaryMask = decodedMask.binary;
    }

    lastStrokeIdRef.current = null;
    keepPersistentUndoOnly();
    setError(null);
    setClicks([]);
    setSelectedMaskId(null);
    setSavedMasks((previous) => previous.filter((mask) => mask.id !== maskId));
    setCurrentMask({
      id: `editable-${savedMask.id}`,
      overlayUrl: savedMask.overlayUrl,
      maskPngDataUrl: savedMask.maskPngDataUrl,
      color: savedMask.color,
      clicks: savedMask.clicks.map((click) => ({ ...click })),
      area: savedMask.area,
      bbox: savedMask.bbox,
      width,
      height,
      binaryMask: new Uint8ClampedArray(binaryMask),
      source: savedMask.source || "manual",
      score: savedMask.score ?? null,
    });
    setInteractionMode("brush");
    setBusyMessage(null);
  };

  const clearCurrentObject = () => {
    lastStrokeIdRef.current = null;
    setClicks([]);
    setCurrentMask(null);
    keepPersistentUndoOnly();
    setSelectedMaskId(null);
    setInteractionMode("positive");
  };

  const clearImageState = () => {
    lastStrokeIdRef.current = null;
    setImage(null);
    setClicks([]);
    setCurrentMask(null);
    setEditHistory([]);
    setDeleteHistory([]);
    setSaveHistory([]);
    setUndoOrder([]);
    setSavedMasks([]);
    setSelectedMaskId(null);
    setInteractionMode("positive");
  };

  const loadImageElement = (source: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${source}`));
      img.src = source;
    });

  const itemCacheKey = (item: DatasetItem) => `${item.dataset}:${item.id}`;

  const pruneCache = <T,>(cache: Map<string, Promise<T>>, keepKeys: string[]) => {
    for (const key of cache.keys()) {
      if (!keepKeys.includes(key)) {
        cache.delete(key);
      }
    }
  };

  const loadCachedImage = (item: DatasetItem) => {
    const key = itemCacheKey(item);
    const cached = imageCacheRef.current.get(key);
    if (cached) {
      return cached;
    }
    const nextPromise = loadImageElement(item.imageUrl);
    imageCacheRef.current.set(key, nextPromise);
    return nextPromise;
  };

  const fetchPeekNext = async (current: DatasetItem) => {
    const response = await fetch(`/api/items/${current.id}/peek-next?dataset=${current.dataset}`);
    if (!response.ok) {
      throw new Error(`Failed to peek next item for ${current.id}`);
    }
    return response.json();
  };

  const fetchAutomaticMasks = async (item: DatasetItem) => {
    const response = await fetch(`/api/items/${item.id}/automatic-masks?dataset=${item.dataset}`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Failed to generate automatic masks for ${item.id}`);
    }
    return response.json();
  };

  const applyNextPayload = async (nextDataset: string, payload: any) => {
    setDataset(nextDataset);
    setProgress(payload.progress || emptyProgress);
    const nextRuntime = (payload.runtime || DEFAULT_RUNTIME_INFO) as RuntimeInfo;
    setRuntimeInfo(nextRuntime);
    setRuntimeMode(nextRuntime.mode);
    clearImageState();
    setError(null);

    if (!payload.item) {
      setCurrentItem(null);
      setBusyMessage(null);
      return;
    }

    const loadId = loadCounterRef.current + 1;
    loadCounterRef.current = loadId;
    setBusyMessage("Loading image...");
    const nextItem = payload.item as DatasetItem;
    setCurrentItem(nextItem);

    try {
      const nextImage = await loadCachedImage(nextItem);
      if (loadCounterRef.current !== loadId) {
        return;
      }
      setImage(nextImage);
      pruneCache(imageCacheRef.current, [itemCacheKey(nextItem)]);

      setBusyMessage("Generating automatic masks...");
      const automaticPayload = await fetchAutomaticMasks(nextItem);
      if (loadCounterRef.current !== loadId) {
        return;
      }
      const automaticMasks = Array.isArray(automaticPayload.masks) ? automaticPayload.masks : [];
      const hydratedMasks = await Promise.all(
        automaticMasks.map(async (mask: any, index: number) => {
          const decodedMask = await maskDataUrlToBinaryMask(mask.maskPngDataUrl);
          return {
            id: mask.id || `${nextItem.id}-auto-${index + 1}`,
            overlayUrl: mask.overlayUrl,
            maskPngDataUrl: mask.maskPngDataUrl,
            color: mask.color || MASK_PALETTE[index % MASK_PALETTE.length],
            clicks: Array.isArray(mask.clicks) ? mask.clicks : [],
            area: mask.area || 0,
            bbox: Array.isArray(mask.bbox) ? mask.bbox : undefined,
            width: decodedMask.width,
            height: decodedMask.height,
            binaryMask: decodedMask.binary,
            source: mask.source || "automatic",
            score: typeof mask.score === "number" ? mask.score : null,
          } as SavedMask;
        }),
      );
      if (loadCounterRef.current !== loadId) {
        return;
      }
      setSavedMasks(hydratedMasks);
      setBusyMessage(null);
    } catch (loadError: any) {
      if (loadCounterRef.current !== loadId) {
        return;
      }
      setError(loadError.message || "Failed to load item assets.");
      setBusyMessage(null);
    }
  };

  const fetchBootstrap = async (nextDataset: string, nextRuntimeMode?: RuntimeMode | null) => {
    const search = new URLSearchParams({ dataset: nextDataset });
    const requestedRuntime = nextRuntimeMode === undefined ? runtimeMode : nextRuntimeMode;
    if (requestedRuntime) {
      search.set("runtime", requestedRuntime);
    }
    const response = await fetch(`/api/bootstrap?${search.toString()}`);
    if (!response.ok) {
      throw new Error(`Failed to bootstrap dataset ${nextDataset}`);
    }
    return response.json();
  };

  useEffect(() => {
    const init = async () => {
      try {
        const payload = await fetchBootstrap(DEFAULT_DATASET, startupRuntimeModeRef.current);
        await applyNextPayload(DEFAULT_DATASET, payload);
        const nextMode = (payload.runtime?.mode || DEFAULT_RUNTIME_INFO.mode) as RuntimeMode;
        syncRuntimeSearchParam(nextMode);
      } catch (initError: any) {
        setError(initError.message || "Failed to initialize the app.");
        setBusyMessage(null);
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (!currentItem) {
      pruneCache(imageCacheRef.current, []);
      return;
    }

    let cancelled = false;
    const currentKey = itemCacheKey(currentItem);

    const prefetchNext = async () => {
      try {
        const payload = await fetchPeekNext(currentItem);
        if (cancelled) {
          return;
        }
        const nextItem = payload.item as DatasetItem | null;
        const keepKeys = [currentKey];
        if (nextItem) {
          const nextKey = itemCacheKey(nextItem);
          keepKeys.push(nextKey);
          void loadCachedImage(nextItem).catch(() => undefined);
        }
        pruneCache(imageCacheRef.current, keepKeys);
      } catch {
        pruneCache(imageCacheRef.current, [currentKey]);
      }
    };

    prefetchNext();

    return () => {
      cancelled = true;
    };
  }, [currentItem]);

  useEffect(() => {
    if (!selectedMaskId) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement as HTMLElement | null;
      const tagName = activeElement?.tagName;
      if (tagName === "INPUT" || tagName === "TEXTAREA") {
        return;
      }

      if (event.key === "Enter") {
        event.preventDefault();
        void promoteSavedMaskToCurrent(selectedMaskId);
        return;
      }

      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        setSavedMasks((previous) => previous.filter((mask) => mask.id !== selectedMaskId));
        const deletedIndex = savedMasks.findIndex((mask) => mask.id === selectedMaskId);
        if (deletedIndex >= 0) {
          setDeleteHistory((previous) => [
            ...previous,
            {
              index: deletedIndex,
              mask: cloneSavedMask(savedMasks[deletedIndex]),
            },
          ]);
          setUndoOrder((previous) => [...previous, "delete"]);
        }
        setSelectedMaskId(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [selectedMaskId, savedMasks]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") {
        return;
      }
      const activeElement = document.activeElement as HTMLElement | null;
      const tagName = activeElement?.tagName;
      if (tagName === "INPUT" || tagName === "TEXTAREA") {
        return;
      }

      if (undoOrder.length > 0 || clicks.length > 0) {
        event.preventDefault();
        handleUndo();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [clicks.length, undoOrder.length]);

  useEffect(() => {
    const runPrediction = async () => {
      if (!currentItem || clicks.length === 0) {
        return;
      }
      const requestId = predictionCounterRef.current + 1;
      predictionCounterRef.current = requestId;
      try {
        setError(null);
        let result: {
          overlayUrl: string;
          maskPngDataUrl: string;
          area: number;
          bbox: [number, number, number, number];
          score: number | null;
          width: number;
          height: number;
        };

        if (runtimeMode === "browser") {
          const browserResult = await runBrowserPrediction(runtimeInfo, currentItem, clicks);
          if (predictionCounterRef.current !== requestId) {
            return;
          }
          const nextLabel = getBrowserRuntimeLabel(browserResult.provider);
          setRuntimeInfo((previous) => ({
            ...previous,
            label: nextLabel,
            interactiveLabel: nextLabel,
            device: browserResult.provider,
          }));
          const layer = createMaskLayer(browserResult.mask, currentItem.width, currentItem.height, PREVIEW_COLOR);
          result = {
            ...layer,
            score: browserResult.score,
            width: currentItem.width,
            height: currentItem.height,
          };
        } else {
          const response = await fetch(`/api/items/${currentItem.id}/predict?dataset=${dataset}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              clicks,
              color: PREVIEW_COLOR,
            }),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
          result = await response.json();
        }

        if (predictionCounterRef.current !== requestId) {
          return;
        }
        const decodedMask = await maskDataUrlToBinaryMask(result.maskPngDataUrl);
        if (predictionCounterRef.current !== requestId) {
          return;
        }
        setCurrentMask({
          id: "preview",
          overlayUrl: result.overlayUrl,
          maskPngDataUrl: result.maskPngDataUrl,
          color: PREVIEW_COLOR,
          clicks: clicks.map((click) => ({ ...click })),
          area: result.area,
          bbox: result.bbox,
          source: "manual",
          score: typeof result.score === "number" ? result.score : null,
          width: decodedMask.width,
          height: decodedMask.height,
          binaryMask: decodedMask.binary,
        });
        lastStrokeIdRef.current = null;
        keepPersistentUndoOnly();
      } catch (predictionError: any) {
        setError(predictionError.message || "Failed to run the ONNX model.");
      }
    };

    if (clicks.length === 0) {
      setCurrentMask((previous) => (previous?.id === "preview" ? null : previous));
      return;
    }

    runPrediction();
  }, [clicks, currentItem, dataset, runtimeInfo.modelUrl, runtimeMode]);

  const handleDatasetChange = async (nextDataset: string) => {
    if (nextDataset === dataset) {
      return;
    }
    try {
      setBusyMessage("Switching dataset...");
      const payload = await fetchBootstrap(nextDataset, runtimeMode);
      await applyNextPayload(nextDataset, payload);
    } catch (datasetError: any) {
      setError(datasetError.message || "Failed to switch dataset.");
      setBusyMessage(null);
    }
  };

  const handleRuntimeChange = async (nextRuntimeMode: RuntimeMode) => {
    if (nextRuntimeMode === runtimeMode) {
      return;
    }
    try {
      setBusyMessage(`Switching to ${nextRuntimeMode === "browser" ? "browser" : "server"} runtime...`);
      const payload = await fetchBootstrap(dataset, nextRuntimeMode);
      await applyNextPayload(dataset, payload);
      syncRuntimeSearchParam(nextRuntimeMode);
    } catch (runtimeError: any) {
      setError(runtimeError.message || "Failed to switch runtime.");
      setBusyMessage(null);
    }
  };

  const handleImageClick = (click: modelInputProps, button: number) => {
    setError(null);
    setSelectedMaskId(null);
    lastStrokeIdRef.current = null;
    keepPersistentUndoOnly();
    const nextClick = {
      ...click,
      clickType: button === 2 ? 0 : click.clickType,
    };
    setUndoOrder((previous) => [...previous, "prompt"]);
    setClicks((previous) => [...previous, nextClick]);
  };

  const handleCreateBoxPrompt = (boxClicks: [modelInputProps, modelInputProps]) => {
    lastStrokeIdRef.current = null;
    keepPersistentUndoOnly();
    setError(null);
    setSelectedMaskId(null);
    setCurrentMask(null);
    setUndoOrder((previous) => [...previous, "prompt"]);
    setClicks((previous) => {
      const pointPrompts = previous.filter((click) => click.clickType !== 2 && click.clickType !== 3);
      return [...boxClicks, ...pointPrompts.filter((click) => click.clickType === 0 || click.clickType === 1)];
    });
  };

  const handleRefineMask = (stroke: MaskEditStroke) => {
    setCurrentMask((previous) => {
      if (!previous) {
        return previous;
      }
      if (lastStrokeIdRef.current !== stroke.strokeId) {
        lastStrokeIdRef.current = stroke.strokeId;
        setEditHistory((history) => [...history, cloneMask(previous)]);
        setUndoOrder((history) => [...history, "edit"]);
      }
      const nextBinary = applyBrushStroke(
        previous.binaryMask,
        previous.width,
        previous.height,
        stroke.fromX,
        stroke.fromY,
        stroke.toX,
        stroke.toY,
        stroke.radius,
        stroke.tool === "brush" ? 255 : 0
      );
      const nextLayer = createMaskLayer(nextBinary, previous.width, previous.height, previous.color);
      return {
        ...previous,
        ...nextLayer,
        binaryMask: nextBinary,
      };
    });
  };

  const handleUndo = () => {
    setSelectedMaskId(null);
    const lastAction = undoOrder[undoOrder.length - 1];

    if (lastAction === "delete" && deleteHistory.length > 0) {
      const snapshot = deleteHistory[deleteHistory.length - 1];
      setDeleteHistory((previous) => previous.slice(0, -1));
      setUndoOrder((previous) => previous.slice(0, -1));
      setSavedMasks((previous) => {
        const next = [...previous];
        next.splice(snapshot.index, 0, cloneSavedMask(snapshot.mask));
        return next;
      });
      setSelectedMaskId(snapshot.mask.id);
      return;
    }

    if (lastAction === "save" && saveHistory.length > 0) {
      const snapshot = saveHistory[saveHistory.length - 1];
      lastStrokeIdRef.current = null;
      setSaveHistory((previous) => previous.slice(0, -1));
      setUndoOrder((previous) => previous.slice(0, -1));
      setSavedMasks((previous) => previous.filter((mask) => mask.id !== snapshot.mask.id));
      setCurrentMask(cloneMask(snapshot.restoredCurrentMask));
      setClicks(snapshot.restoredClicks.map((click) => ({ ...click })));
      setInteractionMode(snapshot.restoredInteractionMode);
      setEditHistory([]);
      return;
    }

    if (lastAction === "edit" && editHistory.length > 0 && currentMask) {
      const previousMask = editHistory[editHistory.length - 1];
      lastStrokeIdRef.current = null;
      setEditHistory((previous) => previous.slice(0, -1));
      setUndoOrder((previous) => previous.slice(0, -1));
      setCurrentMask(cloneMask(previousMask));
      return;
    }

    if (lastAction === "prompt" || clicks.length > 0) {
      setUndoOrder((previous) => (previous.length > 0 ? previous.slice(0, -1) : previous));
      setClicks((previous) => {
        if (previous.length === 0) {
          return previous;
        }
        const last = previous[previous.length - 1];
        if (last.clickType === 3) {
          const hasTopLeftBefore = previous.slice(0, -1).some((click) => click.clickType === 2);
          if (hasTopLeftBefore) {
            return previous.filter((click) => click.clickType !== 2 && click.clickType !== 3);
          }
        }
        return previous.slice(0, -1);
      });
    }
  };

  const handleSaveMask = async () => {
    if (!currentMask) {
      return;
    }
    lastStrokeIdRef.current = null;
    keepPersistentUndoOnly();
    const color = MASK_PALETTE[savedMasks.length % MASK_PALETTE.length];
    const savedLayer = createMaskLayer(currentMask.binaryMask, currentMask.width, currentMask.height, color);
    const savedMask: SavedMask = {
      id: `${currentItem?.id || "item"}-mask-${savedMasks.length + 1}`,
      overlayUrl: savedLayer.overlayUrl,
      maskPngDataUrl: currentMask.maskPngDataUrl,
      color,
      clicks: currentMask.clicks,
      area: savedLayer.area,
      bbox: savedLayer.bbox,
      width: currentMask.width,
      height: currentMask.height,
      binaryMask: new Uint8ClampedArray(currentMask.binaryMask),
      source: "manual",
      score: currentMask.score ?? null,
    };
    setSaveHistory((previous) => [
      ...previous,
      {
        index: savedMasks.length,
        mask: cloneSavedMask(savedMask),
        restoredCurrentMask: cloneMask(currentMask),
        restoredClicks: currentMask.clicks.map((click) => ({ ...click })),
        restoredInteractionMode: interactionMode,
      },
    ]);
    setUndoOrder((previous) => [...previous.filter((action) => action === "delete" || action === "save"), "save"]);
    setSavedMasks((previous) => [...previous, savedMask]);
    lastStrokeIdRef.current = null;
    setClicks([]);
    setCurrentMask(null);
    setEditHistory([]);
    setSelectedMaskId(null);
    setInteractionMode("positive");
  };

  const advanceFromResponse = async (response: Response) => {
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    await applyNextPayload(payload.dataset || dataset, payload);
  };

  const handleCompleteImage = async () => {
    if (!currentItem) {
      return;
    }
    try {
      setBusyMessage("Saving annotations...");
      const annotations = [...savedMasks];
      if (currentMask) {
        const color = MASK_PALETTE[annotations.length % MASK_PALETTE.length];
        annotations.push({
          id: `${currentItem.id}-mask-${annotations.length + 1}`,
          overlayUrl: currentMask.overlayUrl,
          maskPngDataUrl: currentMask.maskPngDataUrl,
          color,
          clicks: currentMask.clicks,
          area: currentMask.area,
        });
      }
      const response = await fetch(`/api/items/${currentItem.id}/complete?dataset=${dataset}&runtime=${runtimeMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotations: annotations.map((annotation) => ({
            clicks: annotation.clicks,
            maskPngDataUrl: annotation.maskPngDataUrl,
            color: annotation.color,
            source: annotation.source || "manual",
            score: annotation.score ?? null,
          })),
        }),
      });
      await advanceFromResponse(response);
    } catch (saveError: any) {
      setError(saveError.message || "Failed to save annotations.");
      setBusyMessage(null);
    }
  };

  const handleSkipImage = async () => {
    if (!currentItem) {
      return;
    }
    try {
      setBusyMessage("Skipping image...");
      const response = await fetch(`/api/items/${currentItem.id}/skip?dataset=${dataset}&runtime=${runtimeMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note: "skipped from web demo" }),
      });
      await advanceFromResponse(response);
    } catch (skipError: any) {
      setError(skipError.message || "Failed to skip image.");
      setBusyMessage(null);
    }
  };

  return (
    <Stage
      dataset={dataset}
      runtimeInfo={runtimeInfo}
      runtimeMode={runtimeMode}
      currentItem={currentItem}
      image={image}
      savedMasks={savedMasks}
      selectedMaskId={selectedMaskId}
      currentMask={currentMask}
      clicks={clicks}
      interactionMode={interactionMode}
      brushSize={brushSize}
      canUndo={undoOrder.length > 0 || clicks.length > 0}
      progress={progress}
      busyMessage={busyMessage}
      error={error}
      finished={finished}
      onDatasetChange={handleDatasetChange}
      onRuntimeChange={handleRuntimeChange}
      onModeChange={setInteractionMode}
      onBrushSizeChange={setBrushSize}
      onImageClick={handleImageClick}
      onCreateBoxPrompt={handleCreateBoxPrompt}
      onSelectSavedMask={setSelectedMaskId}
      onPromoteSavedMask={(maskId) => {
        void promoteSavedMaskToCurrent(maskId);
      }}
      onRefineMask={handleRefineMask}
      onUndo={handleUndo}
      onResetObject={clearCurrentObject}
      onSaveMask={handleSaveMask}
      onRemoveLastMask={() =>
        setSavedMasks((previous) => {
          const next = previous.slice(0, -1);
          if (selectedMaskId && !next.some((mask) => mask.id === selectedMaskId)) {
            setSelectedMaskId(null);
          }
          return next;
        })
      }
      onClearSavedMasks={() => {
        setSelectedMaskId(null);
        setSavedMasks([]);
      }}
      onCompleteImage={handleCompleteImage}
      onSkipImage={handleSkipImage}
    />
  );
};

export default App;
