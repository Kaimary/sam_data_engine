import React, { useEffect, useMemo, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import * as ort from "onnxruntime-web";
import npyjs from "npyjs";
import "./assets/scss/App.scss";
import Stage from "./components/Stage";
import { handleImageScale } from "./components/helpers/scaleHelper";
import {
  CurrentMask,
  DatasetItem,
  ProgressSummary,
  SavedMask,
  modelInputProps,
  modelScaleProps,
} from "./components/helpers/Interfaces";
import { createMaskLayer } from "./components/helpers/maskUtils";
import { modelData } from "./components/helpers/onnxModelAPI";

const DEFAULT_DATASET = "diagram";
const MASK_PALETTE = ["#ff7a59", "#5abf90", "#4fa3ff", "#f6bd60", "#9d7bff", "#f28482"];
const PREVIEW_COLOR = "#2e86ab";

const emptyProgress: ProgressSummary = {
  total: 0,
  completed: 0,
  skipped: 0,
  pending: 0,
};

const App = () => {
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const [model, setModel] = useState<InferenceSession | null>(null);
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [progress, setProgress] = useState<ProgressSummary>(emptyProgress);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);
  const [tensor, setTensor] = useState<Tensor | null>(null);
  const [clickMode, setClickMode] = useState<"positive" | "negative">("positive");
  const [clicks, setClicks] = useState<modelInputProps[]>([]);
  const [currentMask, setCurrentMask] = useState<CurrentMask | null>(null);
  const [savedMasks, setSavedMasks] = useState<SavedMask[]>([]);
  const [busyMessage, setBusyMessage] = useState<string | null>("Loading model...");
  const [error, setError] = useState<string | null>(null);

  const loadCounterRef = useRef(0);
  const predictionCounterRef = useRef(0);
  const lowResMaskRef = useRef<Tensor | null>(null);
  const npLoaderRef = useRef<any>(null);

  const finished = useMemo(() => !currentItem && progress.pending === 0 && progress.total > 0, [currentItem, progress]);

  useEffect(() => {
    ort.env.wasm.numThreads = Math.min(window.navigator.hardwareConcurrency || 4, 6);
  }, []);

  const clearCurrentObject = () => {
    lowResMaskRef.current = null;
    setClicks([]);
    setCurrentMask(null);
  };

  const clearImageState = () => {
    lowResMaskRef.current = null;
    setImage(null);
    setTensor(null);
    setModelScale(null);
    setClicks([]);
    setCurrentMask(null);
    setSavedMasks([]);
  };

  const ensureModel = async (nextModelUrl: string) => {
    if (model && modelUrl === nextModelUrl) {
      return model;
    }
    setBusyMessage("Loading ONNX decoder...");
    const nextModel = await InferenceSession.create(nextModelUrl);
    setModel(nextModel);
    setModelUrl(nextModelUrl);
    return nextModel;
  };

  const loadImageElement = (source: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${source}`));
      img.src = source;
    });

  const loadEmbeddingTensor = async (source: string) => {
    if (!npLoaderRef.current) {
      npLoaderRef.current = new npyjs();
    }
    const npArray = await npLoaderRef.current.load(source);
    return new ort.Tensor("float32", npArray.data, npArray.shape);
  };

  const applyNextPayload = async (nextDataset: string, payload: any) => {
    setDataset(nextDataset);
    setProgress(payload.progress || emptyProgress);
    clearImageState();
    setError(null);

    if (payload.modelUrl) {
      await ensureModel(payload.modelUrl);
    }

    if (!payload.item) {
      setCurrentItem(null);
      setBusyMessage(null);
      return;
    }

    const loadId = loadCounterRef.current + 1;
    loadCounterRef.current = loadId;
    setBusyMessage("Preparing image embedding...");
    const nextItem = payload.item as DatasetItem;
    setCurrentItem(nextItem);

    try {
      const [nextImage, nextTensor] = await Promise.all([
        loadImageElement(nextItem.imageUrl),
        loadEmbeddingTensor(nextItem.embeddingUrl),
      ]);
      if (loadCounterRef.current !== loadId) {
        return;
      }
      const nextScale = handleImageScale(nextImage);
      setImage(nextImage);
      setModelScale(nextScale);
      setTensor(nextTensor);
      setBusyMessage(null);
    } catch (loadError: any) {
      if (loadCounterRef.current !== loadId) {
        return;
      }
      setError(loadError.message || "Failed to load item assets.");
      setBusyMessage(null);
    }
  };

  const fetchBootstrap = async (nextDataset: string) => {
    const response = await fetch(`/api/bootstrap?dataset=${nextDataset}`);
    if (!response.ok) {
      throw new Error(`Failed to bootstrap dataset ${nextDataset}`);
    }
    return response.json();
  };

  useEffect(() => {
    const init = async () => {
      try {
        const payload = await fetchBootstrap(DEFAULT_DATASET);
        await applyNextPayload(DEFAULT_DATASET, payload);
      } catch (initError: any) {
        setError(initError.message || "Failed to initialize the app.");
        setBusyMessage(null);
      }
    };
    init();
  }, []);

  useEffect(() => {
    const runPrediction = async () => {
      if (!model || !tensor || !modelScale || clicks.length === 0) {
        return;
      }
      const requestId = predictionCounterRef.current + 1;
      predictionCounterRef.current = requestId;
      try {
        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
          lowResMask: clicks.length > 1 ? lowResMaskRef.current : null,
        });
        if (!feeds) {
          return;
        }
        const results = await model.run(feeds);
        if (predictionCounterRef.current !== requestId) {
          return;
        }
        const maskTensor = results[model.outputNames[0]];
        const lowResMasks = results[model.outputNames[2]] as Tensor;
        const layer = createMaskLayer(maskTensor.data as ArrayLike<number>, maskTensor.dims[3], maskTensor.dims[2], PREVIEW_COLOR);
        lowResMaskRef.current = lowResMasks;
        setCurrentMask({
          id: "preview",
          overlayUrl: layer.overlayUrl,
          maskPngDataUrl: layer.maskPngDataUrl,
          color: PREVIEW_COLOR,
          clicks: clicks.map((click) => ({ ...click })),
          area: layer.area,
        });
      } catch (predictionError: any) {
        setError(predictionError.message || "Failed to run the ONNX model.");
      }
    };

    if (clicks.length === 0) {
      setCurrentMask(null);
      return;
    }

    runPrediction();
  }, [clicks, model, tensor, modelScale]);

  const handleDatasetChange = async (nextDataset: string) => {
    if (nextDataset === dataset) {
      return;
    }
    try {
      setBusyMessage("Switching dataset...");
      const payload = await fetchBootstrap(nextDataset);
      await applyNextPayload(nextDataset, payload);
    } catch (datasetError: any) {
      setError(datasetError.message || "Failed to switch dataset.");
      setBusyMessage(null);
    }
  };

  const handleAddClick = (click: modelInputProps) => {
    setError(null);
    setClicks((previous) => [...previous, click]);
  };

  const handleUndo = () => {
    lowResMaskRef.current = null;
    setClicks((previous) => previous.slice(0, -1));
  };

  const handleSaveMask = async () => {
    if (!currentMask) {
      return;
    }
    const color = MASK_PALETTE[savedMasks.length % MASK_PALETTE.length];
    const response = await fetch(currentMask.maskPngDataUrl);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    context?.drawImage(bitmap, 0, 0);
    const imageData = context?.getImageData(0, 0, canvas.width, canvas.height).data || new Uint8ClampedArray();
    const binary = new Uint8ClampedArray(canvas.width * canvas.height);
    for (let i = 0; i < binary.length; i += 1) {
      binary[i] = imageData[i * 4 + 3];
    }
    const savedLayer = createMaskLayer(binary, canvas.width, canvas.height, color);
    setSavedMasks((previous) => [
      ...previous,
      {
        id: `${currentItem?.id || "item"}-mask-${previous.length + 1}`,
        overlayUrl: savedLayer.overlayUrl,
        maskPngDataUrl: currentMask.maskPngDataUrl,
        color,
        clicks: currentMask.clicks,
        area: currentMask.area,
      },
    ]);
    clearCurrentObject();
  };

  const advanceFromResponse = async (response: Response) => {
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    await applyNextPayload(dataset, payload);
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
      const response = await fetch(`/api/items/${currentItem.id}/complete?dataset=${dataset}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotations: annotations.map((annotation) => ({
            clicks: annotation.clicks,
            maskPngDataUrl: annotation.maskPngDataUrl,
            color: annotation.color,
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
      const response = await fetch(`/api/items/${currentItem.id}/skip?dataset=${dataset}`, {
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
      currentItem={currentItem}
      image={image}
      savedMasks={savedMasks}
      currentMask={currentMask}
      clicks={clicks}
      clickMode={clickMode}
      progress={progress}
      busyMessage={busyMessage}
      error={error}
      finished={finished}
      onDatasetChange={handleDatasetChange}
      onModeChange={setClickMode}
      onAddClick={handleAddClick}
      onUndo={handleUndo}
      onResetObject={clearCurrentObject}
      onSaveMask={handleSaveMask}
      onRemoveLastMask={() => setSavedMasks((previous) => previous.slice(0, -1))}
      onCompleteImage={handleCompleteImage}
      onSkipImage={handleSkipImage}
    />
  );
};

export default App;
