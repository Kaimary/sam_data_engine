import * as ort from "onnxruntime-web";
import npyjs from "npyjs";
import { DatasetItem, RuntimeInfo, modelInputProps } from "./Interfaces";

const MODEL_SIZE = 1024;
const MASK_INPUT_SIZE = 256;

interface LoadedEmbedding {
  itemId: string;
  tensor: ort.Tensor;
}

interface LoadedSession {
  modelUrl: string;
  provider: "webgpu" | "wasm";
  session: ort.InferenceSession;
}

const embeddingLoader = new npyjs();
let embeddingCache = new Map<string, Promise<LoadedEmbedding>>();
let sessionCache = new Map<string, Promise<LoadedSession>>();

const getScaledCoordinate = (value: number, imageSize: number, scale: number) =>
  Math.max(0, Math.min(imageSize * scale, value * scale));

const getSamScale = (width: number, height: number) => MODEL_SIZE / Math.max(width, height);

const getSessionKey = (runtime: RuntimeInfo) => `${runtime.modelUrl}:${runtime.mode}`;

const createSession = async (runtime: RuntimeInfo): Promise<LoadedSession> => {
  const sessionKey = getSessionKey(runtime);
  const cached = sessionCache.get(sessionKey);
  if (cached) {
    return cached;
  }

  const nextSession = (async () => {
    const providers: Array<"webgpu" | "wasm"> = [];
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      providers.push("webgpu");
    }
    providers.push("wasm");

    let lastError: unknown = null;
    for (const provider of providers) {
      try {
        const session = await ort.InferenceSession.create(runtime.modelUrl, {
          executionProviders: [provider],
          graphOptimizationLevel: "all",
        });
        return {
          modelUrl: runtime.modelUrl,
          provider,
          session,
        };
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError ?? new Error("Failed to create browser inference session.");
  })();

  sessionCache.set(sessionKey, nextSession);
  try {
    return await nextSession;
  } catch (error) {
    sessionCache.delete(sessionKey);
    throw error;
  }
};

export const loadBrowserEmbedding = async (item: DatasetItem): Promise<LoadedEmbedding> => {
  const cached = embeddingCache.get(item.id);
  if (cached) {
    return cached;
  }

  const nextEmbedding = (async () => {
    const loaded = (await embeddingLoader.load(item.embeddingUrl)) as {
      data: Float32Array;
      shape: number[];
    };
    return {
      itemId: item.id,
      tensor: new ort.Tensor("float32", loaded.data, loaded.shape),
    };
  })();

  embeddingCache.set(item.id, nextEmbedding);
  try {
    return await nextEmbedding;
  } catch (error) {
    embeddingCache.delete(item.id);
    throw error;
  }
};

const createFeed = (item: DatasetItem, clicks: modelInputProps[], embedding: ort.Tensor): Record<string, ort.Tensor> => {
  const scale = getSamScale(item.width, item.height);
  const promptCount = clicks.length + 1;
  const coords = new Float32Array(promptCount * 2);
  const labels = new Float32Array(promptCount);

  clicks.forEach((click, index) => {
    coords[index * 2] = getScaledCoordinate(click.x, item.width, scale);
    coords[index * 2 + 1] = getScaledCoordinate(click.y, item.height, scale);
    labels[index] = click.clickType;
  });

  labels[promptCount - 1] = -1;

  return {
    image_embeddings: embedding,
    point_coords: new ort.Tensor("float32", coords, [1, promptCount, 2]),
    point_labels: new ort.Tensor("float32", labels, [1, promptCount]),
    mask_input: new ort.Tensor("float32", new Float32Array(MASK_INPUT_SIZE * MASK_INPUT_SIZE), [
      1,
      1,
      MASK_INPUT_SIZE,
      MASK_INPUT_SIZE,
    ]),
    has_mask_input: new ort.Tensor("float32", new Float32Array([0]), [1]),
    orig_im_size: new ort.Tensor("float32", new Float32Array([item.height, item.width]), [2]),
  };
};

export const getBrowserRuntimeLabel = (provider: "webgpu" | "wasm") =>
  provider === "webgpu" ? "Browser WebGPU" : "Browser WASM";

export const runBrowserPrediction = async (
  runtime: RuntimeInfo,
  item: DatasetItem,
  clicks: modelInputProps[],
) => {
  const [{ session, provider }, { tensor }] = await Promise.all([
    createSession(runtime),
    loadBrowserEmbedding(item),
  ]);
  const feeds = createFeed(item, clicks, tensor);
  const results = await session.run(feeds);
  const maskTensor = results[session.outputNames[0]];
  const scoreTensor = results[session.outputNames[1]];

  if (!maskTensor) {
    throw new Error("Browser inference did not return a mask tensor.");
  }

  return {
    provider,
    mask: maskTensor.data as Float32Array,
    score: scoreTensor ? Number((scoreTensor.data as Float32Array)[0]) : null,
  };
};
