import { Tensor } from "onnxruntime-web";
import { modeDataProps } from "./Interfaces";

const modelData = ({ clicks, tensor, modelScale, lowResMask }: modeDataProps) => {
  const imageEmbedding = tensor;
  let pointCoords;
  let pointLabels;
  let pointCoordsTensor;
  let pointLabelsTensor;

  if (clicks) {
    const n = clicks.length;
    pointCoords = new Float32Array(2 * (n + 1));
    pointLabels = new Float32Array(n + 1);

    for (let i = 0; i < n; i += 1) {
      pointCoords[2 * i] = clicks[i].x * modelScale.samScale;
      pointCoords[2 * i + 1] = clicks[i].y * modelScale.samScale;
      pointLabels[i] = clicks[i].clickType;
    }

    pointCoords[2 * n] = 0.0;
    pointCoords[2 * n + 1] = 0.0;
    pointLabels[n] = -1.0;

    pointCoordsTensor = new Tensor("float32", pointCoords, [1, n + 1, 2]);
    pointLabelsTensor = new Tensor("float32", pointLabels, [1, n + 1]);
  }

  if (pointCoordsTensor === undefined || pointLabelsTensor === undefined) {
    return;
  }

  const imageSizeTensor = new Tensor("float32", [modelScale.height, modelScale.width]);
  const hasMaskInput = lowResMask ? new Tensor("float32", [1]) : new Tensor("float32", [0]);
  const maskInput =
    lowResMask ??
    new Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);

  return {
    image_embeddings: imageEmbedding,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    orig_im_size: imageSizeTensor,
    mask_input: maskInput,
    has_mask_input: hasMaskInput,
  };
};

export { modelData };
