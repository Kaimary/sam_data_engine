const thresholdMask = (input: ArrayLike<number>) => {
  const binary = new Uint8ClampedArray(input.length);
  let area = 0;
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] > 0.0) {
      binary[i] = 255;
      area += 1;
    }
  }
  return { binary, area };
};

const hexToRgba = (hex: string, alpha: number) => {
  const normalized = hex.replace("#", "");
  const r = parseInt(normalized.slice(0, 2), 16);
  const g = parseInt(normalized.slice(2, 4), 16);
  const b = parseInt(normalized.slice(4, 6), 16);
  return [r, g, b, alpha] as const;
};

const imageDataToCanvas = (imageData: ImageData) => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  context?.putImageData(imageData, 0, 0);
  return canvas;
};

const buildOverlayImageData = (
  input: ArrayLike<number>,
  width: number,
  height: number,
  color: string,
  alpha = 180
) => {
  const [r, g, b, a] = hexToRgba(color, alpha);
  const arr = new Uint8ClampedArray(4 * width * height).fill(0);
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] > 0.0) {
      arr[4 * i + 0] = r;
      arr[4 * i + 1] = g;
      arr[4 * i + 2] = b;
      arr[4 * i + 3] = a;
    }
  }
  return new ImageData(arr, width, height);
};

const buildBinaryMaskImageData = (input: ArrayLike<number>, width: number, height: number) => {
  const { binary } = thresholdMask(input);
  const arr = new Uint8ClampedArray(4 * width * height).fill(0);
  for (let i = 0; i < binary.length; i += 1) {
    if (binary[i] > 0) {
      arr[4 * i + 0] = 255;
      arr[4 * i + 1] = 255;
      arr[4 * i + 2] = 255;
      arr[4 * i + 3] = 255;
    }
  }
  return new ImageData(arr, width, height);
};

export const createMaskLayer = (
  input: ArrayLike<number>,
  width: number,
  height: number,
  color: string
) => {
  const overlayCanvas = imageDataToCanvas(buildOverlayImageData(input, width, height, color));
  const binaryCanvas = imageDataToCanvas(buildBinaryMaskImageData(input, width, height));
  const { area } = thresholdMask(input);
  return {
    overlayUrl: overlayCanvas.toDataURL("image/png"),
    maskPngDataUrl: binaryCanvas.toDataURL("image/png"),
    area,
  };
};
