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

const maskStats = (binary: Uint8ClampedArray, width: number, height: number) => {
  let xMin = width;
  let yMin = height;
  let xMax = -1;
  let yMax = -1;
  let area = 0;

  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      if (binary[rowOffset + x] === 0) {
        continue;
      }
      area += 1;
      if (x < xMin) xMin = x;
      if (y < yMin) yMin = y;
      if (x > xMax) xMax = x;
      if (y > yMax) yMax = y;
    }
  }

  return {
    area,
    bbox: area === 0 ? ([0, 0, 0, 0] as [number, number, number, number]) : ([xMin, yMin, xMax - xMin + 1, yMax - yMin + 1] as [number, number, number, number]),
  };
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
  fillAlpha = 110,
  edgeAlpha = 255
) => {
  const [r, g, b] = hexToRgba(color, fillAlpha);
  const arr = new Uint8ClampedArray(4 * width * height).fill(0);
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] <= 0.0) {
      continue;
    }

    const x = i % width;
    const y = Math.floor(i / width);
    const left = x === 0 || input[i - 1] <= 0.0;
    const right = x === width - 1 || input[i + 1] <= 0.0;
    const top = y === 0 || input[i - width] <= 0.0;
    const bottom = y === height - 1 || input[i + width] <= 0.0;
    const isEdge = left || right || top || bottom;

    if (isEdge) {
      arr[4 * i + 0] = 255;
      arr[4 * i + 1] = 255;
      arr[4 * i + 2] = 255;
      arr[4 * i + 3] = edgeAlpha;
      continue;
    }

    arr[4 * i + 0] = r;
    arr[4 * i + 1] = g;
    arr[4 * i + 2] = b;
    arr[4 * i + 3] = fillAlpha;
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
  const { binary } = thresholdMask(input);
  const stats = maskStats(binary, width, height);
  const overlayCanvas = imageDataToCanvas(buildOverlayImageData(binary, width, height, color));
  const binaryCanvas = imageDataToCanvas(buildBinaryMaskImageData(input, width, height));
  return {
    overlayUrl: overlayCanvas.toDataURL("image/png"),
    maskPngDataUrl: binaryCanvas.toDataURL("image/png"),
    area: stats.area,
    bbox: stats.bbox,
  };
};

export const maskDataUrlToBinaryMask = async (dataUrl: string) => {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  context?.drawImage(bitmap, 0, 0);
  const imageData = context?.getImageData(0, 0, canvas.width, canvas.height).data || new Uint8ClampedArray();
  const binary = new Uint8ClampedArray(canvas.width * canvas.height);
  for (let i = 0; i < binary.length; i += 1) {
    binary[i] = imageData[i * 4 + 3] > 0 ? 255 : 0;
  }
  return {
    binary,
    width: canvas.width,
    height: canvas.height,
  };
};

export const applyBrushStroke = (
  input: Uint8ClampedArray,
  width: number,
  height: number,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  radius: number,
  value: 0 | 255
) => {
  const output = new Uint8ClampedArray(input);
  const dx = toX - fromX;
  const dy = toY - fromY;
  const distance = Math.hypot(dx, dy);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.5)));
  const radiusSq = radius * radius;

  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const cx = Math.round(fromX + dx * t);
    const cy = Math.round(fromY + dy * t);
    const xMin = Math.max(0, Math.floor(cx - radius));
    const xMax = Math.min(width - 1, Math.ceil(cx + radius));
    const yMin = Math.max(0, Math.floor(cy - radius));
    const yMax = Math.min(height - 1, Math.ceil(cy + radius));

    for (let y = yMin; y <= yMax; y += 1) {
      const rowOffset = y * width;
      const yDiff = y - cy;
      for (let x = xMin; x <= xMax; x += 1) {
        const xDiff = x - cx;
        if (xDiff * xDiff + yDiff * yDiff <= radiusSq) {
          output[rowOffset + x] = value;
        }
      }
    }
  }

  return output;
};
