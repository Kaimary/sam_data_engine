import { Tensor } from "onnxruntime-web";

export interface modelScaleProps {
  samScale: number;
  height: number;
  width: number;
}

export interface modelInputProps {
  x: number;
  y: number;
  clickType: number;
}

export interface modeDataProps {
  clicks?: Array<modelInputProps>;
  tensor: Tensor;
  modelScale: modelScaleProps;
  lowResMask?: Tensor | null;
}

export interface DatasetItem {
  id: string;
  dataset: string;
  index: number;
  total: number;
  filename: string;
  width: number;
  height: number;
  title: string;
  split: string | null;
  visualIntent: string | null;
  methodSectionTitle: string | null;
  metadataId: string | null;
  imageUrl: string;
  embeddingUrl: string;
}

export interface ProgressSummary {
  total: number;
  completed: number;
  skipped: number;
  pending: number;
}

export interface SavedMask {
  id: string;
  overlayUrl: string;
  maskPngDataUrl: string;
  color: string;
  clicks: modelInputProps[];
  area: number;
}

export interface CurrentMask extends SavedMask {}

export interface ToolProps {
  image: HTMLImageElement | null;
  savedMasks: SavedMask[];
  currentMask: CurrentMask | null;
  clicks: modelInputProps[];
  clickMode: "positive" | "negative";
  disabled?: boolean;
  onAddClick: (click: modelInputProps) => void;
}
