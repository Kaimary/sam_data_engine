export type InteractionMode = "positive" | "negative" | "box" | "brush" | "eraser";
export type RuntimeMode = "server" | "browser";

export interface modelInputProps {
  x: number;
  y: number;
  clickType: number;
}

export interface MaskEditStroke {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  tool: "brush" | "eraser";
  radius: number;
  strokeId: number;
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

export interface RuntimeInfo {
  mode: RuntimeMode;
  defaultMode: RuntimeMode;
  location: "server" | "browser";
  device: string;
  label: string;
  interactiveLabel: string;
  automaticMaskLabel: string;
  modelUrl: string;
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
  bbox?: [number, number, number, number];
  width?: number;
  height?: number;
  binaryMask?: Uint8ClampedArray;
  source?: "automatic" | "manual";
  score?: number | null;
}

export interface CurrentMask extends SavedMask {
  width: number;
  height: number;
  binaryMask: Uint8ClampedArray;
}

export interface ToolProps {
  image: HTMLImageElement | null;
  savedMasks: SavedMask[];
  selectedMaskId: string | null;
  currentMask: CurrentMask | null;
  clicks: modelInputProps[];
  interactionMode: InteractionMode;
  brushSize: number;
  disabled?: boolean;
  onImageClick: (click: modelInputProps, button: number) => void;
  onCreateBoxPrompt: (boxClicks: [modelInputProps, modelInputProps]) => void;
  onSelectSavedMask: (maskId: string) => void;
  onPromoteSavedMask: (maskId: string) => void;
  onRefineMask: (stroke: MaskEditStroke) => void;
}
