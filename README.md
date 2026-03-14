# SAM Assisted-manual Data Engine Demo

这是一个基于 SAM 论文里 data engine 第一阶段（assisted-manual）的交互标注 demo：

- 自动从 `data/diagram/images` 或 `data/plot/images` 逐张取图
- 后端用原始 SAM checkpoint 计算 image embedding 并缓存
- 前端用源码导出的 lightweight ONNX mask decoder 做交互分割
- 支持正负点提示、保存当前 mask、完成当前图后自动切换下一张
- 标注结果保存到 `outputs/annotations`

## 目录说明

- `backend/`：FastAPI 后端，负责图片队列、embedding 缓存、标注保存
- `segment-anything/demo/`：改造后的前端界面
- `outputs/embeddings/`：缓存的 `.npy` image embedding
- `outputs/annotations/`：保存的 mask PNG 和每张图的标注 JSON

## 环境准备

项目里原有 `.venv` 已损坏，这里使用新的虚拟环境 `.venv_demo`。此外，需要先把官方 `segment-anything` 仓库拉到当前项目的 `segment-anything/` 目录下，供后端导出 ONNX 和加载 SAM 模型：

```bash
git clone https://github.com/facebookresearch/segment-anything.git segment-anything
```

然后安装 Python 依赖：

```bash
python3 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
```

前端依赖：

```bash
cd segment-anything/demo
npm install --legacy-peer-deps
```

## 构建前端

```bash
cd segment-anything/demo
npm run build
```

## 启动系统

在项目根目录执行：

```bash
source .venv_demo/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9000
```

打开：

- [http://127.0.0.1:9000](http://127.0.0.1:9000)

首次启动时如果缺少 ONNX 文件，后端会自动基于 `checkpoint/` 下的 checkpoint 导出并量化：

- `segment-anything/demo/model/vit_h_assisted_manual_quantized.onnx`

首次打开某张图时，后端会自动生成对应 embedding，并缓存到：

- `outputs/embeddings/<dataset>/<item_id>.npy`

## 标注输出

每张图完成后会生成：

- `outputs/annotations/<dataset>/items/<item_id>.json`
- `outputs/annotations/<dataset>/masks/<item_id>/mask_XXX.png`

## 已验证内容

- 成功导出并量化 ONNX decoder
- 成功构建前端 webpack 产物
- 成功生成 `diagram` 数据首张图的 image embedding
- 成功通过 API 保存一条标注并自动切换到下一张图
