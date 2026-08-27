# 本地模型接入

默认 Docker demo 不需要模型。只有调试真实 adapter 时才执行本页步骤；模型许可、来源和存储由操作者负责。

## 1. PaddleOCR

在 Git 外准备权重，再映射到以下目录契约：

```text
models/paddleocr/PP-OCRv6_medium_det/
models/paddleocr/PP-OCRv6_medium_rec/
models/paddleocr/PP-LCNet_x1_0_textline_ori/
```

不要把压缩包或解压后的权重加入仓库。`.gitignore` 和公开安全门禁会拒绝常见权重后缀；默认 Dockerfile 也不会复制 `models/`。

安装真实 adapter 的可选依赖：

```bash
python -m pip install -e "backend[ai]"
```

`PaddleOcrEngine` 接收三个 `Path`，初始化前确认目录存在。它只加载显式传入的本地目录，不依赖仓库自动下载模型。

若以后通过 Compose 接入，可在未跟踪的 override 文件中只读挂载：

```yaml
services:
  api:
    volumes:
      - ${PADDLE_MODELS_HOST:?set-in-untracked-env}:/models/paddleocr:ro
```

这只是目录挂载示例；当前 demo API 不会自动创建真实 OCR pipeline。

## 2. Ollama / Qwen-VL

`OllamaVisionExtractor` 需要：

- 操作者提供的 Ollama base URL；
- 操作者已安装并验证的视觉模型 tag；
- 候选字段白名单；
- 原图、处理图与 PaddleOCR 结果。

环境变量模板：

```text
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL_TAG=your-installed-model-tag
```

仓库不指定一个“保证可用”的模型 tag，也不声称任何模型在当前机器上通过了准确率或性能评测。实际使用前应先用 Ollama 自身接口验证模型支持图像输入，再将 URL/tag 注入自己的 pipeline wiring。

## 3. 验证顺序

1. 先跑 adapter 单元测试；测试使用 fake/mock，不下载权重。
2. 用不含个人信息的本地合成图片验证三个 Paddle 目录能加载。
3. 验证 Ollama 返回符合 Schema 的 JSON，且字段只来自候选白名单。
4. 检查金额字段是否关联 OCR evidence；无证据金额不应被视为高置信自动结果。
5. 真实数据只能在合规的私有环境评测，不能加入 Git、README 截图或公开日志。
