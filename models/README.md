# 本地模型目录

模型权重不进入 Git，也不打进默认 demo 镜像。将合法获取的模型文件放在仓库外部；需要真实 OCR 时，再以只读卷挂载到运行环境。

PaddleOCR adapter 期望三个目录：

```text
models/paddleocr/PP-OCRv6_medium_det/
models/paddleocr/PP-OCRv6_medium_rec/
models/paddleocr/PP-LCNet_x1_0_textline_ori/
```

这些路径只描述本地目录契约，不代表仓库提供或自动下载权重。完整接入说明见 `docs/model-setup.md`。
