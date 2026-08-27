# 医疗票据智能识别与理赔预审

一个可运行的医疗票据识别与预审 MVP：用 **OpenCV → PaddleOCR → Qwen-VL → 字段归一化 → Decimal 序贯规则 → 人工复核** 拆解票据审核流程。

公开版本默认运行合成案例，不需要 GPU、模型权重、Ollama、PaddleOCR 下载或数据库。真实预处理、OCR 与视觉语言模型 adapter 已在后端源码中实现并由单元测试覆盖，但当前 demo API 不提供真实票据上传入口，也不代表保险公司的正式理赔结论。

## 1. 一分钟启动

需要 Docker Desktop 或兼容的 Docker Engine：

```bash
docker compose up --build
```

- 审核工作台：<http://localhost:8080>
- Demo API：<http://localhost:8000/api/v1/review/cases>
- 健康检查：<http://localhost:8000/health/live>

端口可在未跟踪的 `.env` 中覆盖，模板见 `.env.example`。审核修改保存在 API 进程内存中，容器重启后恢复合成 fixture。

## 2. 项目怎么工作

```text
原始票据
  → OpenCV 方向修正、倾斜校正与质量告警
  → PaddleOCR 文本、坐标、置信度
  → Qwen-VL 白名单字段抽取与 OCR 证据关联
  → 标准字段别名归一化 / Pydantic 校验
  → R001…R006 按顺序首条匹配 / Decimal 金额计算
  → 高风险或低置信转三栏工作台人工复核
```

关键设计是让模型负责“看懂与定位”，让确定性 Python 规则负责关键金额计算。模型输出不能跳过字段白名单、Schema、证据和置信度边界。

## 3. 六条业务规则

规则严格按 `R001 → R006` 执行。第一条 `MATCHED` 直接返回；高优先级候选若缺字段、低置信或计算异常则 `NEEDS_REVIEW`，不会静默降级到后续规则。

| 优先级 | 匹配与输出摘要 |
|---|---|
| R001 | 政策范围金额、统筹支付、个人自付齐全，且 `政策范围金额 - 统筹支付 ≤ 个人自付`，返回政策范围金额 |
| R002 | 无政策范围/统筹字段但有个人现金支付，返回个人现金支付 |
| R003 | 医保结算单字段齐全，返回 `统筹基金支付范围内费用 - 基金支付合计` |
| R004 | 同时有自付一、自付二，返回自付一 |
| R005 | `个人自付 + 统筹支付 + 个人自费 ≈ 金额合计`，返回 `个人自付 - 乙类先行自付 - 超限价自付` |
| R006 | 购买方、销售方、价税合计齐全的药店票据，返回价税合计 |

金额统一使用 `Decimal`，缺失值 `null` 与合法金额 `0.00` 分开处理，默认平衡容差为 `0.01`。

## 4. 已实现与边界

| 已实现 MVP | 尚未实现（生产演进） |
|---|---|
| OpenCV 预处理、质量告警与原/处理图产物 | 真实票据上传、病毒扫描、对象存储 |
| 本地 PaddleOCR adapter、坐标与阅读顺序映射 | GPU worker 与模型服务编排 |
| Ollama/Qwen-VL 结构化抽取、白名单、证据绑定 | 模型评测数据集与线上精度/延迟指标 |
| 六条序贯 Decimal 规则及完整 trace | 可配置规则管理后台 |
| 合成 fixture 的 FastAPI 审核 API、版本冲突保护 | PostgreSQL、Redis、Celery、MinIO |
| React 三栏复核台、字段修正、通过/驳回、JSON 导出 | JWT/RBAC、多租户、持久审计 |
| Docker demo、CI、公开仓库安全扫描 | 正式保险理赔决策或生产 SLA |

## 5. 本地开发与质量门禁

后端（Python 3.11）：

```bash
python -m pip install -e "backend[dev]" "numpy>=1.26,<3" "opencv-python-headless>=4.10,<5" "pillow>=10,<13"
python -m pytest backend
python -m mypy backend/src backend/tests
python -m ruff check backend/src backend/tests
uvicorn claim_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

前端（Node 22、pnpm 11）：

```bash
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend test
pnpm --dir frontend typecheck
pnpm --dir frontend build
pnpm --dir frontend dev
```

公开前安全门禁：

```bash
python -m pytest scripts/tests/test_check_public_repo.py
python scripts/check_public_repo.py
```

门禁默认扫描 Git 已跟踪文件，拒绝权重、非合成栅格图片、密钥、环境文件、聊天临时目录、本机绝对路径和内部构建产物；失败信息只显示路径与原因。

## 6. 模型、数据与隐私

- Git 只保存源码、配置和明确标注的合成 JSON；不保存真实医疗票据、二维码、个人信息、密钥或模型权重。
- 默认镜像没有内置 `models.zip`，也不会在启动时下载模型。
- PaddleOCR 使用操作者在 Git 外准备的本地目录；Ollama URL 与模型 tag 由操作者提供。
- 当前仓库没有足量真实标注数据，因此不声明准确率、吞吐、延迟、成本或生产用户数。

真实模型接入见 [docs/model-setup.md](docs/model-setup.md)，架构和信任边界见 [docs/architecture.md](docs/architecture.md)，面试讲法见 [docs/interview-guide.md](docs/interview-guide.md)。

## 7. 目录结构

```text
backend/              FastAPI、领域模型、AI adapters、流水线与规则
frontend/             React + TypeScript 三栏审核工作台
configs/              字段别名、模板与候选字段配置
samples/synthetic/    合成 demo fixture
scripts/              公开仓库安全门禁
docs/                 架构、模型接入、面试说明
models/README.md      本地权重目录契约（不含权重）
docker-compose.yml    无模型 demo 编排
```

## License

[MIT](LICENSE)
