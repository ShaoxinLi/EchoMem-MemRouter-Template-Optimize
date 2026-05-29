# LoCoMo + VikingBot + MemRouter + OpenViking 端到端测评

> **版本**: v1.0  
> **日期**: 2026-05-25  
> **适用场景**: LoCoMo E2E 评估、MemRouter 路由准确率与答案正确率联合评测  
> **全量规模**: 1540 题（Category 1-4），10 个 conversation（conv-26, conv-30, conv-41 ~ conv-50）

## 目录结构

```
benchmarks/locomo/
├── README.md                              # 本文件
├── data/
│   ├── locomo10.json                      # LoCoMo 数据集（1986 questions, 10 conversations）
│   ├── locomo_e2e_route_labels.jsonl      # v1 路由标签（1540 条）
│   └── locomo_e2e_route_labels.v2.jsonl   # v2 路由标签（推荐，已 adjudication 修正）
├── configs/
│   ├── ov.conf.template                   # OpenViking 配置模板
│   ├── ovcli.conf                         # OpenViking CLI 客户端配置
│   └── memrouter_eval.local.yaml.template # MemRouter 配置模板
├── scripts/
│   ├── run_e2e.ps1                        # 一键测评 PowerShell 脚本（推荐）
│   ├── eval_locomo_vikingbot_memrouter_e2e.py  # Agent 级测评主脚本
│   ├── eval_vikingbot_memrouter_e2e.py    # Tool 级测评脚本
│   └── postprocess_judge.py               # Answer Judge 后处理脚本
└── patches/                               # OpenViking 端集成补丁
    ├── 01-openviking-fast-path.patch
    ├── 02-localclient-role-root.patch
    └── 03-vikingbot-local-client-switch.patch
```

## 前置依赖

### 代码仓库

本评测**不包含** OpenViking 的完整源码，运行前需确保以下仓库已克隆到本地：

| 仓库 | 期望路径 | 说明 |
|------|---------|------|
| OpenViking | `D:\Code\cursorProject\OpenViking` | 向量存储、VikingBot、HTTP Server |
| EchoMem | `D:\Code\cursorProject\EchoMem` | MemRouter 路由引擎（本仓库） |

> 如路径不同，请修改 `scripts/run_e2e.ps1` 中的 `$ovRoot` 变量。

### Python 依赖

在 OpenViking 和 EchoMem 环境中分别安装依赖：

```powershell
# OpenViking 环境
cd D:\Code\cursorProject\OpenViking
pip install -e .
pip install -r bot/requirements.txt

# EchoMem 环境
cd D:\Code\cursorProject\EchoMem
pip install -e .
```

### API Key 准备

复制模板文件并填入真实 API Key：

```powershell
cd D:\Code\cursorProject\EchoMem

# MemRouter 配置
copy benchmarks\locomo\configs\memrouter_eval.local.yaml.template benchmarks\locomo\configs\memrouter_eval.local.yaml

# OpenViking 配置
copy benchmarks\locomo\configs\ov.conf.template benchmarks\locomo\configs\ov.conf
```

然后编辑两个配置文件：

**`benchmarks/locomo/configs/ov.conf`:**
- `embedding.api_key` — DashScope API Key，用于 text-embedding-v3
- `vlm.api_key` — DeepSeek API Key，用于 OpenViking IntentAnalyzer
- `bot.agents.api_key` — DeepSeek API Key，用于 VikingBot answer generation

**`benchmarks/locomo/configs/memrouter_eval.local.yaml`:**
- `embedding.api_key` — DashScope API Key（同上）
- `llm.auth_token` — MiniMax API Key，用于 MemRouter LLM fallback

### OpenViking 补丁

> **OpenViking 版本要求**：`0.3.12`

端到端评测需要 3 个关键补丁，请在 OpenViking 仓库中依次应用：

```powershell
cd D:\Code\cursorProject\OpenViking

# 1. Fast Path 端点
git apply D:\Code\cursorProject\EchoMem\benchmarks\locomo\patches\01-openviking-fast-path.patch

# 2. LocalClient 权限修复
git apply D:\Code\cursorProject\EchoMem\benchmarks\locomo\patches\02-localclient-role-root.patch

# 3. VikingBot LocalClient 开关
git apply D:\Code\cursorProject\EchoMem\benchmarks\locomo\patches\03-vikingbot-local-client-switch.patch
```

### 索引 Workspace 数据

本评测使用的 workspace 已提前注入 LoCoMo conv-26 ~ conv-50 的用户记忆数据（markdown 格式）。OpenViking 的向量库不会自动索引 workspace 中的 markdown 文件，必须显式执行：

```powershell
cd D:\Code\cursorProject\OpenViking
python scripts/index_workspace_memories.py
```

这会为 conv-26 ~ conv-50 的所有用户数据生成向量索引。**必须等待 queue 排空后再开始评测**（通常需要 5-10 分钟）。

## 运行评测

### 全量评测（推荐夜间挂机）

```powershell
cd D:\Code\cursorProject\EchoMem
.\benchmarks\locomo\scripts\run_e2e.ps1 -ForceMemorySearch -Judge -JudgeToken "sk-your-deepseek-token"
```

参数说明：
- `-ForceMemorySearch` — 强制 VikingBot 在回答前调用 memory search
- `-Judge` — 启用 LLM Judge 评估答案正确性
- `-JudgeToken` — DeepSeek API Token（用于 judge）

预估耗时：**约 15 小时**（1540 题，每题 30-45 秒）。

### Pilot 小批量验证（50 题）

```powershell
.\benchmarks\locomo\scripts\run_e2e.ps1 -ForceMemorySearch -LimitQuestions 50 -Judge -JudgeToken "sk-xxxxx"
```

### 按 Category 分批执行

| 批次 | Category | 题目数 | 预估耗时 | 建议时段 |
|------|----------|--------|----------|----------|
| 1 | **1** | 282 | ~2h45m | 白天 |
| 2 | **3** | 96 | ~1h | 白天（快速验证） |
| 3 | **2** | 321 | ~3h | 傍晚 |
| 4 | **4** | 841 | ~8h | 夜间挂机 |

```powershell
# Category 1
.\benchmarks\locomo\scripts\run_e2e.ps1 -ForceMemorySearch -Category 1 -Judge -JudgeToken "sk-xxxxx"
# ... 以此类推
```

### 复用已启动的服务

```powershell
.\benchmarks\locomo\scripts\run_e2e.ps1 -ForceMemorySearch -SkipServerStart -Judge -JudgeToken "sk-xxxxx"
```

## 输出产物

评测完成后，在 `benchmarks/locomo/runs/{timestamp}_locomo_e2e/` 目录下产生：

```
run_dir/
├── logs/
│   ├── route_events.jsonl              # MemRouter 路由决策记录
│   ├── vikingbot.gateway.stderr.log   # VikingBot 网关日志
│   └── openviking.stderr.log          # OpenViking 服务日志
└── results/
    ├── qa_results.csv                  # 逐题明细
    ├── report.md                       # 人类可读汇总报告
    └── metrics_summary.json            # JSON 格式结构化指标
```

## 后处理：补跑 Answer Judge

如果首次运行未加 `-Judge` 参数，或 judge 过程中断，可以事后补跑：

```powershell
cd D:\Code\cursorProject\EchoMem
python benchmarks\locomo\scripts\postprocess_judge.py `
  --qa-csv benchmarks\locomo\runs\{timestamp}_locomo_e2e\results\qa_results.csv `
  --judge-token "sk-your-deepseek-token"
```

## 关键指标说明

| 指标 | 计算方式 | 目标参考 |
|------|---------|---------|
| `backend_accuracy` | 路由正确的题数 / 有标签且非 infra 失败的题数 | > 50% |
| `template_hit_rate` | template 命中题数 / 总题数 | > 40% |
| `llm_fallback_rate` | LLM fallback 题数 / 总题数 | < 50% |
| `answer_accuracy` | Judge 判定正确数 / 被 judge 的题数 | 端到端最终指标 |
| `token_savings_rate` | template_hit_rate 本身 | 越高越省 |

## 注意事项

1. **MemRouter 模板来源**：本评测使用的 YAML 路由模板来自 EchoMem 主仓库的 `echomem/templates_data/` 目录，评测目录下不单独维护模板副本。
2. **OpenViking 版本依赖**：补丁基于 OpenViking 2026-05-20 左右的代码。若应用失败，请检查文件是否已被修改，并手动合并。
3. **数据隔离**：确保 `VIKING_SEARCH_USE_LOCAL_CLIENT=true` 环境变量已设置，否则 HTTP server 向量库与索引进程的数据可能不一致。
4. **评测目录自动创建**：`runs/` 目录由脚本自动创建，无需手动准备。

## 相关文件

| 文件 | 说明 |
|------|------|
| `echomem/templates_data/*.yaml` | MemRouter 路由模板（评测直接引用） |
| `echomem/agent_sdk/memrouter_viking_client.py` | MemRouterVikingClient（VikingBot 集成） |
| `reports/` | 历史评测报告与 Dashboard（根目录） |
