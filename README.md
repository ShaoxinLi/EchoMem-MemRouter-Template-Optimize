# EchoMem MemRouter

EchoMem MemRouter 是一个面向 AI Agent 记忆检索场景的路由与查询指令生成引擎。它回答两个核心问题：

> 1. 给定一条用户 query，应该把记忆检索请求路由到哪个记忆后端？
> 2. 该后端应如何执行检索——能否跳过昂贵的意图分析，直接使用什么参数检索？

MemRouter 不执行真实记忆检索，也不决定后端内部如何合并、排序或注入记忆。但它会在路由决策的同时，通过 adapter 生成后端可直接消费的 **Query Instruction**（含 `skip_intent_analysis`、`typed_query`、`target_uri`、`level` 等检索参数），使支持 fast path 的后端能够绕过自身的 VLM IntentAnalyzer，显著降低 token 成本与延迟。

当前 v1.4 范围：

- 在 OpenViking、Graph、Temporal 三类逻辑后端之间进行路由。
- 主路径优先使用低成本的模板多原型 embedding 匹配。
- 当模板匹配低置信或处于灰区时，可启用 LLM 后备路由。
- 输出标准化的后端路由结果（`MemBackendRouteResult`），含可执行的 `query_instructions`。
- 通过 `BackendQueryInstruction` 机制，使 OpenViking 等支持 fast path 的后端跳过 IntentAnalyzer。
- 不执行真实记忆检索，真实检索与记忆合并由后端或 adapter 自行完成。

## 设计目标

- **路由 + 查询指令生成**：MemRouter 输出目标后端，同时通过 adapter 生成后端可直接执行的 Query Instruction（检索参数、skip_intent_analysis 策略等）。不输出后端内部如何合并、排序记忆的策略。
- **低成本优先**：模板多原型 embedding 匹配是主路径，LLM 只作为兜底。
- **后端可扩展**：新增记忆后端时注册 backend、模板和 adapter 即可，不需要改 matcher 主算法。
- **评测可观测**：每轮评测记录模板命中率、LLM fallback 率、后端准确率、延迟和模板分数。
- **Adapter 隔离**：backend adapter 负责将路由结果 enrich 为后端特定的 `BackendQueryInstruction`，包含检索参数、过滤条件、intent 类型等。

## 总体链路

```text
User Query
    |
    v
QueryNormalizer
    |
    v
QueryFeatureBuilder
    |
    v
BackendRouteTemplateIndex
    |
    v
TemplateMatcher
    |
    v
RouteDecision
    |-- 高置信单后端  -> template_embedding
    |-- 高置信多后端  -> template_embedding_multi_backend
    |-- 低置信或灰区  -> llm_backend_fallback
    |
    v
MemBackendRouteResult
    |
    v
Backend Adapter
```

## 安装

```bash
pip install -e .

# 本地 embedding provider
pip install -e ".[local]"

# OpenAI-compatible embedding provider
pip install -e ".[openai]"

# 开发依赖
pip install -e ".[dev]"
```

可选 LLM fallback 依赖：

```bash
pip install -e ".[llm]"
```

## 快速开始

### 基础路由

```python
from echomem.embeddings.base import create_provider
from echomem.pipeline import MemRouterPipeline

embedder = create_provider("sentence-transformers", model_name="all-MiniLM-L6-v2")
pipeline = MemRouterPipeline.with_defaults(embedder)

result = pipeline.route("你还记得我之前跟你说过什么吗？")

print(result.route_method)
print(result.routes[0].backend_id)
print(result.routes[0].role)
print(result.query_instructions[0].model_dump())   # v1.4+: 后端可直接执行的查询指令
```

示例路由结果：

```json
{
  "route_method": "template_embedding",
  "routes": [
    {
      "backend_id": "openviking_memory_backend",
      "role": "primary",
      "confidence": 0.81,
      "matched_template_id": "openviking.personal_fact_lookup.en.v2"
    }
  ],
  "query_instructions": [
    {
      "backend_id": "openviking_memory_backend",
      "query": "你还记得我之前跟你说过什么吗？",
      "search_mode": "find",
      "skip_intent_analysis": true,
      "target_uri": "viking://memories",
      "backend_params": {
        "context_type": "memory",
        "level": [2],
        "typed_query": {
          "query": "你还记得我之前跟你说过什么吗？",
          "context_type": "memory",
          "intent": "personal_fact_lookup",
          "priority": 1
        }
      }
    }
  ]
}
```

## LLM 后备路由

LLM fallback 是可选能力，适合在 benchmark 评测或模板覆盖仍不足的实验阶段启用。

```python
from echomem.llm_fallback import LLMRouterConfig

llm_config = LLMRouterConfig(
    provider="anthropic_compatible",
    model="MiniMax-M2.7",
    api_key_env="MINIMAX_API_KEY",
    base_url="https://api.minimaxi.com/anthropic",
    max_tokens=1024,
    max_secondary_routes=1,
)

pipeline = MemRouterPipeline.with_defaults(embedder, llm_router_config=llm_config)
```

LLM fallback 使用闭集约束：模型只能从 `MemoryBackendRegistry` 已注册且启用的 backend_id 中选择。

## 与 OpenViking 对接

MemRouter v1.4 新增了 **Query Instruction** 机制。当模板匹配高置信命中时，MemRouter 会直接生成包含 `skip_intent_analysis` 标志的后端查询指令，OpenViking 可据此跳过自身的 VLM IntentAnalyzer（节省 3K-15K tokens）。

### 对接方式

**方式一：直接消费 Query Instruction（推荐）**

```python
from echomem.pipeline import MemRouterPipeline
from echomem.embeddings.base import create_provider

embedder = create_provider("sentence-transformers", model_name="all-MiniLM-L6-v2")
pipeline = MemRouterPipeline.with_defaults(embedder)

result = pipeline.route("我最喜欢的颜色是什么？")
inst = result.query_instructions[0]   # primary backend instruction

if inst.backend_id == "openviking_memory_backend" and inst.skip_intent_analysis:
    # Fast path: 直接调用 OpenViking execute_instruction，绕过 IntentAnalyzer
    response = await openviking_client.execute_instruction(inst.model_dump())
else:
    # Normal path: 走 OpenViking 原生 search（含 IntentAnalyzer）
    response = await openviking_client.search(result.routes[0].raw_query)
```

**方式二：通过 MemRouterVikingClient 自动拦截**

```python
from echomem.agent_sdk import MemRouterVikingClient

# 一行替换：把 VikingClient 包进 MemRouterVikingClient
client = MemRouterVikingClient(viking_client=VikingClient(agent_id="shared"))
await client.initialize()

# search() 自动走 MemRouter fast path，其他方法透传
result = await client.search("我最喜欢的颜色是什么？")
await client.commit(session_id="xxx", messages=[...])   # 透传，MemRouter 不拦截
```

### OpenViking 模板与 skip_intent_analysis 策略

| 模板 | intent_family | skip_intent_analysis | search_mode | level | 场景 |
|---|---|---|---|---|---|
| `openviking.personal_fact_lookup.en.v2` | personal_fact_lookup | **true** | `find` | [2] | 查询具体个人事实 |
| `openviking.preference_profile.v1` | preference_profile | **true** | `find` | [0,1,2] | 查询偏好、习惯、profile |
| `openviking.aggregation_summary.v2` | aggregation_summary | false | `search` | [0,1] | 聚合、对比、总结 |
| `openviking.previous_chat_recall.v1` | previous_chat_recall | false | `search` | [2] | 历史对话回忆 |

- `skip_intent_analysis=true` 时，MemRouter 已识别意图并生成 `typed_query`，OpenViking 可直接执行检索。
- `skip_intent_analysis=false` 时（聚合类、对话回忆类），OpenViking 仍需运行 IntentAnalyzer 做复杂意图拆解。

### OpenViking Fast Path

OpenViking 需暴露 `execute_instruction()` 方法以支持 fast path。若接口不可用，MemRouterVikingClient 自动降级为 `VikingClient.find()`。详见 `benchmarks/locomo/patches/` 中的集成补丁。

## 与 VikingBot / Claw 对接

MemRouter 通过 `MemRouterVikingClient` 提供**零代码改动**的对接方式。现有使用 `VikingClient` 的 Agent（VikingBot、Claw、higo 等）只需把初始化处替换一行。

### 修改前

```python
from viking_sdk import VikingClient

client = VikingClient(agent_id="shared")
await client.initialize()

result = await client.search("你还记得我喜欢什么颜色吗")
await client.commit(session_id="xxx", messages=[...])
```

### 修改后

```python
from viking_sdk import VikingClient
from echomem.agent_sdk import MemRouterVikingClient

client = MemRouterVikingClient(viking_client=VikingClient(agent_id="shared"))
await client.initialize()

# 以下代码完全不变
result = await client.search("你还记得我喜欢什么颜色吗")   # MemRouter 自动拦截，可能走 fast path
await client.commit(session_id="xxx", messages=[...])      # 透传，行为不变
```

### 接口对照

MemRouterVikingClient 暴露与 VikingClient 完全一致的方法签名：

| 方法 | MemRouter 是否拦截 | 行为 |
|---|---|---|
| `search(query, ...)` | **是** | 先由 MemRouter 路由；高置信且 `skip_intent_analysis=true` 时走 fast path；否则 fallback 到 `VikingClient.search()` |
| `find(query, ...)` | 否 | 透传 |
| `commit(session_id, messages, ...)` | 否 | 透传 |
| `read_content(uri, level)` | 否 | 透传 |
| `add_resource(local_path, desc)` | 否 | 透传 |
| `list_resources(path, recursive)` | 否 | 透传 |
| `read_user_profile(user_id)` | 否 | 透传 |
| `search_user_memory(query, user_id)` | 否 | 透传 |
| `search_memory(query, user_ids, ...)` | 否 | 透传 |
| `search_experiences(query, limit)` | 否 | 透传 |
| `grep(uri, pattern, ...)` | 否 | 透传 |
| `glob(pattern, uri)` | 否 | 透传 |

MemRouter 是**只读路由层**，所有写入操作（commit、add_resource）均直接透传，不参与、不修改。

## 端到端验证

MemRouter 的端到端验证涵盖单测、fast path token 节省验证、fallback 行为验证和 VikingBot 回归测试。详细步骤和运行命令见 `benchmarks/locomo/README.md`。

### 预期效果

| 指标 | 基准 (纯 OpenViking) | MemRouter 增强后 |
|---|---|---|
| personal_fact_lookup / preference_profile 类 query VLM tokens | ~3K-15K | **0**（fast path） |
| 聚合/对话回忆类 query VLM tokens | ~3K-15K | ~3K-15K（不变） |
| 路由延迟（模板命中） | — | < 50ms |
| 模板命中率 | — | 目标 > 70% |

## 已注册逻辑后端

| backend_id | backend_kind | 路由范围 |
|---|---|---|
| `openviking_memory_backend` | `openviking_native` | 个人语义记忆、profile、preferences、历史对话回忆、宽泛用户上下文 |
| `graph_memory_backend` | `knowledge_graph` | 实体关系、关系遍历、多跳图问题、系统依赖关系 |
| `temporal_memory_backend` | `temporal_store` | 时间线事实、持续时间比较、顺序推理、时间范围问题 |

`BackendEntry.capabilities` 仅保留为 adapter 层元数据。MemRouter 不基于 capability 进行路由。

## 模板匹配

模板包含正向 query prototypes 和 hard negatives。Matcher 使用如下公式给模板打分：

```text
S_pos   = 0.50 * S_max + 0.30 * S_mean@3 + 0.20 * S_centroid
S_final = S_pos - lambda * max(0, delta_neg - M_neg)
```

- `S_max`：最强单个 prototype 命中分数。
- `S_mean@3`：top 3 prototypes 的平均分，降低单条 prototype 偶然高分的影响。
- `S_centroid`：query 与模板整体语义中心的相似度。
- `M_neg`：query 与 hard negative 的安全间隔。

`RouteDecision` 会比较 `S_final` 与模板阈值，输出后端 route 或进入 LLM fallback。

## 评测

### 路由层评测

```bash
python scripts/check_templates.py
python scripts/eval_memrouter.py --dataset data/memrouter_eval/smoke_routes.jsonl --mode ci-smoke-mock
python scripts/eval_memrouter.py --dataset data/memrouter_eval/golden_mini_routes.jsonl --mode full_route_eval --config configs/memrouter_eval.local.yaml
```

产物写入 `runs/`（git 忽略）。核心指标：`primary_backend_accuracy`、`template_hit_rate`、`llm_fallback_rate`、`invalid_route_rate`。

### 端到端评测

LoCoMo + VikingBot + MemRouter + OpenViking 的完整端到端评测位于 `benchmarks/locomo/`，含一键脚本、数据集、补丁和详细文档。详见 [benchmarks/locomo/README.md](benchmarks/locomo/README.md)。

## 项目结构

```text
echomem/
|-- registry.py                    # MemoryBackendRegistry
|-- normalizer.py                  # QueryNormalizer
|-- features.py                    # QueryFeatureBuilder
|-- templates.py                   # MemoryBackendRouteTemplate 与模板索引
|-- matcher.py                     # TemplateMatcher
|-- decision.py                    # RouteDecision
|-- result.py                      # MemBackendRouteResult (v2 含 query_instructions)
|-- request.py                     # MemoryRouteRequest 与 CallerContext
|-- pipeline.py                    # MemRouterPipeline (集成 QueryInstructionBuilder)
|-- llm_fallback.py                # LLM 后备路由实现
|-- query_spec.py                  # TemplateQuerySpec / BackendQueryInstruction 模型
|-- query_instruction_builder.py   # QueryInstructionBuilder (Template-First Hybrid)
|-- embeddings/
|   `-- base.py                    # Embedding providers
|-- adapters/
|   `-- openviking.py              # OpenVikingAdapter (含 get_default_spec / enrich_spec)
|-- agent_sdk/
|   `-- memrouter_viking_client.py # MemRouterVikingClient (VikingBot 零改动对接)
|-- templates_data/                # 当前启用的路由模板 (含 query_spec)
`-- templates_archive/             # 已归档模板

scripts/
|-- build_memrouter_eval_dataset.py
|-- check_templates.py
|-- eval_memrouter.py
`-- archive/                       # 一次性数据生成/调试脚本（已归档）

benchmarks/
`-- locomo/                        # LoCoMo E2E 端到端评测
    `-- scripts/
        |-- eval_locomo_vikingbot_memrouter_e2e.py
        |-- eval_vikingbot_memrouter_e2e.py
        `-- postprocess_judge.py

data/memrouter_eval/               # 小型评测数据集（大型标签文件已移除 git 跟踪）
tests/                             # 单元测试
```

## OpenViking Adapter

`OpenVikingAdapter` 现已扩展为支持 **Query Instruction** 生成。它在模板 `query_spec` 的基础上，根据 query hints（实体、时间提示等）动态 enrich 检索参数，输出 `BackendQueryInstruction`。

### 分工边界

**MemRouter 决定：**

- query 是否应该路由到 OpenViking；
- 路由置信度和命中的模板元数据；
- `skip_intent_analysis` 策略（模板声明为主，adapter 可在运行时 enrich）；
- `typed_query`、`target_uri`、`context_type`、`level` 等具体检索参数。

**OpenViking 自行决定（仅当 `skip_intent_analysis=false` 时）：**

- 使用 search、read、find、graph 或其他内部机制；
- 如何合并、排序、注入检索到的记忆。

**OpenViking 直接执行（当 `skip_intent_analysis=true` 时）：**

- 接收 `BackendQueryInstruction`，直接调用 `execute_instruction()` 检索并返回结果。

### Adapter 扩展点

新增 backend 时，只需实现 `MemoryBackendAdapter` 的两个方法：

```python
class MyBackendAdapter(MemoryBackendAdapter):
    @property
    def backend_id(self) -> str:
        return "my_backend"

    def get_default_spec(self) -> Dict[str, Any]:
        # 返回该后端的默认 query 参数
        return {"some_param": "default_value"}

    def enrich_spec(self, spec: TemplateQuerySpec, hints: QueryHints) -> Dict[str, Any]:
        # 根据 query hints  enrich 参数
        if hints.entities:
            spec.backend_params["filter"] = {"entities": hints.entities}
        return spec.model_dump()
```

## 当前限制

- Graph 和 Temporal 的真实 adapter 尚未实现（仅 OpenViking adapter 已完成）。
- OpenViking 侧需要新增 `execute_instruction()` 接口以支持 fast path；当前 MemRouterVikingClient 在接口不可用时自动降级为 `find()`。
- `local_http` LLM provider 已预留接口，但尚未实现。
- 模板覆盖仍在调优中，尤其是自然分布下的 OpenViking 个人记忆类 query。
- End-to-end 验证依赖 OpenViking 侧 `execute_instruction()` 的实现，目前可用 fallback 路径做渐进式验证。

## 测试状态

当前单元测试基线：`61 passed`。

## License

TBD
