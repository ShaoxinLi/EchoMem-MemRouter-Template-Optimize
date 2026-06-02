# MemRouter Template Online Optimization 实验方案

## 1. 实验目标

本实验目标是设计 MemRouter templates 的 online optimization 流程。

online optimization 指：

```text
从线上请求中记录 Trace，基于 Trace 生成可信训练样本，再复用 GEPA 生成、验证并发布新的 Template Bundle。
```

本方案按以下步骤逐步展开：

1. 记录 Trace。
2. 生成训练样本。
3. 使用 GEPA 生成 Candidate Template Bundle。
4. 验证 Candidate Template Bundle。
5. 发布新的 Template Bundle。

本文档当前定义第 1 步到第 5 步，并补充 Online Simulation 实验设计。

## 2. 术语

| 术语 | 定义 |
| --- | --- |
| `Template Bundle` | MemRouter matcher 消费的一组 YAML template 文件。 |
| `Active Template Bundle` | 当前线上正在使用的 Template Bundle。 |
| `Trace` | 一次线上 memory retrieval 请求的结构化记录。它包含 route、retrieval 和 answer 三部分信息。 |
| `Trace Store` | 存储 Trace 的持久化位置，例如 JSONL 文件、数据库表或日志系统。 |
| `route` | MemRouter 对 query 的路由决策信息，包括命中的 template、backend、score 和 query instruction。 |
| `retrieval` | backend 执行检索后的结果摘要，包括执行路径、返回数量、top result、错误和延迟。 |
| `answer` | Agent 基于 retrieval 结果生成最终回答后的表现信息，包括是否为空答、是否报错、judge 结果或用户反馈。 |
| `template_bundle_id` | Template Bundle 的版本标识。每条 Trace 必须记录它，用于后续 replay 和回滚分析。 |
| `template hit` | MemRouter 通过 template route 接受某个 backend 的情况。 |
| `fast path` | backend 执行时跳过自身 intent analysis 的路径。 |
| `Training Sample` | 用于 GEPA optimization 的训练样本。它至少包含 `question`、`expected_backend` 和 `label_confidence`。 |
| `expected_backend` | Training Sample 中标注的目标 backend。它表示当前 query 应该路由到哪个 backend。 |
| `label_confidence` | `expected_backend` 的可信度分数，范围是 0 到 1。 |
| `Label Source` | 产生 `expected_backend` 的证据来源，例如 user feedback、answer judge、counterfactual retrieval 或 human review。 |
| `Rejected Trace` | 被判断为不适合生成 Training Sample 的 Trace。 |
| `Minibatch` | GEPA 每轮使用的一小批 Training Sample。 |
| `Proposal LLM` | 根据当前 Template Bundle 和 feedback 生成模板修改内容的 LLM。 |
| `Complete Editable Fields Bundle` | Proposal LLM 的输出。它只包含每个 template 的 `query_prototypes`、`hard_negatives`、`thresholds`。 |
| `Validator` | 检查 Template Bundle 是否合法、可加载、可运行的固定程序。 |
| `Deterministic Feedback` | 固定代码根据 routing 结果和 Training Sample 生成的结构化反馈。 |
| `LLM Feedback` | LLM 根据 Deterministic Feedback 生成的泛化修改建议。 |
| `GEPA Train Set` | 输入 GEPA optimization 的 Training Sample 集合，用于生成 feedback 和采样 Minibatch。 |
| `GEPA Candidate Set` | GEPA 运行中用于初步比较候选效果的 Training Sample 集合。它不等同于最终验证集。 |
| `Candidate Template Bundle` | GEPA 生成并通过 Validator 的 Template Bundle。它只是候选版本，不代表可以发布。 |
| `GEPA Run Report` | 记录一次 GEPA 运行过程、主要修改、分数和 Validator 结果的报告。 |
| `Verification Set` | 第 4 步用于最终验证 Candidate Template Bundle 的样本集合。它不参与 GEPA 生成过程。 |
| `Regression Anchor` | 已知行为正确、用于防止 Candidate Template Bundle 破坏原有能力的 Training Sample。 |
| `Offline Regression Set` | 固定的离线回归样本集合，用于防止 Candidate Template Bundle 破坏已有稳定能力。 |
| `Verification Run` | 在同一批样本上对比 Active Template Bundle 和 Candidate Template Bundle 的验证过程。 |
| `Verification Report` | Verification Run 的输出报告，包含对比指标、regression 和通过/拒绝结论。 |
| `Verification Regression` | Active Template Bundle 路由正确，但 Candidate Template Bundle 路由错误的样本。 |
| `Bundle Registry` | 存放所有 Template Bundle 版本和元数据的位置。简单说，它是 Template Bundle 的版本仓库。 |
| `Active Bundle Pointer` | 指向当前线上 Active Template Bundle 的配置。简单说，切换它就等于切换线上使用的模板版本。 |
| `Shadow Run` | 只让 Candidate Template Bundle 在后台计算 route，但不影响用户结果。简单说，它是“偷偷试跑”。 |
| `Canary Release` | 只让很小比例的线上请求使用 Candidate Template Bundle。简单说，它是“小范围试用”。 |
| `Rollout` | 逐步扩大 Candidate Template Bundle 的使用比例。简单说，它是“分阶段放量”。 |
| `Rollback` | 把线上 Template Bundle 切回上一个稳定版本。简单说，它是“快速撤回”。 |
| `Release Gate` | 发布前必须满足的检查条件。简单说，它是“能不能继续发布的门槛”。 |
| `Release Report` | 记录发布过程、指标和最终结果的报告。 |
| `Online Simulation` | 用 offline 数据模拟 online optimization 流程的实验。简单说，它是在真实线上链路完成前，先用离线数据演练 online 流程。 |
| `Trace Simulation` | 用 offline 数据生成模拟 Trace 的实验。简单说，它是“先造出像线上一样的 Trace”。 |
| `Training Sample Builder` | 从 Trace 生成 Training Sample 的程序或流程。简单说，它是“把 Trace 变成 GEPA 能用的训练样本”。 |
| `Noisy Label` | 错误或不可靠的 `expected_backend`。简单说，它是“脏标签”。 |
| `Online Replay` | 按时间顺序重放历史样本，模拟线上流量持续到来的过程。简单说，它是“把历史数据当成线上流量重新播放”。 |
| `Simulation Batch` | Online Replay 中的一批模拟线上样本。简单说，它是“一段时间内来的请求集合”。 |

## 3. Step 1：记录 Trace

### 3.1 目标

记录 Trace 的目标是为后续生成训练样本提供 evidence。

Trace 不直接等于训练样本。Trace 只回答：

```text
这次 query 是怎么被路由、怎么被检索、最后回答表现如何？
```

训练样本生成会在下一步处理，它才会判断：

```text
这条 Trace 是否可信，是否可以转成 <query, expected_backend>。
```

### 3.2 设计原则

1. Trace 记录不能影响线上请求成功率。
2. Trace 记录不能显著增加线上延迟。
3. Trace 必须支持 replay。
4. Trace 必须能区分 route 问题和 retrieval / answer 问题。
5. Trace 必须记录 `template_bundle_id`。
6. Trace 默认不保存完整记忆内容，只保存结果摘要和可追溯引用。

### 3.3 Trace 边界

Trace 只记录一次 memory retrieval 请求的事实。

Trace 记录：

- query 输入。
- MemRouter route 结果。
- backend retrieval 结果摘要。
- Agent answer 结果摘要。
- 错误、延迟和执行路径。
- `template_bundle_id`。

Trace 不负责：

- 判断 expected backend。
- 生成训练样本。
- 修改 Template Bundle。
- 调用 GEPA。
- 发布新 Template Bundle。

### 3.4 Trace 字段

Trace 建议使用以下结构。

```json
{
  "trace_id": "string",
  "timestamp": "string",
  "template_bundle_id": "string",
  "query": {
    "raw_user_query": "string",
    "normalized_user_query": "string",
    "caller": "string",
    "user_id_hash": "string",
    "session_id_hash": "string"
  },
  "route": {
    "route_method": "string",
    "primary_backend_id": "string",
    "matched_template_id": "string",
    "confidence": 0.0,
    "top_templates": [],
    "top_backends": [],
    "query_instruction": {},
    "fallback_used": false,
    "fallback_reason": ""
  },
  "retrieval": {
    "execution_path": "string",
    "executed_backend_id": "string",
    "skip_intent_analysis": false,
    "result_count": 0,
    "top_results": [],
    "latency_ms": 0,
    "error": ""
  },
  "answer": {
    "status": "unknown",
    "answer_id": "string",
    "answer_length": 0,
    "judge_score": null,
    "user_feedback": null,
    "followup_signal": null,
    "error": ""
  }
}
```

### 3.5 字段说明

`trace_id`：

```text
一次 memory retrieval 请求的唯一标识。
```

`template_bundle_id`：

```text
当前线上使用的 Template Bundle 版本。
```

`query.raw_user_query`：

```text
用户原始 query。
```

`query.normalized_user_query`：

```text
MemRouter 实际用于 embedding 的归一化 query。
```

`route.top_templates`：

```text
MemRouter matcher 的 top template 列表。至少包含 template_id、backend_id、score 和 score_components。
```

`route.top_backends`：

```text
MemRouter 聚合后的 backend 排名。至少包含 backend_id、best_template_id 和 score。
```

`route.query_instruction`：

```text
MemRouter 输出给 backend 的 query instruction。
```

`retrieval.execution_path`：

```text
实际执行路径。例如 fast_path、native_search、no_instructions_fallback。
```

`retrieval.top_results`：

```text
检索结果摘要。建议只保存 uri、score、context_type 和短摘要，不默认保存完整记忆内容。
```

`answer.status`：

```text
最终回答状态。建议取值：unknown、answered、empty_answer、short_answer、error。
```

`answer.judge_score`：

```text
可选字段。answer judge 给出的回答质量分数。
```

`answer.user_feedback`：

```text
可选字段。用户显式反馈，例如 positive、negative。
```

`answer.followup_signal`：

```text
可选字段。用户后续行为信号，例如 correction、retry、clarification。
```

### 3.6 写入策略

Trace 可以分两阶段写入。

第一阶段在 memory retrieval 完成后写入：

```text
query + route + retrieval
```

第二阶段在 answer 生成或用户反馈出现后补充：

```text
answer
```

如果 Trace Store 是 JSONL，第二阶段可以追加一条带相同 `trace_id` 的更新记录。后续处理时按 `trace_id` 合并。

如果 Trace Store 是数据库，第二阶段可以更新同一条 Trace。

### 3.7 最小可行实现

第一版只需要记录以下字段：

- `trace_id`
- `timestamp`
- `template_bundle_id`
- `query.raw_user_query`
- `query.normalized_user_query`
- `route.route_method`
- `route.primary_backend_id`
- `route.matched_template_id`
- `route.confidence`
- `route.top_templates`
- `route.query_instruction`
- `retrieval.execution_path`
- `retrieval.executed_backend_id`
- `retrieval.skip_intent_analysis`
- `retrieval.result_count`
- `retrieval.latency_ms`
- `retrieval.error`
- `answer.status`
- `answer.error`

这些字段足够支持下一步生成初版训练样本。

### 3.8 质量要求

Trace 记录需要满足以下要求：

1. 每条 Trace 必须有 `trace_id`。
2. 每条 Trace 必须有 `template_bundle_id`。
3. 每条 Trace 必须记录 route 的 top templates。
4. 每条 Trace 必须记录实际 `execution_path`。
5. retrieval error 不能被记录成 route error。
6. answer error 不能被记录成 route error。
7. Trace 写入失败不能影响线上 retrieval 返回。

### 3.9 本步骤输出

本步骤输出是 Trace Store。

Trace Store 后续会作为第 2 步“生成训练样本”的输入。

本步骤不输出训练样本，也不修改 Template Bundle。

## 4. Step 2：从 Trace 生成可信训练样本

### 4.1 目标

本步骤目标是从 Trace Store 中生成可信的 Training Sample。

Training Sample 用于后续 GEPA optimization。

本步骤只回答：

```text
哪些 Trace 可以转成 Training Sample？
如果可以，它的 expected_backend 是什么，可信度是多少？
```

本步骤不负责：

- 生成 Candidate Template Bundle。
- 验证 Candidate Template Bundle。
- 发布新的 Template Bundle。

### 4.2 输入与输出

输入：

```text
Trace Store
```

输出：

```text
Training Sample Set
Rejected Trace Set
```

`Training Sample Set` 用于 GEPA optimization。

`Rejected Trace Set` 用于人工分析和规则改进，不进入 GEPA optimization。

### 4.3 Training Sample 结构

Training Sample 建议使用以下结构。

```json
{
  "sample_id": "string",
  "trace_id": "string",
  "question": "string",
  "expected_backend": "string",
  "label_confidence": 0.0,
  "label_source": "string",
  "route_context": {
    "template_bundle_id": "string",
    "production_backend": "string",
    "matched_template_id": "string",
    "route_method": "string",
    "confidence": 0.0
  },
  "evidence": {
    "answer_status": "string",
    "judge_score": null,
    "user_feedback": null,
    "followup_signal": null,
    "retrieval_error": "",
    "answer_error": "",
    "notes": []
  }
}
```

### 4.4 字段说明

`sample_id`：

```text
Training Sample 的唯一标识。
```

`trace_id`：

```text
Training Sample 来源 Trace 的唯一标识。
```

`question`：

```text
用于 GEPA optimization 的 query 文本。默认使用 Trace 中的 raw_user_query。
```

`expected_backend`：

```text
该 query 应该路由到的 backend。
```

`label_confidence`：

```text
expected_backend 的可信度。GEPA optimization 可以使用它作为样本权重。
```

`label_source`：

```text
expected_backend 的来源。建议取值：human_review、user_feedback、answer_judge、counterfactual_retrieval、production_success。
```

`route_context`：

```text
生成样本时的线上 route 背景。它用于后续分析，不作为 expected_backend 的唯一依据。
```

`evidence`：

```text
生成 Training Sample 的证据摘要。
```

### 4.5 样本生成原则

1. 不是所有 Trace 都能生成 Training Sample。
2. `expected_backend` 不能简单等于线上实际路由 backend。
3. retrieval error 和 answer error 不能直接归因给 route。
4. `label_confidence` 低的 Trace 不进入 Training Sample Set。
5. 同一 query 的重复 Trace 需要去重或降权。
6. Training Sample 需要保留 `trace_id`，方便回溯证据。

### 4.6 可信度分级

建议使用以下 `label_confidence` 分级。

| 级别 | `label_confidence` | 含义 |
| --- | ---: | --- |
| high | 0.90 - 1.00 | label 很可信，可以直接进入 GEPA optimization。 |
| medium | 0.70 - 0.89 | label 基本可信，可以进入 GEPA optimization，但建议降权。 |
| low | 0.40 - 0.69 | label 不够可信，不进入 GEPA optimization。 |
| rejected | 0.00 - 0.39 | Trace 被拒绝。 |

第一版建议只使用：

```text
label_confidence >= 0.70
```

的 Training Sample。

### 4.7 Label Source 规则

#### 4.7.1 human_review

人工审核可以直接给出 `expected_backend`。

规则：

```text
如果 human_review 明确给出 expected_backend，则 label_confidence = 1.00。
```

适用场景：

- 高价值 query。
- 模型判断冲突的 query。
- 多次被用户纠正的 query。

#### 4.7.2 user_feedback

用户显式反馈可以作为强证据。

规则：

```text
如果用户明确表示当前回答错误，并且后续 correction 指向某个 backend，则使用 correction 指向的 backend。
```

推荐可信度：

```text
label_confidence = 0.90
```

限制：

```text
如果用户只表达“不满意”，但没有足够信息判断 backend，则不生成 Training Sample。
```

#### 4.7.3 answer_judge

answer judge 可以作为中等证据。

规则：

```text
如果生产路径 answer judge 分数高，并且没有 retrieval error 或 answer error，可以把生产 backend 作为 expected_backend。
```

推荐可信度：

```text
label_confidence = 0.75
```

限制：

```text
如果 answer judge 分数低，不能直接说明 route 错误。
```

#### 4.7.4 counterfactual_retrieval

counterfactual retrieval 指对同一 query 额外执行另一个 backend 或执行路径，用于判断是否存在更好的 route。

规则：

```text
如果生产 backend answer 失败，counterfactual backend answer 成功，则使用 counterfactual backend 作为 expected_backend。
```

推荐可信度：

```text
label_confidence = 0.80
```

限制：

```text
如果生产 backend 和 counterfactual backend 都失败，则不生成 Training Sample。
```

#### 4.7.5 production_success

production success 指线上生产路径成功回答。

规则：

```text
如果生产路径 answer 成功，且没有 error，可以把生产 backend 作为 expected_backend。
```

推荐可信度：

```text
label_confidence = 0.70
```

限制：

```text
production_success 只适合作为保留现有行为的样本，不适合作为修复错误的主要样本。
```

### 4.8 Rejected Trace 规则

以下 Trace 不生成 Training Sample：

1. 缺少 `trace_id`。
2. 缺少 `template_bundle_id`。
3. 缺少 route 信息。
4. 缺少 retrieval 信息。
5. query 为空。
6. retrieval error 明显来自 backend 服务异常。
7. answer error 明显来自生成模型异常。
8. 用户反馈不明确。
9. answer judge 分数低，但没有 counterfactual 证据。
10. 多个 Label Source 给出冲突的 `expected_backend`。

Rejected Trace 需要记录拒绝原因。

示例：

```json
{
  "trace_id": "string",
  "reject_reason": "answer_error_not_route_error",
  "notes": ["answer generation failed, route cannot be judged"]
}
```

### 4.9 去重与采样

同一 query 可能出现多次。

第一版使用简单规则：

1. 对 `normalized_user_query` 相同的 Trace 做分组。
2. 如果同组内 `expected_backend` 一致，保留 `label_confidence` 最高的样本。
3. 如果同组内 `expected_backend` 冲突，将该组放入 Rejected Trace Set。
4. 对同一用户或同一 session 的高频 query 降权，避免少数用户主导 optimization。

### 4.10 输出文件建议

第一版可以输出两个 JSONL 文件。

Training Sample Set：

```text
online_train_samples.jsonl
```

Rejected Trace Set：

```text
online_rejected_traces.jsonl
```

`online_train_samples.jsonl` 中每行是一个 Training Sample。

`online_rejected_traces.jsonl` 中每行是一个 Rejected Trace。

### 4.11 本步骤输出

本步骤输出 Training Sample Set。

Training Sample Set 后续会作为第 3 步“使用 GEPA 生成 Candidate Template Bundle”的输入。

本步骤不修改 Template Bundle。

## 5. Step 3：使用 GEPA 生成 Candidate Template Bundle

### 5.1 目标

本步骤目标是复用 offline GEPA optimization 流程，基于 Training Sample Set 生成 Candidate Template Bundle。

本步骤只回答：

```text
能否基于当前 Training Sample Set 生成一个看起来更好的 Template Bundle？
```

本步骤不负责：

- 判断 Candidate Template Bundle 是否可以上线。
- 发布 Candidate Template Bundle。
- 修改线上正在使用的 Template Bundle。

### 5.2 与 offline GEPA 的关系

online 模式复用 offline GEPA 的核心边界：

1. 固定 matcher 代码。
2. 固定 matcher 配置。
3. 固定 backend 集合。
4. 固定 template 文件集合。
5. 不修改 `template_id`、`target`、`query_spec` 等结构字段。
6. 只允许修改 `query_prototypes`、`hard_negatives`、`thresholds`。
7. Proposal LLM 仍输出 `Complete Editable Fields Bundle`。
8. Candidate Template Bundle 必须通过 Validator。

online 模式与 offline 模式的主要区别：

1. offline 输入来自人工标注数据集。
2. online 输入来自 Training Sample Set。
3. online Training Sample 带有 `label_confidence` 和 `label_source`。
4. online Training Sample 可能有噪声，需要过滤和降权。
5. online 不能把完整 Trace 直接喂给 Proposal LLM，只能使用压缩后的 Training Sample evidence。
6. online 生成的 Candidate Template Bundle 必须经过第 4 步验证后才能发布。

### 5.3 输入与输出

输入：

```text
Active Template Bundle
Training Sample Set
```

输出：

```text
Candidate Template Bundle
GEPA Run Report
```

`Active Template Bundle` 指当前线上正在使用的 Template Bundle。

`GEPA Run Report` 记录 GEPA 运行过程、Validator 结果、Candidate Template Bundle score 和主要修改摘要。

### 5.4 Training Sample 划分

Training Sample Set 需要划分为三部分。

| 集合 | 用途 |
| --- | --- |
| `GEPA Train Set` | 用于 GEPA Minibatch、feedback 和 proposal 生成。 |
| `GEPA Candidate Set` | 用于 GEPA 内部初步比较 Candidate Template Bundle。 |
| `Verification Set` | 保留给第 4 步最终验证，不参与本步骤。 |

关键约束：

1. `Verification Set` 不能进入 GEPA prompt。
2. `Verification Set` 不能参与 GEPA early stop。
3. `Verification Set` 不能参与 Candidate Template Bundle 的生成。

第一版建议：

```text
GEPA Train Set: 70%
GEPA Candidate Set: 15%
Verification Set: 15%
```

如果 Training Sample Set 数量较少，优先保证 `Verification Set` 有足够 high confidence 样本。

### 5.5 样本权重

online Training Sample 有不同可信度。

第一版建议使用简单规则：

```text
只使用 label_confidence >= 0.70 的 Training Sample。
```

如果 GEPA runner 支持样本权重，则使用：

```text
sample_weight = label_confidence
```

如果 GEPA runner 暂不支持样本权重，则：

1. `label_confidence >= 0.90` 的样本正常使用。
2. `0.70 <= label_confidence < 0.90` 的样本可以进入训练，但在 report 中单独统计。
3. `label_confidence < 0.70` 的样本不进入 GEPA。

### 5.6 Feedback 输入

GEPA 的 feedback 仍然由 Deterministic Feedback 和 LLM Feedback 组成。

online 模式需要在 Deterministic Feedback 中增加以下字段：

- `label_confidence`
- `label_source`
- `trace_id`
- `answer_status`
- `retrieval_error`
- `answer_error`

这些字段的作用是提醒 Proposal LLM：

```text
哪些样本是强证据，哪些样本只是弱证据。
```

注意：

```text
Proposal LLM 不应该看到完整 Trace。
```

原因是完整 Trace 可能包含过多线上细节，容易导致 Proposal LLM 过拟合具体用户、具体事件或具体 session。

### 5.7 Proposal 输出约束

Proposal LLM 输出仍使用 offline 模式中的 `Complete Editable Fields Bundle`。

必须满足：

1. 每个 template 文件都输出完整的 `query_prototypes`、`hard_negatives`、`thresholds`。
2. 不新增 template 文件。
3. 不删除 template 文件。
4. 不修改 `template_id`。
5. 不修改 `target`。
6. 不修改 `query_spec`。
7. 不复制 Training Sample 中的 exact question。
8. 不生成过窄的用户特定 prototype。
9. 不生成包含具体用户、具体 session、具体日期或具体私有事件的 prototype。

### 5.8 Validator

Validator 仍是硬门槛。

Validator 需要检查：

1. Template Bundle 可解析。
2. Template Bundle 可被 matcher 加载。
3. 文件集合与 Active Template Bundle 一致。
4. 不可编辑字段未被修改。
5. 只修改 `query_prototypes`、`hard_negatives`、`thresholds`。
6. 新增文本没有 exact copy Training Sample question。
7. 新增文本没有明显用户私有信息。
8. matcher smoke test 可运行。

Validator 失败时：

```text
不产生 Candidate Template Bundle。
```

Validator 通过时：

```text
产生 Candidate Template Bundle。
```

### 5.9 Online 模式的注意点

#### 5.9.1 不要过度降低 thresholds

online 样本通常来自失败或边界 case。

如果 GEPA 过度降低 `thresholds.accept` 或 `thresholds.margin`，可能导致更多错误 template hit。

因此需要在 feedback 和 Validator 中强调：

```text
优先添加泛化 prototypes 或 hard_negatives，不优先大幅降低 thresholds。
```

#### 5.9.2 区分 route 问题和执行问题

如果 Trace 显示 retrieval error 或 answer error，不能直接要求 GEPA 修改 Template Bundle。

只有当 Training Sample 已经明确给出可信 `expected_backend` 时，才进入本步骤。

#### 5.9.3 保留 production_success 样本

online optimization 不能只学习失败样本。

`GEPA Train Set` 中需要包含一部分 `production_success` 样本，作为 Regression Anchor。

目的：

```text
防止 Candidate Template Bundle 修复少数失败 case，同时破坏原来正确的主流 case。
```

#### 5.9.4 控制近期流量偏置

Training Sample Set 可能被近期热点 query 主导。

第一版需要按以下维度做基本均衡：

- `expected_backend`
- `label_source`
- `template_bundle_id`
- 时间窗口

如果某个 backend 样本过多，需要采样降权。

### 5.10 本步骤输出

本步骤输出 Candidate Template Bundle 和 GEPA Run Report。

Candidate Template Bundle 后续会进入第 4 步“验证 Candidate Template Bundle”。

本步骤不发布 Candidate Template Bundle。

## 6. Step 4：验证 Candidate Template Bundle

### 6.1 目标

本步骤目标是判断 Candidate Template Bundle 是否真的优于 Active Template Bundle。

本步骤只回答：

```text
Candidate Template Bundle 是否值得进入发布流程？
```

本步骤不负责：

- 生成新的 Candidate Template Bundle。
- 修改线上 Active Template Bundle。
- 发布 Candidate Template Bundle。

### 6.2 核心判断

Candidate Template Bundle 不能只因为 GEPA score 更高就进入发布流程。

它必须同时满足：

1. 在 Verification Set 上优于 Active Template Bundle。
2. 没有明显增加 Verification Regression。
3. 没有破坏 Offline Regression Set。
4. 没有明显增加错误 template hit。
5. 没有明显增加 fast path 风险。

### 6.3 输入与输出

输入：

```text
Active Template Bundle
Candidate Template Bundle
Verification Set
Offline Regression Set
GEPA Run Report
```

输出：

```text
Verification Report
```

`Verification Report` 必须给出一个明确结论：

```text
pass
reject
needs_review
```

### 6.4 Verification Run

Verification Run 使用同一批样本分别运行：

```text
Active Template Bundle
Candidate Template Bundle
```

对每条样本记录：

- `expected_backend`
- `label_confidence`
- Active 的 `predicted_backend`
- Candidate 的 `predicted_backend`
- Active 的 `matched_template_id`
- Candidate 的 `matched_template_id`
- Active 的 `route_method`
- Candidate 的 `route_method`
- Active 的 `confidence`
- Candidate 的 `confidence`
- 是否是 Verification Regression

关键约束：

1. Verification Run 不调用 Proposal LLM。
2. Verification Run 不修改 Candidate Template Bundle。
3. Verification Run 不使用 GEPA Train Set。
4. Verification Run 不使用 GEPA Candidate Set 作为最终结论依据。
5. Active 和 Candidate 必须使用相同 matcher 代码、embedding provider 和 route decision 配置。

### 6.5 核心指标

Verification Report 至少包含以下指标。

| 指标 | 含义 |
| --- | --- |
| `weighted_backend_accuracy` | 按 `label_confidence` 加权后的 backend accuracy。 |
| `macro_backend_recall` | 按 backend 分组后的平均 recall。 |
| `verification_regression_count` | Verification Regression 数量。 |
| `verification_regression_rate` | Verification Regression 占比。 |
| `improvement_count` | Active 错误但 Candidate 正确的样本数。 |
| `no_decision_rate` | Candidate 没有接受任何 backend 的比例。 |
| `template_accept_rate` | Candidate 接受 template route 的比例。 |
| `fast_path_rate` | Candidate 可能触发 fast path 的比例。 |
| `per_backend_recall` | 每个 backend 的 recall。 |
| `top_confusion_pairs` | 常见错误路由方向。 |

`weighted_backend_accuracy` 的计算方式：

```text
sum(label_confidence * backend_correct) / sum(label_confidence)
```

### 6.6 通过标准

第一版建议使用以下通过标准。

Candidate Template Bundle 必须同时满足：

1. `weighted_backend_accuracy` 高于 Active Template Bundle。
2. `macro_backend_recall` 不低于 Active Template Bundle。
3. `verification_regression_rate <= 0.02`。
4. `Offline Regression Set` 上的 `weighted_backend_accuracy` 不下降。
5. 任一 backend 的 `per_backend_recall` 下降不超过 0.03。
6. `template_accept_rate` 没有异常升高。
7. `fast_path_rate` 没有异常升高。

如果样本量较小，不使用硬性百分比结论，结果应为：

```text
needs_review
```

### 6.7 Verification Regression 分析

每个 Verification Regression 必须记录原因。

建议分类：

| 类型 | 含义 |
| --- | --- |
| `threshold_too_low` | Candidate 降低 threshold 后错误接受。 |
| `prototype_overbroad` | Candidate 的 prototype 过宽，吸收了错误 query。 |
| `hard_negative_removed` | Candidate 删除或弱化 hard negative 后导致错误。 |
| `backend_boundary_shift` | Candidate 改变了 backend 边界。 |
| `unknown` | 暂时无法判断。 |

如果 Verification Regression 集中在同一个 template，该 Candidate Template Bundle 应优先 reject。

### 6.8 Offline Regression Set

Offline Regression Set 用于防止 online optimization 过拟合近期线上样本。

Offline Regression Set 应包含：

- 原有 offline test set。
- synthetic hard negative。
- 历史 production_success 样本。
- 每个 backend 的稳定代表样本。

Offline Regression Set 不参与 GEPA 生成。

Offline Regression Set 只用于验证。

### 6.9 Fast Path 风险检查

Candidate Template Bundle 可能改变 matched template，从而改变 `query_instruction` 和 `skip_intent_analysis`。

因此需要检查：

1. `fast_path_rate` 是否明显升高。
2. 新增 fast path 样本是否集中在低 confidence 区间。
3. 新增 fast path 样本是否来自 medium confidence Training Sample。
4. 新增 fast path 样本是否有更高的 Verification Regression。

如果 Candidate Template Bundle 只是提升 route accuracy，但明显增加低置信 fast path，则结论应为：

```text
needs_review
```

或：

```text
reject
```

### 6.10 Verification Report 结构

Verification Report 建议使用以下结构。

```json
{
  "candidate_bundle_id": "string",
  "active_bundle_id": "string",
  "decision": "pass",
  "summary": {
    "weighted_backend_accuracy_delta": 0.0,
    "macro_backend_recall_delta": 0.0,
    "verification_regression_count": 0,
    "verification_regression_rate": 0.0,
    "fast_path_rate_delta": 0.0
  },
  "metrics": {
    "active": {},
    "candidate": {}
  },
  "regressions": [],
  "improvements": [],
  "risk_notes": []
}
```

### 6.11 本步骤输出

本步骤输出 Verification Report。

如果 Verification Report 的 `decision = pass`，Candidate Template Bundle 可以进入第 5 步“发布新的 Template Bundle”。

如果 Verification Report 的 `decision = reject`，Candidate Template Bundle 不进入发布流程。

如果 Verification Report 的 `decision = needs_review`，需要人工确认后再决定是否进入发布流程。

## 7. Step 5：发布新的 Template Bundle

### 7.1 目标

本步骤目标是把通过验证的 Candidate Template Bundle 安全地变成新的 Active Template Bundle。

本步骤只回答：

```text
如何让新的 Template Bundle 上线，同时保留观察、暂停和回滚能力？
```

本步骤不负责：

- 生成 Candidate Template Bundle。
- 重新验证 Candidate Template Bundle。
- 修改 matcher 代码。

### 7.2 发布原则

发布新的 Template Bundle 必须遵守以下原则：

1. 每次发布必须有唯一 `template_bundle_id`。
2. 每次发布必须能回滚。
3. 发布过程必须记录 Release Report。
4. 发布不能直接全量替换。
5. 发布必须先经过 Shadow Run。
6. Canary Release 阶段必须持续记录 Trace。
7. 如果核心指标异常，必须停止 Rollout 或执行 Rollback。

### 7.3 输入与输出

输入：

```text
Candidate Template Bundle
Verification Report
Active Template Bundle
```

输出：

```text
New Active Template Bundle
Release Report
```

只有当 Verification Report 的 `decision = pass` 时，Candidate Template Bundle 才能进入发布流程。

如果 Verification Report 的 `decision = needs_review`，必须人工确认后才能进入发布流程。

如果 Verification Report 的 `decision = reject`，不能进入发布流程。

### 7.4 发布前准备

发布前需要完成以下检查。

1. Candidate Template Bundle 已通过 Validator。
2. Verification Report 的结论是 `pass` 或人工确认后的 `needs_review`。
3. Candidate Template Bundle 已写入 Bundle Registry。
4. Candidate Template Bundle 有唯一 `template_bundle_id`。
5. 当前 Active Template Bundle 的 `template_bundle_id` 已记录为 rollback target。
6. Trace 中能记录新的 `template_bundle_id`。

`rollback target` 指回滚目标版本。简单说，就是“如果新版本出问题，要切回哪个旧版本”。

### 7.5 Shadow Run

Shadow Run 指：

```text
线上请求仍然使用 Active Template Bundle 返回结果，
但后台同时用 Candidate Template Bundle 计算一次 route。
```

简单说：

```text
新版本只在后台试跑，不影响用户。
```

Shadow Run 需要记录：

- Active 的 route 结果。
- Candidate 的 route 结果。
- Active 和 Candidate 是否一致。
- Candidate 是否产生新的 template hit。
- Candidate 是否产生新的 fast path。
- Candidate 是否明显改变 backend 分布。

Shadow Run 通过条件：

1. Candidate 没有明显增加高风险 route。
2. Candidate 的 backend 分布没有异常偏移。
3. Candidate 没有明显增加低 confidence fast path。
4. Shadow Run 期间没有运行时错误。

如果 Shadow Run 不通过：

```text
停止发布，不进入 Canary Release。
```

### 7.6 Canary Release

Canary Release 指：

```text
只让很小比例的真实线上请求使用 Candidate Template Bundle。
```

简单说：

```text
新版本先给少量请求真正使用，观察是否稳定。
```

第一版建议：

```text
Canary Release 比例 = 1% 到 5%
```

Canary Release 必须观察：

- route error。
- retrieval error。
- answer error。
- `template_accept_rate`。
- `fast_path_rate`。
- 用户负反馈。
- latency。

Canary Release 通过条件：

1. route error 没有明显升高。
2. retrieval error 没有明显升高。
3. answer error 没有明显升高。
4. 用户负反馈没有明显升高。
5. latency 没有明显升高。
6. `fast_path_rate` 没有异常升高。

如果 Canary Release 不通过：

```text
执行 Rollback。
```

### 7.7 Rollout

Rollout 指逐步扩大 Candidate Template Bundle 的使用比例。

简单说：

```text
先小范围用，再逐步放大到全量。
```

第一版建议使用以下阶段：

| 阶段 | 使用比例 | 说明 |
| --- | ---: | --- |
| Shadow Run | 0% | 只后台试跑，不影响用户。 |
| Canary Release | 1% - 5% | 少量真实请求使用。 |
| Stage 1 | 10% | 小范围扩大。 |
| Stage 2 | 25% | 中等范围扩大。 |
| Stage 3 | 50% | 半量使用。 |
| Full Release | 100% | 全量使用，Candidate Template Bundle 成为新的 Active Template Bundle。 |

每个阶段都必须检查 Release Gate。

Release Gate 指：

```text
进入下一阶段前必须满足的指标门槛。
```

简单说：

```text
如果指标不好，就不能继续放量。
```

### 7.8 Rollback

Rollback 指把线上 Template Bundle 切回上一个稳定版本。

简单说：

```text
新版本出问题时，快速切回旧版本。
```

触发 Rollback 的条件包括：

1. route error 明显升高。
2. retrieval error 明显升高。
3. answer error 明显升高。
4. 用户负反馈明显升高。
5. latency 明显升高。
6. `fast_path_rate` 异常升高。
7. 出现无法解释的 backend 分布偏移。

Rollback 操作应该只修改 Active Bundle Pointer。

也就是说：

```text
不要删除 Candidate Template Bundle，只把线上指针切回旧版本。
```

这样做的原因是：

```text
保留 Candidate Template Bundle 方便后续分析问题。
```

### 7.9 Active Bundle Pointer

Active Bundle Pointer 指向当前线上使用的 Template Bundle。

简单说：

```text
它是一条配置，告诉 MemRouter 当前应该加载哪个 Template Bundle。
```

发布成功时：

```text
Active Bundle Pointer -> Candidate Template Bundle
```

回滚时：

```text
Active Bundle Pointer -> rollback target
```

关键要求：

1. Active Bundle Pointer 切换必须是原子操作。
2. 每次切换必须记录旧值和新值。
3. 每条 Trace 必须记录切换后的 `template_bundle_id`。

`原子操作` 指一次切换要么完全成功，要么完全失败。简单说，不能出现一半请求用新版本、一半请求拿不到配置的中间状态。

### 7.10 Release Report 结构

Release Report 建议使用以下结构。

```json
{
  "release_id": "string",
  "candidate_bundle_id": "string",
  "previous_active_bundle_id": "string",
  "final_active_bundle_id": "string",
  "verification_report_id": "string",
  "decision": "released",
  "stages": [
    {
      "stage": "shadow_run",
      "traffic_percent": 0,
      "decision": "pass",
      "metrics": {}
    }
  ],
  "rollback": {
    "used": false,
    "reason": "",
    "rollback_target_bundle_id": ""
  },
  "risk_notes": []
}
```

### 7.11 本步骤输出

本步骤输出 Release Report。

如果发布成功：

```text
Candidate Template Bundle 成为新的 Active Template Bundle。
```

如果发布失败：

```text
Active Template Bundle 保持不变，或者通过 Rollback 切回 rollback target。
```

## 8. Online Simulation 实验设计

### 8.1 目标

当前 EchoAgent 的完整链路还没有稳定完成：

```text
HIGO -> EchoMem -> MemRouter -> MultiMemBackend
```

因此现阶段不能验证真实 online optimization 效果。

但可以先做 Online Simulation。

Online Simulation 指：

```text
不用真实线上流量，而是用 LoCoMo 或其他 offline 数据模拟 Trace、Training Sample、Verification Run 和发布前判断。
```

简单说：

```text
在真实线上链路完成前，先验证 online optimization 流程是否可运行、是否抗噪声、是否能发现风险。
```

### 8.2 实验边界

Online Simulation 验证：

- Trace 结构是否够用。
- Training Sample 生成规则是否可执行。
- GEPA 是否能基于 Training Sample 生成 Candidate Template Bundle。
- Verification Run 是否能判断 Candidate Template Bundle 是否更好。
- Online Replay 是否能连续运行多轮。

Online Simulation 不验证：

- HIGO 和 EchoMem 的真实调用稳定性。
- MemRouter 和 MultiMemBackend 的真实对接效果。
- 真实 retrieval 质量。
- 真实 answer 质量。
- 真实用户反馈。

因此实验结论应写成：

```text
online optimization 流程在 simulation 中可行。
```

不能写成：

```text
真实 online optimization 已经有效。
```

### 8.3 实验总流程

Online Simulation 的总流程如下：

```text
1. 使用 offline 数据构造模拟线上请求。
2. 使用 Active Template Bundle 跑 route。
3. 生成模拟 Trace。
4. 从模拟 Trace 生成 Training Sample。
5. 使用 GEPA 生成 Candidate Template Bundle。
6. 使用 Verification Run 对比 Active 和 Candidate。
7. 生成 Verification Report。
8. 在 Online Replay 中重复以上过程。
```

### 8.4 实验 1：Trace Simulation

Trace Simulation 指用 offline 数据生成模拟 Trace。

简单说：

```text
先不接真实线上系统，而是用 LoCoMo query 跑 MemRouter，构造“像线上一样”的 Trace。
```

目标：

```text
验证 Trace 结构是否足够支撑后续 Training Sample 生成。
```

做法：

1. 选择一批 LoCoMo query。
2. 使用 Active Template Bundle 跑 MemRouter route。
3. 记录 route 信息。
4. retrieval 和 answer 暂时使用模拟字段。
5. 输出模拟 Trace Store。

模拟 Trace 至少包含：

- `trace_id`
- `template_bundle_id`
- `query.raw_user_query`
- `query.normalized_user_query`
- `route.primary_backend_id`
- `route.matched_template_id`
- `route.confidence`
- `route.top_templates`
- `retrieval.execution_path`
- `retrieval.result_count`
- `answer.status`

输出：

```text
simulated_traces.jsonl
```

判断标准：

1. 每条 Trace 都能追溯到原始 query。
2. 每条 Trace 都能追溯到 `template_bundle_id`。
3. Trace 能支持下一步生成 Training Sample。
4. 缺失字段能被明确列出。

### 8.5 实验 2：Training Sample Builder

Training Sample Builder 指从 Trace 生成 Training Sample 的程序或流程。

简单说：

```text
把模拟 Trace 变成 GEPA 可以使用的 <question, expected_backend> 样本。
```

目标：

```text
验证 Training Sample 生成规则是否可执行，Rejected Trace 是否可解释。
```

做法：

1. 输入 `simulated_traces.jsonl`。
2. 根据 LoCoMo 的真实 label 模拟 `expected_backend`。
3. 根据规则生成 `label_confidence`。
4. 根据规则生成 `label_source`。
5. 生成 Training Sample Set。
6. 生成 Rejected Trace Set。

建议模拟以下 `label_source`：

| `label_source` | 模拟方式 |
| --- | --- |
| `human_review` | 直接使用 LoCoMo label，`label_confidence = 1.00`。 |
| `answer_judge` | 根据是否 route 正确模拟 judge 结果，`label_confidence = 0.75`。 |
| `production_success` | Active route 正确的样本，`label_confidence = 0.70`。 |
| `counterfactual_retrieval` | Active route 错误但 expected backend 在 top2 的样本，`label_confidence = 0.80`。 |

输出：

```text
online_train_samples.jsonl
online_rejected_traces.jsonl
```

判断标准：

1. Training Sample 包含 `question`、`expected_backend`、`label_confidence`、`label_source`。
2. `label_confidence < 0.70` 的样本不进入 Training Sample Set。
3. Rejected Trace 有明确 `reject_reason`。
4. 不同 `label_source` 的数量分布可统计。

### 8.6 实验 3：Noisy Label

Noisy Label 指错误或不可靠的 `expected_backend`。

简单说：

```text
故意往 Training Sample 里混入一些错误 label，测试流程能不能扛住脏数据。
```

目标：

```text
验证 online optimization 对错误 label 的敏感程度。
```

做法：

1. 从 Training Sample Set 复制多个版本。
2. 分别注入 5%、10%、20% 的 Noisy Label。
3. 每个版本都运行 GEPA。
4. 对每个 Candidate Template Bundle 做 Verification Run。
5. 比较 Verification Report。

Noisy Label 注入方式：

```text
随机把一部分 expected_backend 改成错误 backend。
```

实验组：

| 实验组 | Noisy Label 比例 |
| --- | ---: |
| clean | 0% |
| noisy_5 | 5% |
| noisy_10 | 10% |
| noisy_20 | 20% |

观察指标：

- `weighted_backend_accuracy`
- `macro_backend_recall`
- `verification_regression_rate`
- `template_accept_rate`
- `fast_path_rate`

判断标准：

1. 5% Noisy Label 下，Candidate Template Bundle 不应明显退化。
2. 10% Noisy Label 下，Verification Report 应能发现风险。
3. 20% Noisy Label 下，Candidate Template Bundle 大概率应被 reject 或 needs_review。

### 8.7 实验 4：Regression Anchor

Regression Anchor 指已知正确、用于防止破坏旧能力的 Training Sample。

简单说：

```text
加入一批原本做对的样本，防止 GEPA 只修失败样本却破坏旧能力。
```

目标：

```text
验证 Regression Anchor 是否能减少 Verification Regression。
```

做法：

1. 构造两个 GEPA Train Set。
2. 第一组只包含失败或边界样本。
3. 第二组加入 production_success 样本作为 Regression Anchor。
4. 分别运行 GEPA。
5. 分别做 Verification Run。

实验组：

| 实验组 | 说明 |
| --- | --- |
| without_anchor | 不加入 production_success 样本。 |
| with_anchor | 加入 production_success 样本。 |

观察指标：

- `verification_regression_count`
- `verification_regression_rate`
- `improvement_count`
- `weighted_backend_accuracy`
- `macro_backend_recall`

判断标准：

1. `with_anchor` 的 Verification Regression 应低于 `without_anchor`。
2. `with_anchor` 不应明显降低 improvement_count。
3. 如果 `with_anchor` 明显减少 regression，则 online GEPA 必须保留 Regression Anchor。

### 8.8 实验 5：Verification Run

Verification Run 指在同一批样本上对比 Active Template Bundle 和 Candidate Template Bundle。

简单说：

```text
让旧模板和新模板做同一张试卷，看新模板是不是真的更好。
```

目标：

```text
验证 Verification Report 是否能稳定给出 pass、reject 或 needs_review。
```

做法：

1. 准备 Active Template Bundle。
2. 准备多个 Candidate Template Bundle。
3. 准备 Verification Set。
4. 对每个 Candidate Template Bundle 运行 Verification Run。
5. 输出 Verification Report。

Candidate Template Bundle 应包含三类：

| 类型 | 说明 |
| --- | --- |
| good_candidate | 预期会提升的 Candidate Template Bundle。 |
| bad_candidate | 故意降低 threshold 或制造过宽 prototype 的 Candidate Template Bundle。 |
| neutral_candidate | 改动很小、预期变化不明显的 Candidate Template Bundle。 |

期望结论：

| Candidate 类型 | Verification Report 结论 |
| --- | --- |
| good_candidate | pass |
| bad_candidate | reject |
| neutral_candidate | needs_review |

判断标准：

1. Verification Report 能发现 bad_candidate。
2. Verification Report 不会因为单一指标提升就放过高 regression 的 Candidate。
3. Verification Report 能识别样本量不足时的 needs_review。

### 8.9 实验 6：Online Replay

Online Replay 指按时间顺序重放历史样本，模拟线上请求持续到来。

简单说：

```text
把 offline 数据切成多批，假装它们是一段时间内陆续来的线上流量。
```

目标：

```text
验证 online optimization 流程是否能连续运行多轮。
```

做法：

1. 按 `sample_id` 或时间顺序把 LoCoMo 样本切成多个 Simulation Batch。
2. 第 1 个 Simulation Batch 生成 Trace。
3. Trace 生成 Training Sample。
4. GEPA 生成 Candidate Template Bundle。
5. Verification Run 判断 pass、reject 或 needs_review。
6. 如果 pass，则把 Candidate Template Bundle 设为下一轮 Active Template Bundle。
7. 对后续 Simulation Batch 重复以上流程。

Simulation Batch 指一批模拟线上样本。

简单说：

```text
它代表“一段时间内来的请求集合”。
```

输出：

```text
online_replay_report.md
online_replay_rounds.jsonl
```

每轮需要记录：

- 当前 Active Template Bundle。
- 当前 Simulation Batch。
- 生成的 Training Sample 数量。
- Rejected Trace 数量。
- Candidate Template Bundle。
- Verification Report 结论。
- 是否更新 Active Template Bundle。

观察指标：

- 每轮 `weighted_backend_accuracy`。
- 每轮 `macro_backend_recall`。
- 每轮 `verification_regression_rate`。
- 每轮 pass / reject / needs_review 数量。
- Active Template Bundle 是否持续改善。

判断标准：

1. 多轮运行不能持续退化。
2. Verification Report 能阻止明显变差的 Candidate Template Bundle。
3. Active Template Bundle 不应频繁震荡。
4. 如果连续多轮没有提升，应停止 optimization，而不是继续强行生成新版本。

### 8.10 推荐实验顺序

推荐按以下顺序执行：

1. Trace Simulation。
2. Training Sample Builder。
3. Verification Run。
4. Noisy Label。
5. Regression Anchor。
6. Online Replay。

原因：

```text
先验证数据是否能流动，再验证判断是否可靠，最后验证多轮流程是否稳定。
```

### 8.11 最终实验结论格式

Online Simulation 的最终结论应回答以下问题：

1. Trace 结构是否足够支撑 Training Sample 生成？
2. Training Sample Builder 是否能产生可信样本和可解释的 Rejected Trace？
3. GEPA 面对 Noisy Label 是否仍然稳定？
4. Regression Anchor 是否能减少 Verification Regression？
5. Verification Report 是否能正确区分 pass、reject、needs_review？
6. Online Replay 是否能连续运行多轮且不持续退化？

最终结论建议使用以下格式：

```text
Online Simulation 证明 online optimization 控制流程可运行。
当前实验不证明真实线上效果。
真实线上效果需要等 HIGO、EchoMem、MemRouter、MultiMemBackend 和真实 Trace 全链路完成后再验证。
```
