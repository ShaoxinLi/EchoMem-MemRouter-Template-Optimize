# MemRouter Template YAML GEPA Matcher-Only 实验方案

## 1. 实验目标

MemRouter 的 routing 任务是把用户问题路由到合适的 memory backend。本实验固定 matcher 代码和 matcher 配置，只优化 matcher 使用的 YAML templates。

本实验要验证的问题是：

```text
在固定 matcher 代码和 matcher 配置的前提下，GEPA 优化 Template Bundle 是否能提升 matcher-only routing 效果？
```

每条样本包含：

```text
<question, expected_backend>
```

实验只评估 routing，不评估最终答案质量。当 matcher 没有接受任何 backend 时，结果记为 `No Decision`，本实验不会调用 LLM fallback。

最终结果以 `Selected` 相比 `Initial` 在 `val set` 和 `test set` 上的指标变化为准。

## 2. 术语

| 术语 | 定义 |
| --- | --- |
| `GEPA` | 用 score 和 feedback 驱动文本资产迭代优化的优化器。本实验中，GEPA 负责选择 Parent、生成 Proposal、评估 Candidate 并选择 Selected。 |
| `Backend` | MemRouter 的路由目标，例如 `openviking_memory_backend`、`graph_memory_backend`、`temporal_memory_backend`。 |
| `Matcher` | 固定代码模块，读取 Template Bundle 后对 question 进行 backend routing。 |
| `Template Bundle` | matcher 消费的一组 YAML template 文件。 |
| `Initial` | 实验开始前的原始 Template Bundle。 |
| `Parent` | 当前 GEPA 轮次被修改的完整 Template Bundle。第一轮 Parent 等于 Initial。 |
| `Proposal LLM` | 根据 Parent 和 Feedback 生成 Complete Editable Fields Bundle 的 LLM。 |
| `Complete Editable Fields Bundle` | Proposal LLM 的原始输出。它为 Parent 中每个 template 文件输出完整的 `query_prototypes`、`hard_negatives`、`thresholds`，不输出不可编辑字段。 |
| `Proposal` | 运行时代码把 Complete Editable Fields Bundle 合并回 Parent 后得到的完整 Template Bundle。 |
| `Candidate` | 通过 Minibatch 比较并进入候选池的 Proposal。 |
| `Selected` | 预算结束后，在 val set 上 GEPA Score 最高的 Candidate。 |
| `Minibatch` | 每轮 GEPA 用于生成 feedback 和比较 Parent / Proposal 的一批训练样本。 |
| `train set` | 用于生成 feedback 和采样 Minibatch 的训练集。 |
| `val set` | 用于评估 Candidate 并选择 Selected 的验证集。 |
| `test set` | Selected 固定后才使用的最终评估集。 |
| `expected_backend` | 样本的人工标注 backend。 |
| `predicted_backend` | matcher 输出的 primary backend；如果没有接受任何 backend，则为 `No Decision`。 |
| `predicted_backends` | matcher 接受的 backend 列表；multi-backend accept 时包含多个 backend。 |
| `Matcher Result` | matcher 对样本的原始 routing 输出，包括 ranking、scores、matched template、route method 和是否正确。 |
| `Deterministic Feedback` | 固定代码基于 Matcher Result 和 expected backend 生成的结构化诊断。 |
| `LLM Feedback` | LLM 基于 Deterministic Feedback 生成的泛化修改建议。 |
| `Validator` | 固定程序，检查 Proposal 是否满足输出契约、schema、不可变字段和 matcher 可运行性。 |
| `Validator Feedback` | Validator 失败时生成的结构化错误反馈。 |
| `Generation Prompt` | Proposal LLM 的完整指令，包含任务目标、Parent、Feedback 和 Output Contract。 |
| `Output Contract` | Complete Editable Fields Bundle 必须满足的输出格式和字段约束。 |
| `GEPA Score` | evaluator 计算的优化目标，用于比较 Parent、Proposal 和 Candidate。 |
| `No Decision` | matcher 没有接受任何 backend。 |
| `LLM fallback` | matcher 无法决策时由 LLM 兜底判断 backend 的路径。本实验禁用该路径。 |

## 3. 实验边界

本实验固定以下内容：

- matcher 代码。
- matcher 配置。
- backend 集合。
- template 文件集合。
- `template_id`、`target`、`query_spec` 等不可编辑字段。

本实验只允许优化以下字段：

- `query_prototypes`
- `hard_negatives`
- `thresholds`

本实验不做：

- 不调用 LLM fallback。
- 不使用 answer judge。
- 不让 LLM Feedback 参与打分。
- 不让 test set 参与 GEPA 生成、选择或 early stop。
- 不让 Proposal LLM 输出完整 Template Bundle。
- 不使用 JSON 编辑指令或增量变更格式。

## 4. 总体流程

```text
1. 读取 train / val / test 数据。
2. 读取 Initial Template Bundle。
3. 用 Initial 评估 train / val / test。
4. 用 Initial train results 构建 bucket_random 的 initial sampling buckets。
5. GEPA 迭代：
   5.1 选择 Parent。
   5.2 sampler 采样 Minibatch。
   5.3 使用 Parent 跑 Minibatch，得到 Matcher Result。
   5.4 生成 Deterministic Feedback。
   5.5 LLM Feedback 只基于 Deterministic Feedback 生成泛化建议。
   5.6 Proposal LLM 基于 Parent 和 Feedback 输出 Complete Editable Fields Bundle。
   5.7 运行时代码把 Complete Editable Fields Bundle 合并回 Parent，得到 Proposal。
   5.8 Validator 检查 Proposal。
       - 如果失败：Proposal 得分为 0，不进入 Candidate pool。
       - 如果通过：继续评估。
   5.9 使用 Proposal 跑同一个 Minibatch。
   5.10 如果 Proposal Minibatch GEPA Score 高于 Parent，则评估 val set，并把 Proposal 加入 Candidate pool。
   5.11 bucket_random 模式下，达到刷新条件时用当前 best Candidate 重新评估完整 train set 并重建 buckets。
6. 预算或 stopper 触发后，选择 val set GEPA Score 最高的 Candidate 作为 Selected。
7. 使用 Selected 评估 val / test。
8. 输出 Selected vs Initial 报告。
```

关键约束：

- Parent 和 Proposal 必须在同一个 Minibatch 上比较。
- test set 只在 Selected 固定后使用。
- Selected 只由 val set 选择。
- LLM Feedback 不直接访问 Matcher Result；它只接收 Deterministic Feedback。

## 5. Matcher Route Decision

Matcher 先生成 template ranking 和 backend ranking，再做 route decision。

### 5.1 单 backend accept

如果第一名 backend 满足：

```text
best.score >= best_template.thresholds.accept
best.score - second.score >= best_template.thresholds.margin
```

则接受第一名 backend：

```text
route_method = template_embedding
predicted_backends = [best.backend_id]
predicted_backend = best.backend_id
```

### 5.2 Multi-backend accept

如果第一名和第二名属于不同 backend，且满足：

```text
best.score >= best_template.thresholds.accept
second.score >= second_template.thresholds.accept
best.score - second.score < best_template.thresholds.margin
second.score / best.score >= 0.85
```

则接受两个 backend：

```text
route_method = template_embedding_multi_backend
predicted_backends = [best.backend_id, second.backend_id]
predicted_backend = best.backend_id
```

由于样本标签只有一个 `expected_backend`，`is_correct` 仍按 `predicted_backend == expected_backend` 计算。

### 5.3 No Decision

如果不满足单 backend accept 或 multi-backend accept，则输出：

```text
route_method = no_decision
predicted_backend = No Decision
predicted_backends = []
```

`thresholds.fallback` 当前不驱动 LLM fallback。本实验中的 `No Decision` 直接进入指标统计。

## 6. GEPA 输入

每轮 Proposal LLM 的输入由 `Generation Prompt` 组织，包含：

| 输入 | 说明 |
| --- | --- |
| `Generation Prompt` | 任务目标、编辑原则和输出格式。 |
| `Output Contract` | Complete Editable Fields Bundle 的硬性输出规则，嵌入 Generation Prompt。 |
| `Parent Template Bundle` | 当前轮完整 Parent，放在 `CURRENT TEMPLATE BUNDLE START/END` 区块中。 |
| `Feedback` | JSON 字符串，包含 Validator Feedback、Deterministic Feedback、LLM Feedback、Parent Val Summary、Initial Summary。 |

`Feedback` 的结构是：

```json
{
  "validator_feedback": null,
  "deterministic_feedback": {},
  "llm_feedback": {},
  "parent_val_summary": {},
  "initial_summary": {
    "train": {},
    "val": {}
  }
}
```

`Minibatch Cases` 不作为独立区块输入 Proposal LLM；它们以样本级信息进入 `deterministic_feedback.case_feedback`。

`Parent Val Summary` 和 `Initial Summary` 是压缩指标摘要，不包含完整样本列表。

## 7. Proposal 输出与合并

Proposal LLM 必须输出 `Complete Editable Fields Bundle`。

`Complete Editable Fields Bundle` 的定义是：

```text
为 Parent 中每个 template 文件输出完整修改后的可编辑字段，
且只输出 query_prototypes、hard_negatives、thresholds。
```

每个文件 section 必须满足：

```yaml
# FILE: <template-file-name>.yaml
query_prototypes:
  - "..."
hard_negatives:
  - query: "..."
    confusing_with_backend: "..."
    reason: "..."
thresholds:
  accept: 0.0
  fallback: 0.0
  margin: 0.0
  hard_negative_margin: 0.0
  hard_negative_penalty: 0.0
```

输出要求：

- 必须包含 Parent 中每个 template 文件。
- 每个文件必须包含完整的 `query_prototypes`、`hard_negatives`、`thresholds`。
- 不允许只输出新增项、删除项或部分字段值。
- 不允许省略未修改文件。
- 不允许省略未修改字段。
- 不允许新增或删除 template 文件。
- 不允许输出 Markdown fence 或解释文本。
- 不允许复制当前 Minibatch 的 exact question 到 `query_prototypes`。
- 不允许复制当前 Minibatch 的 exact question 到 `hard_negatives.query`。
- 每条 `hard_negatives` 必须包含 `query`、`confusing_with_backend`、`reason`。

运行时代码会把 Complete Editable Fields Bundle 合并回 Parent：

- `query_prototypes`、`hard_negatives`、`thresholds` 来自 Proposal LLM 输出。
- `schema_version`、`template_id`、`version`、`status`、`target`、`intent_family`、`semantic_card`、`query_spec`、`calibration` 等不可编辑字段来自 Parent。
- 如果 Proposal LLM 误输出不可编辑字段，合并时不会采纳。
- 如果输出不满足 Complete Editable Fields Bundle 契约，合并结果会保留原始输出，并由 Validator 生成 `proposal_output_contract_error`。

## 8. Validator

Validator 是 Proposal 进入 matcher 前的硬门槛。

Validator 检查：

- Template Bundle 可解析。
- 文件集合必须与 Initial 一致。
- 每个 template 可通过 schema 校验。
- `template_id`、`target`、`query_spec` 必须与 Initial 一致。
- 除 `query_prototypes`、`hard_negatives`、`thresholds` 以外的字段必须与 Initial 一致。
- `query_prototypes` 必须是字符串列表。
- `hard_negatives` 必须是对象列表。
- 每条 `hard_negatives` 必须包含 `query`、`confusing_with_backend`、`reason`。
- `query_prototypes` 单条长度不超过 240 字符。
- `hard_negatives.query` 单条长度不超过 240 字符。
- 相对 Parent 新增的 `query_prototypes` 不得 exact copy 当前 Minibatch question。
- 相对 Parent 新增的 `hard_negatives.query` 不得 exact copy 当前 Minibatch question。
- `thresholds` 中每个数值必须在 `[0, 1]`。
- Template Bundle 可被 template loader 加载。
- matcher 可基于 Proposal 初始化并跑通 smoke cases。

当前 Validator 不限制 `query_prototypes` 或 `hard_negatives` 的新增数量和总数量。

Validator 失败时：

```text
Proposal score = 0
Proposal 不进入 Candidate pool
Validator Feedback 进入下一轮 Feedback
```

Validator Feedback 示例：

```json
{
  "validator_status": "failed",
  "errors": [
    {
      "type": "missing_required_field",
      "message": "Missing required field 'confusing_with_backend' at 'hard_negatives.5.confusing_with_backend'.",
      "template_id": "temporal.sequence_reasoning.v1",
      "field": "hard_negatives",
      "filename": "temporal.sequence_reasoning.v1.yaml",
      "path": "hard_negatives.5.confusing_with_backend",
      "missing_field": "confusing_with_backend",
      "item_index": 5
    }
  ],
  "required_fix": [
    "Return a Complete Editable Fields Bundle only, with one '# FILE: <name>.yaml' marker for every Parent template file.",
    "Every file section must include complete query_prototypes, hard_negatives, and thresholds fields.",
    "Do not output template_id, target, query_spec, or any non-whitelisted fields."
  ]
}
```

## 9. Feedback

Feedback 流程是：

```text
Matcher Result -> Deterministic Feedback -> LLM Feedback -> GEPA
```

### 9.1 Matcher Result

Matcher Result 是样本级原始 routing 输出。核心字段包括：

- `case_id`
- `question`
- `expected_backend`
- `predicted_backend`
- `predicted_backends`
- `is_correct`
- `is_no_decision`
- `route_method`
- `expected_backend_rank`
- `expected_backend_score`
- `expected_backend_best_template_id`
- `winning_backend`
- `winning_backend_score`
- `margin_to_win`
- `matched_template_id`
- `matched_template_accept`
- `matched_template_margin`
- `top_backend_ranking`
- `top_template_ranking`
- `score`

### 9.2 Deterministic Feedback

Deterministic Feedback 的职责是说明“发生了什么”和“规则建议做什么”。

结构：

```json
{
  "feedback_type": "deterministic_feedback",
  "action_type_definitions": {},
  "batch_summary": {},
  "case_feedback": [],
  "batch_action_recommendations": []
}
```

`case_feedback` 保留每条样本的主要 routing 事实、diagnosis 和 action recommendation。核心字段包括：

- `case_id`
- `question`
- `expected_backend`
- `predicted_backend`
- `is_correct`
- `is_no_decision`
- `expected_backend_rank`
- `expected_backend_score`
- `winning_backend`
- `winning_backend_score`
- `margin_to_win`
- `matched_template_id`
- `matched_template_accept`
- `matched_template_margin`
- `GEPA_case_score`
- `top_backend_ranking`
- `top_template_ranking`
- `diagnosis`
- `suggested_action_type`
- `action_recommendation`

`top_backend_ranking` 和 `top_template_ranking` 在 Deterministic Feedback 中只保留前 3 名。

`batch_summary` 由 `summarize_outputs()` 生成，核心字段包括：

- `gepa_score`
- `mean_case_score`
- `backend_accuracy`
- `macro_backend_recall`
- `expected_backend_top2_rate`
- `expected_backend_margin_score`
- `no_decision_rate`
- `template_accept_rate`
- `regression_count`
- `regression_rate`
- `regression_penalty`
- `overbroad_penalty`
- `per_backend_recall`
- `top_confusion_pairs`
- `suspected_overbroad_templates`

### 9.3 Deterministic Action

固定代码生成以下 action type：

| action type | 触发条件 | 含义 |
| --- | --- | --- |
| `preserve_behavior` | 当前样本正确 | 保持当前正确行为。 |
| `lower_expected_accept_or_margin` | `No Decision` 且 expected backend 排第 1 | 小幅降低 expected template 的 accept 或 margin。 |
| `add_discriminative_prototypes_or_hard_negatives` | expected backend 排前 2 且接近 winner | 增加可区分的正例 prototype 或 hard negative。 |
| `add_expected_backend_prototypes` | `No Decision`，或 expected backend 排名弱 | 增加 expected backend 的泛化 prototypes。 |
| `add_wrong_template_hard_negatives` | 错误 backend/template 接受了样本 | 给误吸 template 增加 hard negatives。 |

`batch_action_recommendations` 会把样本级 action 按 `(action_type, target_template_id, field)` 聚合，并记录 evidence case ids。

### 9.4 LLM Feedback

LLM Feedback 的输入只有 Deterministic Feedback：

```json
{
  "deterministic_feedback": {}
}
```

LLM Feedback 的职责是把规则诊断转成少量泛化编辑建议。它不输出 YAML，不参与打分，不决定 Proposal 是否接受。

LLM Feedback prompt 要求：

- 不复制 exact question、姓名、日期或事件细节。
- 不逐条复述样本。
- 以 `batch_action_recommendations` 为主要结构化信号，但不盲从。
- 结合 `case_feedback` 和 `batch_summary` 做一致性检查。
- 优先给出高影响、可泛化的建议。
- 最多输出 5 条 `template_suggestions`，优先 1 到 3 条。
- 字段名只能使用 `query_prototypes`、`hard_negatives`、`thresholds`。
- action 只能使用 `add`、`rewrite`、`remove`、`increase`、`decrease`、`preserve`。

LLM Feedback 输出示例：

```json
{
  "feedback_type": "llm_feedback",
  "summary": "Temporal queries are being absorbed by OpenViking templates, while correct personal fact behavior should be preserved.",
  "template_suggestions": [
    {
      "template_id": "temporal.timeline_fact.v1",
      "field": "query_prototypes",
      "action": "add",
      "priority": "high",
      "guidance": "Add generalized prototypes for questions asking when a remembered personal event happened.",
      "evidence": "Repeated temporal cases rank close to the winner but are not selected."
    },
    {
      "template_id": "openviking.personal_fact_lookup.en.v2",
      "field": "hard_negatives",
      "action": "add",
      "priority": "medium",
      "guidance": "Add hard negatives for timestamp-centered questions that should route to temporal templates.",
      "evidence": "OpenViking absorbs temporal questions."
    }
  ],
  "preserve": [
    "Preserve correct personal fact lookup behavior for non-temporal questions."
  ]
}
```

## 10. Generation Prompt 和 Output Contract

Generation Prompt 的目标是让 Proposal LLM 基于 Parent 和 Feedback 输出 Complete Editable Fields Bundle。

Prompt 核心规则：

- 优化 matcher-only routing。
- 没有 LLM fallback。
- 允许 multi-backend accept。
- Validator Feedback 优先级最高。
- LLM Feedback 是主要编辑参考，但不能与 Validator Feedback、Deterministic Feedback 或 Output Contract 冲突。
- Deterministic Feedback 是证据和一致性检查。
- 正确样本是 regression anchors。
- 修改应修复泛化模式，不应过拟合当前 Minibatch。
- 输出 Complete Editable Fields Bundle only。

Output Contract 已嵌入 Generation Prompt，核心要求与第 7 节一致。

## 11. GEPA Score

### 11.1 样本级分数

每条样本的 `GEPA_case_score` 来自 `score_case()`：

```text
GEPA_case_score =
  0.45 * backend_correct
+ 0.20 * expected_top2
+ 0.15 * margin_score
+ 0.10 * accepted
+ 0.10 * no_decision_control
```

其中：

```text
backend_correct = 1 if predicted_backend == expected_backend else 0
expected_top2 = 1 if expected_backend_rank in [1, 2] else 0
accepted = 0 if No Decision else 1
no_decision_control = accepted
margin_score = clamp((margin_to_win + 0.15) / 0.30, 0, 1)
```

### 11.2 批次级 GEPA Score

Candidate 比较使用批次级 `GEPA Score`：

```text
GEPA Score =
  0.40 * backend_accuracy
+ 0.25 * macro_backend_recall
+ 0.15 * expected_backend_top2_rate
+ 0.10 * expected_backend_margin_score
+ 0.05 * template_accept_rate
+ 0.05 * no_decision_control_score
- regression_penalty
- overbroad_penalty
```

其中：

```text
template_accept_rate = 1 - no_decision_rate
no_decision_control_score = 1 - no_decision_rate
expected_backend_margin_score = mean(clamp((margin_to_win + 0.15) / 0.30, 0, 1))
```

`regression_penalty` 只在同一 Minibatch 上比较 Proposal 和 Parent 时启用：

```text
regression_count = Parent 正确但 Proposal 改错的样本数
regression_rate = regression_count / batch_size
regression_penalty = 0.20 * regression_rate
```

`overbroad_penalty` 惩罚单个 template 在 batch 中集中误吸：

```text
max_wrong_template_fraction = max_wrong_absorptions_by_template / batch_size
overbroad_penalty = max(0, max_wrong_template_fraction - 0.35) * 0.20
```

## 12. Sampling

当前实现支持两种 sampler：

- `bucket_random`
- `epoch`

### 12.1 bucket_random

`bucket_random` 先用 Initial 完整评估 train set，并按结果构建 buckets。

bucket 定义：

| bucket | 定义 |
| --- | --- |
| `graph_wrong` | expected 是 graph，且当前 routing 错误。 |
| `temporal_wrong` | expected 是 temporal，且当前 routing 错误。 |
| `openviking_wrong` | expected 是 openviking，且当前 routing 错误。 |
| `no_decision_cases` | 当前 routing 是 No Decision。 |
| `graph_correct` | expected 是 graph，且当前 routing 正确。 |
| `temporal_correct` | expected 是 temporal，且当前 routing 正确。 |
| `openviking_correct` | expected 是 openviking，且当前 routing 正确。 |
| `random_pool` | 完整 train set。 |

每个 bucket 存的是 train data id。

固定配额：

| bucket | quota |
| --- | ---: |
| `graph_wrong` | 2 |
| `temporal_wrong` | 2 |
| `openviking_wrong` | 2 |
| `no_decision_cases` | 2 |
| `graph_correct` | 2 |
| `temporal_correct` | 2 |
| `openviking_correct` | 2 |
| `random_pool` | 2 |

当 `batch_size = 16` 时，以上配额刚好填满一个 Minibatch。

当 `batch_size > 16` 时，剩余位置从 `random_pool` 补满。例如 `batch_size = 32` 时，前 16 条来自固定 bucket 配额，后 16 条来自 `random_pool`。

如果某个 bucket 为空，会回退到 `random_pool`。

同一个 Minibatch 内默认不重复采样。bucket 内先打乱，再按 `(sample_freq, data_id)` 排序，因此优先选择累计采样次数较少的样本；采样次数相同则较小 `data_id` 优先。

### 12.2 bucket refresh

`bucket_random` 支持动态刷新：

```text
--bucket-refresh-after-accepted-candidates N
```

含义：自上次刷新后，每接受 N 个 Candidate，就用当前 best Candidate 重新评估完整 train set，并基于新的 Matcher Result 重建 buckets。

默认值：

```text
N = 3
```

设置为 `0` 表示禁用刷新。

刷新产物写入：

```text
reports/sampling_bucket_refreshes/
```

### 12.3 epoch

`epoch` sampler 按 epoch 遍历 train set。

每个 epoch：

- shuffle 一次完整 train data ids。
- 按 `batch_size` 切分。
- 最后一个 batch 不足时，用累计采样次数较少的样本补齐。

控制参数：

```text
--num-train-epochs K
```

epoch iteration 数：

```text
ceil(train_size / batch_size) * K
```

如果 `sampler = epoch` 且未显式设置 `--max-metric-calls`，runner 不设置 metric-call 上限，而是用 epoch iteration stopper 控制结束。

## 13. Candidate 选择

### 13.1 Proposal 接受

每轮在同一个 Minibatch 上比较：

```text
Proposal GEPA Score > Parent GEPA Score
```

如果成立：

- Proposal 跑完整 val set。
- Proposal 加入 Candidate pool。

如果不成立：

- Proposal 丢弃。

### 13.2 Parent 选择

当前 GEPA 调用使用：

```text
candidate_selection_strategy = current_best
```

因此 Parent 来自当前 val set GEPA Score 最好的 Candidate。

### 13.3 Selected 选择

预算结束后：

```text
Selected = val set GEPA Score 最高的 Candidate
```

最终报告比较：

```text
Initial vs Selected on val
Initial vs Selected on test
```

## 14. 停止条件与缓存

停止条件：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--max-metric-calls` | dry-run 为 96，full 为 5000，epoch 未设置时为无限制 | GEPA metric call 上限。 |
| `--early-stop-rounds` | 20 | 连续若干轮未提升 best val score 后停止。 |
| `--max-invalid-proposals` | 5 | 连续 Validator 失败达到上限后停止。 |
| `--target-accepted-candidates` | 未设置 | 达到指定 accepted Candidate 数后停止；该数量不包含 Initial。 |
| `--num-train-epochs` | 1 | epoch sampler 的训练集遍历次数。 |

缓存：

| 缓存 | 作用 |
| --- | --- |
| `question_embedding_cache` | 缓存 question embedding。 |
| `template_embedding_cache` | 按 Template Bundle hash 缓存 template embedding。 |
| `candidate_eval_cache` | 缓存 Candidate 在同一 case id 集合上的评估结果。 |

GEPA 自带 per-example cache 在本实验中关闭：

```text
cache_evaluation = False
```

原因是本实验的 score 是 batch-level aggregate，并重复写入每个样本；train / val DataLoader ids 也可能重叠。当前实现使用 adapter-level candidate eval cache。

## 15. 数据

当前 runner 使用的输入路径：

```text
benchmarks/locomo/data/gepa_template_yaml/
  locomo_v2_template.train.jsonl
  locomo_v2_template.val.jsonl
  locomo_v2_template.test.jsonl
```

每行 JSONL 至少包含：

```json
{
  "case_id": "conv-42_Q1",
  "sample_id": "conv-42",
  "scenario": "...",
  "category": "...",
  "question": "...",
  "expected_backend": "openviking_memory_backend"
}
```

`train set` 用于生成 feedback 和采样 Minibatch。
`val set` 用于选择 Candidate 和 Selected。
`test set` 只用于最终评估。

## 16. Runner

入口脚本：

```text
scripts/optimize_memrouter_template_yaml_gepa.py
```

`--matcher-config` 当前是兼容参数：runner 会解析并写入 config snapshot，但当前实现不读取该文件来改变 matcher 行为。

### 16.1 Dry-run

```bash
uv run python scripts/optimize_memrouter_template_yaml_gepa.py \
  --profile dry-run \
  --config configs/memrouter_gepa.local.yaml \
  --train benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.train.jsonl \
  --val benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.val.jsonl \
  --test benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.test.jsonl \
  --template-dir echomem/templates_data \
  --matcher-config configs/memrouter_matcher.yaml \
  --runs-dir runs/memrouter_template_yaml_gepa \
  --batch-size 16 \
  --sampler bucket_random \
  --bucket-refresh-after-accepted-candidates 3 \
  --num-threads 4 \
  --early-stop-rounds 20 \
  --max-invalid-proposals 5 \
  --enable-template-embedding-cache \
  --enable-question-embedding-cache \
  --enable-candidate-eval-cache \
  --parallel-val-eval \
  --seed 13 \
  --dry-run-train-limit 48 \
  --dry-run-val-limit 24 \
  --dry-run-test-limit 24
```

### 16.2 Full run with bucket_random

```bash
uv run python scripts/optimize_memrouter_template_yaml_gepa.py \
  --profile full \
  --config configs/memrouter_gepa.local.yaml \
  --train benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.train.jsonl \
  --val benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.val.jsonl \
  --test benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.test.jsonl \
  --template-dir echomem/templates_data \
  --matcher-config configs/memrouter_matcher.yaml \
  --runs-dir runs/memrouter_template_yaml_gepa \
  --batch-size 32 \
  --sampler bucket_random \
  --bucket-refresh-after-accepted-candidates 3 \
  --num-threads 4 \
  --early-stop-rounds 20 \
  --max-invalid-proposals 5 \
  --enable-template-embedding-cache \
  --enable-question-embedding-cache \
  --enable-candidate-eval-cache \
  --parallel-val-eval \
  --seed 13 \
  --max-metric-calls 50000
```

### 16.3 Full run with epoch

```bash
uv run python scripts/optimize_memrouter_template_yaml_gepa.py \
  --profile full \
  --config configs/memrouter_gepa.local.yaml \
  --train benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.train.jsonl \
  --val benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.val.jsonl \
  --test benchmarks/locomo/data/gepa_template_yaml/locomo_v2_template.test.jsonl \
  --template-dir echomem/templates_data \
  --matcher-config configs/memrouter_matcher.yaml \
  --runs-dir runs/memrouter_template_yaml_gepa \
  --batch-size 32 \
  --sampler epoch \
  --num-train-epochs 5 \
  --num-threads 4 \
  --early-stop-rounds 20 \
  --max-invalid-proposals 5 \
  --enable-template-embedding-cache \
  --enable-question-embedding-cache \
  --enable-candidate-eval-cache \
  --parallel-val-eval \
  --seed 13
```

epoch 模式下如果不传 `--max-metric-calls`，主循环由 `--num-train-epochs` 控制。

### 16.4 Resume

```bash
uv run python scripts/optimize_memrouter_template_yaml_gepa.py \
  --resume-run runs/memrouter_template_yaml_gepa/<run_id>
```

`--resume-run` 会读取 run 目录中的 `config_snapshot.yaml`。当前实现不允许同时传入其他 CLI override。

## 17. Run 产物

典型 run 目录：

```text
runs/memrouter_template_yaml_gepa/<run_id>/
  config_snapshot.yaml
  split_summary.json
  sampling_config.json
  sampling_buckets.json
  gepa_result.json
  run_state.json
  cache_manifest.json
  cache/
    question_embeddings/
    template_embeddings/
    candidate_eval/
  tmp/
    validated_bundles/
  initial/
    train_eval/
      summary.json
      results.json
      results.jsonl
    val_eval/
    test_eval/
  candidates/
    cand_0000_initial/
      templates/
      candidate.json
    cand_0001/
      templates/
      candidate.json
  selected/
    templates/
    val_eval/
      summary.json
      results.json
      results.jsonl
    test_eval/
      summary.json
      results.json
      results.jsonl
  reports/
    optimization_trace.jsonl
    feedback_records.jsonl
    lm_calls.jsonl
    gepa_run_log.txt
    candidate_scores.csv
    selected_vs_initial.md
    proposals/
    proposal_diffs/
    sampling_bucket_refreshes/
```

`selected_vs_initial.md` 至少包含：

- best candidate index。
- total candidates。
- total metric calls。
- Validator failures。
- candidate eval cache hits / misses。
- Initial 和 Selected 在 val / test 上的 GEPA Score。
- backend accuracy。
- macro backend recall。
- No Decision rate。
- template accept rate。
- Selected 相比 Initial 的 delta。

`sampling_bucket_refreshes/` 中每次 refresh 会记录：

- 当前 best Candidate。
- 完整 train set refresh summary。
- 重建后的 sampling buckets。
- 对应 train results。

`lm_calls.jsonl` 记录 LLM request / response 和 Proposal merge diagnostics。

`feedback_records.jsonl` 记录每轮用于 Proposal LLM 的 feedback 内容，包括 Matcher outputs、Deterministic Feedback、LLM Feedback 和 reflective dataset record。

## 18. 结果解读

主要看：

- `backend_accuracy`：primary predicted backend 是否等于 expected backend。
- `macro_backend_recall`：三个 backend 的 recall 是否均衡。
- `template_accept_rate`：非 No Decision 的比例。
- `no_decision_rate`：未接受任何 backend 的比例。
- `top_confusion_pairs`：主要 backend 混淆方向。
- `suspected_overbroad_templates`：集中误吸的 template。
- `validator_failure_count`：Proposal LLM 输出与 Output Contract / schema 的冲突频率。

`template_accept_rate` 提升不一定代表 routing quality 提升。如果大量 No Decision 被错误 backend 吸收，`template_accept_rate` 会升高，但 `backend_accuracy` 和 `macro_backend_recall` 会暴露问题。

`temporal_wrong` 等 wrong bucket 在后期可能变大。原因是前期大量样本在 `no_decision_cases` 中；后期覆盖率提升后，这些样本会被路由到某个 backend，其中错误路由会显性进入对应 wrong bucket。


## 19. 下一步计划
我建议下一阶段目标从“提高覆盖率”切到“提高干净度和可泛化性”。
计划
1. 先清理 Selected 以当前 Selected 为输入，去重、删除明显重复或跨 intent 的 prototypes / hard negatives。目标不是继续涨数量，而是得到一个更干净的 Selected v2。
2. 强化 Validator 增加硬约束：
- 禁止同一 template 内重复 query_prototypes。
- 禁止同一 template 内重复 hard_negatives.query。
- 禁止同一 query 同时出现在同一 template 的 positive 和 hard negative 中。
- 对过低 threshold 加 warning 或 hard limit，避免单纯靠降 threshold 提升 accept rate。
3. 调整 GEPA Score 当前 score 对 template_accept_rate 奖励较强，容易鼓励“尽量都接住”。下一版应加入 false_accept_penalty。
false_accept 定义：matcher 接受了 backend，但 predicted_backend != expected_backend。
目标是：奖励正确接受，惩罚错误接受，而不是单纯奖励非 No Decision。
4. 加强 overbroad penalty 当前 overbroad_penalty 只在单 template 错误吸收比例超过 0.35 后才生效，太弱。下一版应让 graph.entity_relation、personal_fact_lookup、timeline_fact 这类集中误吸更早被惩罚。
5. 优化 LLM Feedback 和 Proposal Prompt 明确告诉 Proposal LLM：
- 不优先做纯 additive growth。
- 对 overbroad template，优先 rewrite / remove / add hard negatives。
- threshold 只能小幅调整，不能作为主要优化手段。
- 保留高质量 Parent 项，但删除重复、过宽、跨 intent 的项。
6. 调整 bucket_random 采样 后期 no_decision_cases 为空时，把这部分 quota 转给 wrong buckets，特别是：
- temporal_wrong
- openviking_wrong
- graph_wrong

这样后期 feedback 会更集中在 residual confusion，而不是继续扩大覆盖率。
7. 重新跑对照实验 至少跑两组：
- 当前代码 baseline：已有 20260602_002431
- 新 score + 新 Validator + 新 prompt

对比指标不只看 GEPA Score，还要看：
- backend accuracy
- macro backend recall
- false accept count
- duplicate count
- average prototypes / hard negatives per template
- top confusion pairs
我的判断：下一阶段最关键的是改 GEPA Score 和 Validator。否则 Proposal LLM 即使 prompt 写得更好，也仍然会被当前 score 引导到“降低 threshold + 大量添加 pattern”的方向。