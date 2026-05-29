# MemRouter Template YAML GEPA Matcher-Only 实验方案

## 1. 实验背景与目标

MemRouter 的任务是把用户问题路由到合适的 memory backend。当前 routing 链路中，matcher 会读取一组 YAML templates，并根据 `query_prototypes`、`hard_negatives` 和 threshold fields 预测 backend。

本实验把这组 YAML templates 作为唯一可优化资产。GEPA 在每轮迭代中读取当前 templates、routing 结果和 feedback，然后直接生成一组新的完整 YAML templates。matcher 代码和 matcher 配置在整个实验中保持不变。

本实验验证的问题是：

```text
在固定 matcher 代码和 matcher 配置的前提下，GEPA 直接优化 Template Bundle，是否能提升 matcher-only routing 效果？
```

本实验的输入样本是：

```text
<question, expected_backend>
```

默认数据来自 LoCoMo routing labels。每条样本包含一个 question 和一个人工标注的 expected backend。

本实验的核心输出是：

```text
Selected 是否在 val / test 上优于 Initial
```

本实验只评估 routing，不评估端到端答案质量。当 matcher 没有接受任何 backend 时，记录为 `No Decision`，不会调用 LLM fallback。

## 2. 术语

| 术语 | 定义 |
|---|---|
| `GEPA` | 用 score 和 feedback 驱动文本资产迭代优化的优化器。本实验中，GEPA 的输出是 Proposal。 |
| `Backend` | MemRouter 可选择的路由目标，例如 `openviking_memory_backend`、`graph_memory_backend`、`temporal_memory_backend`。 |
| `Matcher` | 固定代码模块，读取 Template Bundle 并为 question 预测 backend。 |
| `Template Bundle` | matcher 消费的一组 YAML templates。 |
| `Initial` | 实验开始前的原始 Template Bundle。 |
| `Parent` | 当前轮被 GEPA 修改的 Template Bundle。第一轮 Parent 等于 Initial。 |
| `Proposal` | GEPA 基于 Parent 生成的新 Template Bundle，格式为完整 YAML 文件集合。 |
| `Candidate` | 通过 minibatch 比较并进入候选池的 Proposal。 |
| `Selected` | 预算结束后，在 val set 上 `GEPA Score` 最高的 Candidate。 |
| `Minibatch` | 每轮 GEPA 用于生成 feedback 和比较 Parent / Proposal 的小批量样本。 |
| `train set` | 用于 GEPA 生成 feedback 和采样 minibatch 的训练集。 |
| `val set` | 用于评估 Candidate 并选择 Selected 的验证集。 |
| `test set` | Selected 固定后才使用的最终评估集。 |
| `expected_backend` | 样本的人工标注 backend。 |
| `predicted_backend` | matcher 对 question 输出的 backend。 |
| `Matcher Result` | matcher 对样本的原始 routing 输出，包括 predicted backend、template ranking、backend scores、matched template 和是否 No Decision。 |
| `No Decision` | matcher 没有接受任何 backend。本实验不会在这种情况下调用 LLM fallback。 |
| `LLM fallback` | matcher 无法决策时调用 LLM 兜底判断 backend 的路径。本实验禁用该路径。 |
| `Deterministic Feedback` | 固定代码基于 Matcher Result 和 expected backend 生成的结构化诊断。 |
| `LLM Feedback` | LLM 基于 Deterministic Feedback 生成的泛化修改建议。它不参与打分。 |
| `Validator` | 检查 Proposal 是否合法、是否可被 matcher 消费的固定程序。 |
| `Validator Feedback` | Validator 失败时生成的结构化错误反馈。 |
| `Generation Prompt` | 指导 GEPA 根据 Parent 和 feedback 生成 Proposal 的指令。 |
| `Output Contract` | Proposal 必须满足的输出格式和字段约束。 |
| `GEPA Score` | evaluator 计算的优化目标，用于比较 Parent、Proposal 和 Candidate。 |
| `num_threads` | GEPA / runner 用于并行执行 metric calls 的 worker 数。默认值为 4。 |
| `parallel_val_eval` | 使用 `num_threads` 并行评估 val set 的设置。 |
| `question_embedding_cache` | 以 question 和 embedding config 为 key 的 embedding 缓存。 |
| `template_embedding_cache` | 以 Template Bundle hash 和 embedding config 为 key 的 template embedding 缓存。 |
| `candidate_eval_cache` | 以 Candidate hash、split 和 evaluator config 为 key 的评估结果缓存。 |
| `Run Resume` | 从已有 run 目录恢复实验状态的能力。 |

## 3. 实验边界

本实验明确不做：

- 不优化 matcher 代码。
- 不优化 matcher 配置。
- 不设置 LLM fallback；No Decision 会直接进入指标统计。
- 不使用 answer judge。
- 不让 LLM Feedback 参与打分或接受判断。
- 不让 test set 参与 GEPA 生成、选择或 early stop。
- 不使用 patch 或 edit JSON 作为 Proposal 输出格式。

## 4. 总体流程

实验由 data preparation、GEPA optimization、final evaluation 三个阶段组成。

```text
1. 固定 matcher 代码和 matcher 配置。
2. 准备 train / val / test split。
3. 使用 Initial 跑完整 train set，建立 sampling buckets。
4. 使用 Initial 跑 val set，初始化 Candidate pool。
5. GEPA 迭代：
   5.1 选择 Parent。
   5.2 从 sampling buckets 中抽取 minibatch。
   5.3 使用 Parent 跑 minibatch，得到 Matcher Result。
   5.4 evaluator 生成 GEPA Score 和 Deterministic Feedback。
   5.5 LLM 基于 Deterministic Feedback 生成 LLM Feedback。
   5.6 GEPA 基于 Parent、feedback 和 Generation Prompt 生成 Proposal。
   5.7 Validator 检查 Proposal。
       - 如果失败：Proposal 得分为 0，Validator Feedback 进入下一轮。
       - 如果通过：继续下一步。
   5.8 使用 Proposal 跑同一个 minibatch。
   5.9 如果 Proposal 的 GEPA Score 高于 Parent，则跑 val set 并加入 Candidate pool。
6. 预算结束后，选择 val set 上 GEPA Score 最高的 Candidate 作为 Selected。
7. 使用 test set 评估 Initial 和 Selected。
8. 输出 Selected vs Initial 报告。
```

关键约束：

- 每轮 feedback 必须基于当前 Parent 的最新 Matcher Result。
- Proposal 和 Parent 必须在同一个 minibatch 上比较。
- test set 只在 Selected 固定后使用一次。
- Selected 只由 val set 选择，不能由 test set 选择。

## 5. GEPA 输入

每轮 GEPA 输入包含：

| 输入 | 定义 |
|---|---|
| `Generation Prompt` | 说明任务、允许修改范围、输出格式和禁止事项。 |
| `Output Contract` | Proposal 必须满足的格式与合法性约束。 |
| `Parent` | 当前轮完整 YAML Template Bundle。 |
| `Minibatch Cases` | 当前轮样本，格式为 `<question, expected_backend>`。 |
| `Deterministic Feedback` | 当前 Parent 在 minibatch 上的结构化诊断。 |
| `LLM Feedback` | 基于 Deterministic Feedback 的泛化修改建议。 |
| `Validator Feedback` | 最近一次 Proposal 非法时的错误反馈；如果最近一次 Proposal 合法则为空。 |
| `Parent Val Summary` | Parent 最近一次在 val set 上的整体指标摘要。 |
| `Initial Summary` | Initial 在 train / val 上的静态指标摘要。 |

推荐输入优先级：

```text
1. Output Contract
2. Validator Feedback
3. Parent
4. Deterministic Feedback
5. LLM Feedback
6. Parent Val Summary
7. Initial Summary
8. Minibatch Cases
```

`Parent Val Summary` 和 `Initial Summary` 只传压缩摘要，不传完整样本。

## 6. Proposal 输出

GEPA 直接输出 `Proposal`，即完整 YAML Template Bundle。

不使用 patch。
不使用 edit JSON。
不要求 GEPA 描述修改理由。

输出要求：

- 输出完整 YAML 文件内容。
- 保留 Parent 中的全部 template。
- 保持 template 顺序稳定。
- 保持 YAML 可解析。
- 不输出 Markdown fence。
- 不输出解释文本。
- 不复制 train query 原文作为 prototype。

允许修改的字段：

- `query_prototypes`
- `hard_negatives`
- 明确列入白名单的 threshold fields，例如 `accept_threshold`、`fallback_threshold`、`margin`

说明：如果现有 YAML 中保留 `fallback_threshold` 这个字段名，它只表示 matcher 的阈值字段，不表示本实验存在 LLM fallback。

禁止修改的字段：

- `template_id`
- `target_backend`
- `query_spec`
- backend 名称
- template ownership
- matcher 不消费的结构字段

## 7. Validator

Validator 是 Proposal 进入 matcher 前的硬门槛。

Validator 检查：

- YAML 可解析。
- Template Bundle 中的 template 数量与 Parent 一致。
- 每个 `template_id` 与 Parent 一致。
- 每个 `target_backend` 与 Parent 一致。
- `query_spec` 与 Parent 一致。
- 只修改白名单字段。
- `query_prototypes` 和 `hard_negatives` 是字符串列表。
- 单条文本长度、列表长度和总文本增长量不超过上限。
- threshold fields 是数值，并落在合法范围。
- Template Bundle 可被 template loader 加载。
- matcher 可基于 Proposal 初始化。
- smoke routing 能跑通少量样本。

当 Validator 失败：

```text
Proposal score = 0
Proposal 不进入 Candidate pool
不运行 matcher evaluation
Validator Feedback 进入下一轮 GEPA 输入
```

`Validator Feedback` 示例：

```json
{
  "validator_status": "failed",
  "errors": [
    {
      "type": "immutable_field_changed",
      "template_id": "graph.entity_relation.v1",
      "field": "target_backend",
      "message": "target_backend must be identical to Parent."
    },
    {
      "type": "yaml_parse_error",
      "message": "Invalid indentation near line 42."
    }
  ],
  "required_fix": [
    "Return complete YAML only.",
    "Keep template_id, target_backend, and query_spec unchanged.",
    "Only modify query_prototypes, hard_negatives, and allowed threshold fields."
  ]
}
```

## 8. Feedback 设计

### 8.1 串行关系

本实验采用串行 feedback：

```text
Matcher Result
-> Deterministic Feedback
-> LLM Feedback
-> GEPA
```

串行关系的含义是：固定代码先把 Matcher Result 归一化为 Deterministic Feedback，LLM 再基于 Deterministic Feedback 生成泛化建议。LLM 不直接重做 evaluator 判断。

LLM Feedback 可以读取少量 compact Matcher Result 证据，但它的主输入是 Deterministic Feedback。

### 8.2 Deterministic Feedback

Deterministic Feedback 的职责是说明“发生了什么”。

每个样本至少包含：

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
- `top_backend_ranking`
- `top_template_ranking`
- `diagnosis`
- `suggested_action_type`

batch 级摘要至少包含：

- `GEPA Score`
- backend accuracy
- macro backend recall
- No Decision rate
- template accept rate
- per-backend recall
- top confusion pairs
- suspected overbroad templates
- newly fixed cases
- newly broken cases

### 8.3 Deterministic Feedback 规则

规则应由固定代码生成。

推荐规则：

| 条件 | diagnosis | suggested_action_type |
|---|---|---|
| expected backend rank > 2 | expected backend coverage is weak | add generalized query prototypes |
| expected backend rank <= 2 且 margin_to_win 较小 | expected backend is close but lacks margin | add discriminative prototypes or hard negatives |
| wrong backend score 明显领先 | wrong template may be overbroad | add hard negatives to wrong template |
| expected backend rank == 1 但 No Decision | acceptance threshold may be too strict | small threshold adjustment |
| No Decision | templates did not confidently accept the query | improve coverage without broad threshold lowering |
| 同一 wrong template 多次误吸 | template is likely overbroad | add hard negatives and narrow prototypes |
| 当前样本正确 | route should be preserved | avoid changes that reduce this behavior |

### 8.4 LLM Feedback

LLM Feedback 的职责是说明“如何泛化修改”。

LLM Feedback 不直接访问 test set。
LLM Feedback 不参与 score。
LLM Feedback 不决定 Proposal 是否接受。

LLM Feedback prompt 应要求：

- 只基于 Deterministic Feedback 和提供的 summary。
- 输出泛化建议，不输出新的 YAML。
- 不复制具体 question。
- 不包含具体人名、时间、事件细节。
- 给出每个建议对应的 target template。
- 说明建议修改的是 `query_prototypes`、`hard_negatives` 还是 threshold fields。
- 对正确样本给出 preserve 提醒。
- 对 Validator Feedback 给出优先修复建议。

LLM Feedback 输出示例：

```json
{
  "feedback_type": "llm_feedback",
  "summary": "Graph relation queries are under-covered, while one OpenViking personal fact template is absorbing relation-style questions.",
  "template_suggestions": [
    {
      "template_id": "graph.entity_relation.v1",
      "field": "query_prototypes",
      "action": "add",
      "guidance": "Add generalized prototypes for questions asking about relationships, shared interests, and links between two people."
    },
    {
      "template_id": "openviking.personal_fact.v1",
      "field": "hard_negatives",
      "action": "add",
      "guidance": "Add hard negatives for questions whose main intent is a relation between two people rather than a personal attribute."
    }
  ],
  "preserve": [
    "Do not weaken templates that correctly route personal preference questions to OpenViking."
  ]
}
```

## 9. Generation Prompt

Generation Prompt 的职责是指导 GEPA 从 Parent 生成 Proposal。

Prompt 必须强调：

- 任务是改进 matcher-only routing。
- 不存在 LLM fallback。
- Proposal 必须是完整 YAML Template Bundle。
- 只允许修改白名单字段。
- 不允许改变 template ownership。
- 不允许删除或新增 template。
- 不允许复制 minibatch question。
- 修改应具有泛化性。
- Validator Feedback 优先级最高。
- 正确样本是 regression anchors。

Generation Prompt 推荐结构：

```text
You are optimizing a YAML Template Bundle for a deterministic matcher.

Goal:
Improve matcher-only routing accuracy on future samples.

Output:
Return the complete YAML Template Bundle only.

Allowed changes:
- query_prototypes
- hard_negatives
- allowed threshold fields

Forbidden changes:
- template_id
- target_backend
- query_spec
- adding or deleting templates
- copying exact training questions

Inputs:
- Parent
- Validator Feedback
- Deterministic Feedback
- LLM Feedback
- Parent Val Summary
- Initial Summary

Decision rule:
Prefer generalized template improvements that fix the current minibatch while preserving correct behavior.
```

## 10. GEPA Score

`GEPA Score` 用于比较 Parent、Proposal 和 Candidate。

默认公式：

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
no_decision_control_score = 1 - no_decision_rate
```

`regression_penalty` 惩罚 Parent 原本正确但 Proposal 改错的样本。

`overbroad_penalty` 惩罚某个 template 在 batch 中集中误吸。

本实验不设置 hard guard。Candidate 是否接受只看同 minibatch 上 Proposal 是否高于 Parent。

## 11. Sampling

先用 Initial 跑完整 train set，建立固定 sampling buckets。

推荐 buckets：

```text
graph_correct
graph_wrong
temporal_correct
temporal_wrong
openviking_correct
openviking_wrong
no_decision_cases
random_pool
```

每轮 minibatch size 默认 16。

推荐配额：

| bucket 类型 | 数量 |
|---|---:|
| initial wrong graph | 2 |
| initial wrong temporal | 2 |
| initial wrong openviking | 2 |
| initial No Decision | 2 |
| initial correct graph guard | 2 |
| initial correct temporal guard | 2 |
| initial correct openviking guard | 2 |
| random balanced fill | 2 |

每轮抽到的样本必须用当前 Parent 重新 routing。

## 12. Candidate 选择

### 12.1 Proposal 接受

每轮比较：

```text
Proposal GEPA Score on minibatch > Parent GEPA Score on same minibatch
```

如果成立：

- Proposal 跑完整 val set。
- Proposal 加入 Candidate pool。

如果不成立：

- Proposal 丢弃。

### 12.2 Parent 选择

默认策略：

```text
Parent = val set GEPA Score 最高的 Candidate
```

### 12.3 最终选择

预算结束后：

```text
Selected = val set GEPA Score 最高的 Candidate
```

最终报告必须比较：

```text
Selected vs Initial on val
Selected vs Initial on test
```

## 13. 执行效率与恢复设置

本实验默认启用以下执行设置：

| 设置 | 默认值 | 含义 |
|---|---:|---|
| `num_threads` | 4 | GEPA / runner 用于并行执行 metric calls 的 worker 数。 |
| `batch_size` | 16 | 每轮 GEPA 使用的 minibatch 大小。 |
| `early_stop_rounds` | 20 | 连续 20 个合法 Proposal 未刷新 best val GEPA Score 时停止优化。 |
| `max_invalid_proposals` | 5 | 连续 5 个 Proposal 未通过 Validator 时停止优化并输出 Validator failure report。 |
| `template_embedding_cache` | enabled | 复用 Template Bundle 的 embedding 结果。 |
| `question_embedding_cache` | enabled | 复用 question 的 embedding 结果。 |
| `parallel_val_eval` | enabled | 使用 `num_threads` 并行评估 val set。 |
| `candidate_eval_cache` | enabled | 避免重复评估相同 Candidate。 |
| `Run Resume` | enabled | 允许从已有 run 目录恢复优化。 |

缓存 key 设计：

```text
question_embedding_cache key =
  hash(question, embedding_config)

template_embedding_cache key =
  hash(template_bundle, embedding_config)

candidate_eval_cache key =
  hash(candidate, split_name, split_hash, evaluator_config)
```

并行执行约束：

- `num_threads` 只用于 metric calls 和 matcher evaluation，不用于并行生成 Proposal。
- LLM Feedback 和 Proposal 生成保持串行，避免不同轮次的 feedback 互相污染。
- `parallel_val_eval` 必须保持结果聚合顺序稳定。
- 每个 worker 必须使用独立临时目录和日志文件。
- Candidate、split 和 evaluator config 未变化时，优先读取 `candidate_eval_cache`。

Run Resume 必须恢复：

- Candidate pool。
- 当前 Selected。
- GEPA metric call 计数。
- random seed 和 sampler 状态。
- Validator failure 计数。
- `early_stop_rounds` 计数。
- cache manifest。

恢复时必须校验 config snapshot。如果当前 CLI 参数和原 run 的 config snapshot 不一致，runner 应拒绝恢复。

## 14. 数据准备

建议实现入口：

```text
scripts/prepare_memrouter_template_yaml_gepa_dataset.py
```

输入：

```text
benchmarks/locomo/data/locomo_e2e_route_labels.v2.jsonl
```

输出：

```text
benchmarks/locomo/data/gepa_template_yaml/
  locomo_v2_template.train.jsonl
  locomo_v2_template.val.jsonl
  locomo_v2_template.test.jsonl
  locomo_v2_template.manifest.json
  locomo_v2_template.data_report.md
```

要求：

- split 只包含 train / val / test。
- 默认按 `sample_id` 分组切分，避免同一 conversation 同时出现在不同 split。
- manifest 记录 input hash、seed、split ratios、backend 分布、category 分布和 warnings。
- 由于 LoCoMo 只有 10 个 conversation，正式报告应记录 seed，并建议多 seed repeat。

## 15. Runner

建议实现入口：

```text
scripts/optimize_memrouter_template_yaml_gepa.py
```

Dry-run：

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

Full run：

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
  --batch-size 16 \
  --num-threads 4 \
  --early-stop-rounds 20 \
  --max-invalid-proposals 5 \
  --enable-template-embedding-cache \
  --enable-question-embedding-cache \
  --enable-candidate-eval-cache \
  --parallel-val-eval \
  --seed 13 \
  --max-metric-calls 5000 \
  --target-accepted-candidates 15
```

`max_metric_calls` 需要按 val size 调整。若 val set 约 300 条，15 个 accepted Candidate 通常需要数千次 metric call。

恢复已有 run：

```bash
uv run python scripts/optimize_memrouter_template_yaml_gepa.py \
  --resume-run runs/memrouter_template_yaml_gepa/<run_id>
```

`--resume-run` 只从 run 目录读取 config snapshot，不允许用新的 CLI 参数隐式覆盖旧配置。

## 16. 产物

推荐 run 目录：

```text
runs/memrouter_template_yaml_gepa/<run_id>/
  config_snapshot.yaml
  split_summary.json
  run_state.json
  cache_manifest.json
  cache/
    question_embeddings/
    template_embeddings/
    candidate_eval/
  tmp/
    worker_0/
    worker_1/
    worker_2/
    worker_3/
  initial/
    train_eval/
    val_eval/
    test_eval/
  sampling_buckets.json
  candidates/
    cand_0000_initial/
      templates/
      val_eval/
    cand_0001/
      templates/
      validator_report.json
      minibatch_eval.json
      val_eval/
      candidate_summary.md
  selected/
    templates/
    val_eval/
    test_eval/
  reports/
    optimization_trace.jsonl
    candidate_scores.csv
    validator_failures.jsonl
    selected_vs_initial.md
```

最小报告包含：

- Initial val / test metrics。
- Selected val / test metrics。
- Selected 相对 Initial 的 delta。
- accepted / rejected Proposal 数量。
- Validator failure 数量和主要原因。
- per-backend recall。
- backend accuracy。
- No Decision rate。
- template accept rate。
- top confusion pairs。
- suspected overbroad templates。
- 每个 Candidate 的 val GEPA Score。
- cache hit rate。
- resume status。

## 17. 实施清单

1. 实现 train / val / test 数据准备脚本。
2. 实现 matcher-only evaluator，禁用 LLM fallback。
3. 实现 Matcher Result schema。
4. 实现 GEPA Score。
5. 实现 Deterministic Feedback。
6. 实现 LLM Feedback prompt 和 parser。
7. 实现 Generation Prompt。
8. 实现 Proposal YAML 输出读取。
9. 实现 Validator。
10. 实现 Validator Feedback 回流。
11. 实现 sampling buckets 和 minibatch sampler。
12. 实现 Candidate pool。
13. 实现 `num_threads=4` 的 matcher evaluation 并行。
14. 实现 question / template / candidate eval 缓存。
15. 实现 Run Resume。
16. 实现 Selected vs Initial 报告。

## 18. 后续升级

如果本实验有效，再考虑：

- 多 seed repeat。
- 动态刷新 sampling buckets。
- 引入 Pareto Parent selection。
- 增加 hard guard。
- 加入 per-template precision。
- 加入 backend-specific LLM Feedback prompt。
- 在 matcher 支持后开放更多 template 字段。
