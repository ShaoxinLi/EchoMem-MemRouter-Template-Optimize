# MemRouter Template-Only GEPA 优化实验方案

本文档定义一个简化版 MemRouter routing 优化实验。目标是先验证：在固定 matcher 代码和 matcher 配置的前提下，GEPA 是否能够仅通过优化 templates 来提升 routing accuracy。

该方案是对既有 `docs/memrouter_optimize_plan.md` 的收敛版本，不做多阶段 matcher/template/joint 优化，也不做 per-candidate ablation。第一版重点把 GEPA 的 score、feedback、candidate 合法性和 minibatch 采样机制做稳定。

## 1. 实验目标

### 1.1 要验证的问题

核心问题：

```text
fixed matcher + optimized templates 是否能稳定优于 initial templates？
```

这里的 templates 指 matcher 实际会消费的字段，优先包括：

- `query_prototypes`
- `hard_negatives`
- template threshold fields，例如 `accept_threshold`、`fallback_threshold`、`margin`

如果某些字段不进入 matcher scoring 或 RouteDecision，不作为第一版优化对象。

### 1.2 不做的事情

第一版明确不做：

- 不优化 matcher 代码。
- 不优化 matcher policy。
- 不做多阶段 template / threshold / matcher / joint pipeline。
- 不让 GEPA 输出 patch。
- 不做每个 candidate 的 prototype-only / hard-negative-only / threshold-only ablation。
- 不设置 hard guard 丢弃 candidate。
- 不让 test set 参与 GEPA 生成、选择或 early stop。

### 1.3 术语定义

为避免歧义，本文档使用以下术语：

| 术语 | 含义 |
|---|---|
| `initial` | 优化开始前的原始 templates，也就是 `cand_0` |
| `parent` | GEPA 当前轮从候选池中选出、准备被修改的 candidate |
| `proposal` | GEPA 基于 `parent` 生成的新 templates |
| `candidate` | 已被接受并进入 GEPA 候选池的 templates 版本 |
| `selected` | 优化结束后，在 val set 上选出的最终 candidate |
| `train minibatch` | 每轮用于 reflection、proposal 和 proposal-vs-parent 验证的小批量 train samples |
| `val set` | GEPA 用于 candidate 评分和最终选择的数据，不参与 proposal 生成 |
| `test set` | 只在 `selected` 固定后评估一次 |

GEPA 内部接受比较是：

```text
proposal 是否优于 parent
```

最终实验汇报比较是：

```text
selected 是否优于 initial
```

## 2. 总体流程

第一版流程如下：

```text
1. 固定 matcher 代码和 matcher config
2. 划分 train / val / test
3. 使用 initial templates 跑完整 train set，建立 sampling buckets
4. 使用 initial templates 跑 val set，初始化 GEPA candidate pool
5. GEPA 迭代：
   5.1 从候选池选择 parent
   5.2 从固定 sampling buckets 中采样 12 条 train samples
   5.3 使用当前 parent 重新 routing 这 12 条样本
   5.4 evaluator 计算 GEPA score 并生成 deterministic feedback
   5.5 GEPA 根据 feedback 生成 proposal templates
   5.6 最小 validator 检查 proposal 合法性和 matcher 可加载性
   5.7 使用 proposal 在同一批 12 条样本上重新 routing
   5.8 如果 proposal score > parent score，则在 val set 上评估并加入候选池
6. 预算结束后，从候选池选择 val score 最好的 selected
7. 使用 test set 对 selected 做最终评估
8. 报告 selected vs initial
```

注意：sampling buckets 由 `initial` 的完整 train routing 结果建立，但每轮 feedback 必须基于当前 `parent` 在本轮 minibatch 上的最新 routing 结果生成。

## 3. Candidate 生成与合法性

### 3.1 GEPA 输出形式

第一版不采用 patch。GEPA 直接输出一个完整的 candidate template bundle。

推荐输出 JSON，原因是更容易解析和校验。结构示例：

```json
{
  "templates": [
    {
      "template_id": "graph.entity_relation.v1",
      "query_prototypes": [
        "What relationship does one person have with another?",
        "What common interest did two people share?"
      ],
      "hard_negatives": [
        "When did the person meet someone?",
        "How long did the event last?"
      ],
      "thresholds": {
        "accept_threshold": 0.62,
        "fallback_threshold": 0.48,
        "margin": 0.04
      }
    }
  ]
}
```

这里的 candidate bundle 不是 diff，也不是 patch。runner 会把 source templates 复制到临时 candidate workspace，然后只写回白名单字段。

### 3.2 Prompt 约束

prompt 应尽可能约束 GEPA 生成合法、可被 matcher 消费的 templates：

- 只输出 JSON，不输出解释文本。
- 只包含允许修改的字段。
- `template_id` 必须来自提供的 template list。
- 不允许新增 backend。
- 不允许修改 template ownership，例如 `target_backend`。
- `query_prototypes` 和 `hard_negatives` 必须是字符串数组。
- threshold 字段必须是数值，并落在允许范围内。
- 不要复制 train query 原文作为 prototype；应写成泛化模式。
- 不要生成过长、过窄、包含具体人名或具体事件细节的 prototype。

prompt 约束只降低非法输出概率，不能替代 validator。

### 3.3 最小 Validator

GEPA 不保证输出合法。runner 必须在每次 proposal 进入 routing eval 之前执行最小 validator。

validator 检查：

- JSON 可解析。
- 顶层 schema 正确。
- `template_id` 存在于 source templates。
- 只修改白名单字段。
- 不修改 `template_id`、`target_backend`、`query_spec` 等结构字段。
- `query_prototypes` / `hard_negatives` 是字符串列表。
- 单条文本长度、列表长度和总新增规模不超过上限。
- threshold 是数值，范围合法，例如 `[0.0, 1.0]`。
- 数值可按固定步长量化，避免无意义小数抖动。
- materialize 后 template store 能加载。
- matcher 能基于 candidate templates 初始化。
- smoke routing 至少能跑通 1-3 条样本。

非法 candidate 的处理：

```text
score = 0
feedback = validator errors + expected schema reminder
```

validator 不负责判断 candidate 是否“好”，只负责判断它是否可被 matcher 正确使用。

## 4. Stratified Minibatch

### 4.1 为什么需要 Stratified Minibatch

当前 routing 数据分布不均衡，openviking 样本多，graph 样本少。纯随机 minibatch 容易出现两个问题：

- GEPA 主要学习 openviking 主流样本，忽略 graph/temporal 短板。
- 只采失败样本会导致局部修复，伤害原本正确的主流样本。

因此每个 train minibatch 应同时包含失败样本和成功样本，并尽量覆盖不同 backend。

### 4.2 Sampling Buckets

优化开始前，使用 `initial` 跑完整 train set，得到每条样本的初始 routing 结果。基于这些结果建立固定 buckets：

```text
graph_correct
graph_wrong
temporal_correct
temporal_wrong
openviking_correct
openviking_wrong
fallback_cases
random_pool
```

这些 buckets 只用于采样，不用于生成每轮 feedback。

### 4.3 当前 Parent 必须重新 Routing

当 GEPA 选择了新的 `parent` 后，train set 上样本的 correct/wrong 状态可能已经改变。第一版不在每次 parent 变化后全量 reroute train set，因为成本过高。

第一版采用：

```text
initial routing result -> 固定 sampling buckets
current parent routing result on minibatch -> 当前 feedback
```

也就是说：

- buckets 由 `initial` 结果决定。
- 每轮抽到的 12 条样本，必须用当前 `parent` 重新 routing。
- score 和 feedback 只基于当前 `parent` 的最新 minibatch routing 结果。

后续可升级为：

- 每隔 N 个 accepted candidates 用当前 best candidate 刷新 buckets。
- 维护动态 case stats，只更新被采样过的样本状态。
- 混合采样：70% 来自 initial buckets，30% 来自最近 parent failures。

第一版不做动态刷新。

### 4.4 Batch Size 和采样配额

第一版设置：

```text
reflection_minibatch_size = 12
```

推荐目标配额：

| bucket 类型 | 目标数量 |
|---|---:|
| initial wrong graph | 2 |
| initial wrong temporal | 2 |
| initial wrong openviking 或 fallback | 2 |
| initial correct graph/temporal guard | 2 |
| initial correct openviking guard | 2 |
| random balanced fill | 2 |

如果某个 bucket 不足，按以下顺序回填：

1. 同 backend 的其他 wrong/correct bucket。
2. 其他稀有 backend bucket。
3. `random_pool` 中最少被采样过的样本。

样本允许跨 iteration 重复出现，但采样器应记录频次，避免少数样本被过度重复。

## 5. GEPA Score

### 5.1 Score 的作用

本文档中的 GEPA score 是 evaluator 反馈给 GEPA 的优化目标，不是 matcher 内部用于选择 backend 的 score。

GEPA score 用于：

- 比较 `proposal` 和 `parent` 在同一 train minibatch 上是否有提升。
- 评估 candidate 在 val set 上的表现。
- 最终选择 `selected`。

### 5.2 第一版公式

第一版采用组合分：

```text
gepa_score =
  0.35 * decision_backend_accuracy
+ 0.25 * macro_backend_recall
+ 0.10 * expected_backend_top2_rate
+ 0.10 * expected_backend_margin_score
+ 0.08 * template_accept_rate
+ 0.07 * fallback_control_score
+ 0.05 * overbroad_control_score
- regression_penalty
```

所有正向子指标归一化到 `[0, 1]`，越高越好。

其中：

```text
fallback_control_score = 1 - fallback_rate
```

如果当前实现里 `fallback_rate` 与 `template_accept_rate` 基本互补，这两个项会共同形成一个明确的 template-first 信号。保留两者是有意的：`template_accept_rate` 告诉 GEPA “让 templates 接住更多 query”，`fallback_control_score` 告诉 GEPA “减少进入 fallback 的比例”。正确率和 macro recall 仍占 0.60，用于防止 candidate 为了减少 fallback 而错误 accept。

### 5.3 子指标定义

`decision_backend_accuracy`：

```text
最终 predicted_backend 是否等于 expected_backend 的平均值
```

`macro_backend_recall`：

```text
对每个 expected_backend 分别计算 recall，然后取平均
```

该指标用于避免 openviking 占比过高时掩盖 graph/temporal。

`expected_backend_top2_rate`：

```text
expected_backend 是否进入 predecision top2 backend ranking
```

该指标提供更密的中间信号。即使最终 routing 没选中 expected backend，只要 expected backend 排名上升，也能给 GEPA 一个方向信号。

`expected_backend_margin_score`：

```text
margin = expected_backend_score - best_wrong_backend_score
```

推荐归一化：

```text
margin_score = clamp((margin + 0.20) / 0.40, 0, 1)
```

含义：

- `margin >= 0.20` 视为强正确信号。
- `margin = 0.00` 约为 0.5，表示 expected backend 接近胜出。
- `margin <= -0.20` 视为明显落后。

`template_accept_rate`：

```text
通过 template matcher 直接接受并产出 routing decision 的样本比例
```

该项用于鼓励 templates 覆盖更多可路由 query。它不单独判断 route 是否正确；正确性仍由 `decision_backend_accuracy` 和 `macro_backend_recall` 控制。

`fallback_control_score`：

```text
fallback_control_score = 1 - fallback_rate
```

`fallback_rate` 指进入 fallback routing 的样本比例。该项用于明确惩罚 candidate 过度依赖 fallback。即使 fallback 最终预测正确，也会在该项上损失分数，因为本实验的目标是验证 templates-only 是否能提升 routing，而不是依赖 fallback 兜底。

`overbroad_control_score`：

用于惩罚某个 template 在当前 batch 中过度吸收错误 query。第一版可以按 batch 内错误命中的集中度计算：

```text
wrong_hit_concentration = max_wrong_hits_by_one_template / max(total_wrong_hits, 1)
overbroad_control_score = 1 - wrong_hit_concentration
```

如果没有错误命中，则该项为 1。

`regression_penalty`：

第一版不使用 hard guard，但 score 可以软惩罚明显回归。建议只在当前评估 batch 内计算：

```text
regression_penalty = 0.05 * newly_broken_initial_correct_rate
```

这里的 `initial_correct` 指 `initial` 在同一批样本上 routing 正确，而当前 candidate routing 错误。该项不直接丢弃 candidate，只降低 GEPA score。

### 5.4 备选强信号

第一版先记录但不进入 score 的指标：

- per-backend precision
- per-template precision
- top template concentration
- expected template rank
- wrong backend score gap
- candidate change size
- prototype duplication rate
- exact train query leakage rate

如果第一版 score 不稳定，再考虑把其中部分指标加入 GEPA score。

## 6. Deterministic Feedback Evaluator

### 6.1 Deterministic 的含义

这里的 deterministic 指 feedback 由固定代码生成，不由 LLM 判断。

LLM 的职责是：

```text
读取 score + feedback -> 生成新的 candidate templates
```

evaluator 的职责是：

```text
根据 routing result、gold label、top-k ranking、template hit 和 validator 结果 -> 生成 score + feedback
```

### 6.2 Feedback 输入

每个样本的 feedback 至少使用以下信息：

- `case_id`
- `question`
- `expected_backend`
- `predicted_backend`
- `is_correct`
- `routing_method`
- `template_accepted`
- `fallback_used`
- `fallback_reason`
- `expected_backend_rank`
- `expected_backend_score`
- `winning_backend`
- `winning_backend_score`
- `margin_to_win`
- top backend ranking
- top template ranking
- matched template id
- matched prototype 或 nearest prototype
- hard negative 命中信息

每个 batch 的 feedback 还应包含：

- batch 指标摘要。
- batch `template_accept_rate` 和 `fallback_rate`。
- fallback cases 的数量、expected backend 分布和主要原因。
- 当前 batch 中修复和新增错误的数量。
- 当前 batch 中最常见 confusion pair。
- 当前 batch 中疑似 overbroad template。
- `initial` 在完整 train set 上的静态摘要，例如 backend 分布、initial confusion matrix、低召回 backend。
- 当前 `parent` 在 val set 上的最近一次摘要，如果可用。

这些全局上下文用于提醒 GEPA 不要只针对当前 12 条样本做过窄修改。

### 6.3 Per-Case Diagnosis 规则

feedback 中的 diagnosis 和 suggested actions 由固定规则生成。

#### 格式非法

条件：

```text
validator failed
```

feedback：

```text
Candidate is invalid. Fix JSON schema, template ids, field types, threshold ranges, and make sure matcher can load the templates.
```

#### 正确 backend 排名很低

条件：

```text
predicted_backend != expected_backend
expected_backend_rank > 2
```

diagnosis：

```text
expected backend is under-covered by current query_prototypes
```

suggested actions：

```text
add generalized query_prototypes to the expected backend template
avoid copying the concrete train query
```

#### 正确 backend 接近胜出但未胜出

条件：

```text
predicted_backend != expected_backend
expected_backend_rank <= 2
margin_to_win is small
```

diagnosis：

```text
expected backend is close but lacks discriminative margin
```

suggested actions：

```text
add more discriminative query_prototypes to expected template
add hard_negatives to the winning wrong template if the wrong match is semantically over-broad
```

#### 错误 backend 分数明显领先

条件：

```text
predicted_backend != expected_backend
winning_backend_score - expected_backend_score is large
```

diagnosis：

```text
wrong backend has an over-broad template match or expected backend lacks coverage
```

suggested actions：

```text
add hard_negatives to the wrong winning template
add positive prototypes only if expected backend has no close match
```

#### 正确 backend top1 但未被接受

条件：

```text
expected_backend_rank == 1
predicted_backend != expected_backend
routing_method indicates fallback or reject
```

diagnosis：

```text
expected backend ranks first but acceptance is too strict
```

suggested actions：

```text
consider small threshold adjustment for the relevant template
do not broadly lower thresholds for unrelated templates
```

#### 进入 fallback

条件：

```text
fallback_used == true
```

diagnosis：

```text
query was not confidently accepted by templates and fell back to fallback routing
```

suggested actions：

```text
if expected backend has a close template match, add discriminative query_prototypes or consider a small threshold adjustment
if expected backend rank is low, add generalized query_prototypes to the expected backend template
if a wrong template nearly accepted the query, add hard_negatives to that wrong template
preserve correctness on non-fallback guard cases
```

#### batch fallback rate 偏高

条件：

```text
batch fallback_rate is high relative to the target range or parent val summary
```

diagnosis：

```text
candidate relies too much on fallback instead of template acceptance
```

suggested actions：

```text
prioritize template changes that convert fallback cases into correct template accepts
avoid broad threshold lowering that increases wrong template accepts
```

#### 某个 template 在 batch 中频繁误吸

条件：

```text
same wrong template appears in multiple incorrect cases
```

diagnosis：

```text
template appears over-broad in this batch
```

suggested actions：

```text
add hard_negatives to that template
make its query_prototypes more specific only if allowed by the output schema
```

#### 当前样本正确

条件：

```text
predicted_backend == expected_backend
```

diagnosis：

```text
current route is correct; preserve this behavior
```

suggested actions：

```text
avoid changes that would reduce this backend/template score
```

Correct cases should appear in feedback because they act as regression anchors.

### 6.4 Feedback 输出格式

推荐每轮 feedback 包含 batch summary 和 compact per-case feedback：

```json
{
  "batch_summary": {
    "gepa_score": 0.71,
    "decision_backend_accuracy": 0.75,
    "macro_backend_recall": 0.68,
    "expected_backend_top2_rate": 0.83,
    "expected_backend_margin_score": 0.61,
    "template_accept_rate": 0.67,
    "fallback_rate": 0.33,
    "fallback_control_score": 0.67,
    "overbroad_control_score": 0.80,
    "regression_penalty": 0.02,
    "fallback_case_count": 4,
    "fallback_expected_backend_counts": {
      "graph_memory_backend": 2,
      "temporal_memory_backend": 1,
      "openviking_memory_backend": 1
    },
    "most_common_confusions": [
      ["graph_memory_backend", "openviking_memory_backend", 2]
    ],
    "suspected_overbroad_templates": [
      "openviking.personal_fact.v1"
    ]
  },
  "global_context": {
    "initial_train_summary": "...",
    "parent_val_summary": "..."
  },
  "cases": [
    {
      "case_id": "...",
      "question": "...",
      "expected_backend": "graph_memory_backend",
      "predicted_backend": "openviking_memory_backend",
      "routing_method": "fallback",
      "template_accepted": false,
      "fallback_used": true,
      "fallback_reason": "no template exceeded accept threshold",
      "expected_backend_rank": 3,
      "expected_backend_score": 0.42,
      "winning_backend": "openviking_memory_backend",
      "winning_backend_score": 0.61,
      "margin_to_win": -0.19,
      "top_templates": [
        {
          "template_id": "openviking.personal_fact.v1",
          "backend": "openviking_memory_backend",
          "score": 0.61,
          "matched_prototype": "..."
        },
        {
          "template_id": "graph.entity_relation.v1",
          "backend": "graph_memory_backend",
          "score": 0.42,
          "matched_prototype": "..."
        }
      ],
      "diagnosis": "expected backend is under-covered and wrong backend appears over-broad",
      "suggested_actions": [
        "add generalized graph query_prototypes for shared/common relationship intent",
        "add hard_negatives to openviking.personal_fact.v1 for two-person shared-interest questions"
      ]
    }
  ]
}
```

实际传给 GEPA 的文本可以是上述 JSON 的压缩版，避免 prompt 过长。

## 7. Candidate Selection

### 7.1 GEPA 内部接受

每轮 proposal 生成后，GEPA 在同一批 train minibatch 上比较：

```text
sum(score(proposal on minibatch)) > sum(score(parent on minibatch))
```

如果不提升，proposal 丢弃。

如果提升，则 proposal 会在 val set 上评估，并进入 candidate pool。

### 7.2 Parent 选择策略

第一版建议默认：

```text
candidate_selection_strategy = current_best
```

原因是流程更容易解释和复现：每轮从当前 val 平均分最高的 candidate 继续优化。

如果后续发现 graph/temporal 这类少数类的局部改进被过早放弃，可以改为：

```text
candidate_selection_strategy = pareto
```

`pareto` 会更倾向保留和继续演化那些在部分 val samples 上表现好的候选，但分析成本更高。

### 7.3 最终选择

预算结束后：

```text
selected = argmax(candidate val gepa_score)
```

最终报告比较：

```text
selected vs initial on val
selected vs initial on test
```

第一版不设置 hard guard。也就是说，不因为某个单项指标下降而直接丢弃 candidate；所有取舍先体现在 GEPA score 中。

## 8. 预算

第一版使用：

```text
batch_size = 12
```

一次成功 proposal 的主要成本大约是：

```text
parent on minibatch: 12
proposal on same minibatch: 12
accepted proposal on val: len(val)
```

因此一次 accepted iteration 约为：

```text
2 * batch_size + len(val)
```

如果 `val` 有 40 条，一次 accepted iteration 约为 64 次 metric call。为了得到足够候选，`max_metric_calls` 需要随 batch size 和 val size 增加。

建议第一版目标：

```text
accepted candidates: 10-20
batch_size: 12
max_metric_calls: 根据 val size 反推，至少覆盖 10 个 accepted iterations
```

如果候选数量明显不足，优先增加 `max_metric_calls`，不要盲目减小 val set。

## 9. 执行入口

第一版至少提供三个入口：

```text
data preparation
GEPA dry-run
GEPA full
```

### 9.1 数据准备

数据准备使用 template-only 专用 grouped split 脚本，不再复用会输出 `dev/calibration` 的旧数据脚本。

建议新增入口：

```text
scripts/prepare_memrouter_template_gepa_dataset.py
```

该脚本只输出 `train/val/test`：

- `train` 用于 GEPA reflection/minibatch。
- `val` 用于 GEPA candidate 评分和最终选择。
- `test` 只在 `selected` 固定后评估一次。

入口命令：

```bash
uv run python scripts/prepare_memrouter_template_gepa_dataset.py \
  --input benchmarks/locomo/data/locomo_e2e_route_labels.v2.jsonl \
  --output-dir benchmarks/locomo/data/gepa_template_only \
  --seed 42 \
  --prefix locomo_v2_template \
  --train-ratio 0.70 \
  --val-ratio 0.20 \
  --test-ratio 0.10
```

预期产物：

```text
benchmarks/locomo/data/gepa_template_only/
  locomo_v2_template.train.jsonl
  locomo_v2_template.val.jsonl
  locomo_v2_template.test.jsonl
  locomo_v2_template.manifest.json
  locomo_v2_template.data_report.md
```

实现要求：

- split 必须按 `sample_id` 分组，避免同一对话同时出现在 train/val/test。
- split 只包含 `train`、`val`、`test`，不得输出 `dev` 或 `calibration`。
- 默认比例建议为 `70/20/10`，并允许通过 CLI 覆盖。
- manifest 必须记录 input hash、seed、split 分布和 warnings。
- manifest 必须记录 split ratios，以及每个 split 的 backend/scenario/category 分布。
- data prep 不运行 GEPA，不生成 candidate。

### 9.2 GEPA Dry-Run

Dry-run 用于验证完整 template-only 优化链路能跑通，而不是追求最终分数。

建议新增入口：

```text
scripts/optimize_memrouter_template_gepa.py
```

Dry-run 命令：

```bash
uv run python scripts/optimize_memrouter_template_gepa.py \
  --profile dry-run \
  --config configs/memrouter_gepa.local.yaml \
  --train benchmarks/locomo/data/gepa_template_only/locomo_v2_template.train.jsonl \
  --val benchmarks/locomo/data/gepa_template_only/locomo_v2_template.val.jsonl \
  --test benchmarks/locomo/data/gepa_template_only/locomo_v2_template.test.jsonl \
  --template-dir echomem/templates_data \
  --matcher-config configs/memrouter_matcher.yaml \
  --runs-dir runs/memrouter_template_gepa \
  --batch-size 12 \
  --budget dry-run \
  --seed 13 \
  --dry-run-train-limit 48 \
  --dry-run-val-limit 24 \
  --dry-run-test-limit 24
```

Dry-run 行为：

- 固定 matcher 代码和 matcher config。
- 对 dry-run train subset 运行 `initial` routing，生成 sampling buckets。
- 对 dry-run val subset 初始化 `initial` candidate score。
- 使用 `batch_size=12` 跑少量 GEPA iterations。
- 允许只产生少量 accepted candidates。
- 每个 accepted candidate 跑 dry-run val。
- `selected` 固定后跑 dry-run test。
- 输出 validator report、candidate scores、optimization trace 和 `selected vs initial` 报告。

Dry-run 通过标准：

- 数据加载成功。
- sampling buckets 生成成功。
- GEPA 能收到 deterministic score + feedback。
- 至少尝试生成一个 proposal。
- 非法 proposal 能被 validator 拦截并反馈。
- 合法 proposal 能被 matcher 加载并完成 route eval。
- 最终能生成 `selected` 和 test report。

Dry-run 不要求：

- 分数必须提升。
- candidate 数量达到 full 规模。
- test 结论具备统计意义。

### 9.3 GEPA Full

Full 用于正式验证 template-only 优化是否提升 routing accuracy。

Full 命令：

```bash
uv run python scripts/optimize_memrouter_template_gepa.py \
  --profile full \
  --config configs/memrouter_gepa.local.yaml \
  --train benchmarks/locomo/data/gepa_template_only/locomo_v2_template.train.jsonl \
  --val benchmarks/locomo/data/gepa_template_only/locomo_v2_template.val.jsonl \
  --test benchmarks/locomo/data/gepa_template_only/locomo_v2_template.test.jsonl \
  --template-dir echomem/templates_data \
  --matcher-config configs/memrouter_matcher.yaml \
  --runs-dir runs/memrouter_template_gepa \
  --batch-size 12 \
  --budget full \
  --seed 13 \
  --max-metric-calls 1200 \
  --target-accepted-candidates 15 \
  --candidate-selection-strategy current_best
```

Full 行为：

- 使用完整 train/val/test split。
- 使用 `initial` 完整 train routing 生成 sampling buckets。
- 每轮从固定 buckets 中 stratified sample 12 条 train samples。
- 每轮 feedback 基于当前 `parent` 对 minibatch 的最新 routing 结果。
- GEPA score 显式包含 `template_accept_rate` 和 `fallback_control_score`。
- 每个 accepted proposal 跑完整 val。
- 预算结束后选择 val GEPA score 最高的 `selected`。
- `selected` 固定后才跑 test。

Full 产物必须支持复核：

- `initial` train/val/test eval。
- sampling buckets。
- 每轮 selected parent、minibatch ids、score、feedback 摘要。
- 每个 proposal 的 validator report。
- 每个 accepted candidate 的 materialized templates。
- 每个 accepted candidate 的 val metrics。
- final `selected vs initial` val/test report。

### 9.4 Runner 参数设计

`scripts/optimize_memrouter_template_gepa.py` 第一版建议支持：

| 参数 | 含义 |
|---|---|
| `--profile {dry-run,full}` | 运行 profile |
| `--config` | route eval config |
| `--train` | train labels JSONL |
| `--val` | val labels JSONL |
| `--test` | held-out test labels JSONL |
| `--template-dir` | initial templates |
| `--matcher-config` | 固定 matcher config |
| `--runs-dir` | run 输出目录 |
| `--batch-size` | GEPA reflection minibatch size，第一版默认 12 |
| `--budget` | 预设预算档位，例如 `dry-run`、`full` |
| `--max-metric-calls` | 覆盖预算档位中的 metric call 上限 |
| `--target-accepted-candidates` | 目标 accepted candidate 数，用于报告和 early warning |
| `--candidate-selection-strategy` | `current_best` 或 `pareto`，第一版默认 `current_best` |
| `--seed` | 随机种子 |
| `--dry-run-train-limit` | dry-run train subset 上限 |
| `--dry-run-val-limit` | dry-run val subset 上限 |
| `--dry-run-test-limit` | dry-run test subset 上限 |
| `--resume-run` | 可选，恢复已有 run |

`--budget` 的建议含义：

| budget | 用途 | 建议值 |
|---|---|---|
| `dry-run` | 链路验证 | `max_metric_calls` 约 150-250 |
| `full` | 正式优化 | `max_metric_calls` 约 1000-1500，按 val size 调整 |

`--max-metric-calls` 显式传入时覆盖 `--budget` 默认值。

## 10. 产物

推荐 run 目录：

```text
runs/memrouter_template_gepa/<run_id>/
  config_snapshot.yaml
  split_summary.json
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
    selected_vs_initial.md
```

最小 report 应包含：

- `initial` val/test metrics。
- `selected` val/test metrics。
- `selected` 相对 `initial` 的 delta。
- accepted/rejected proposal 数量。
- validator failure 数量和主要原因。
- 每个 candidate 的 val GEPA score。
- per-backend recall/precision。
- fallback rate。
- template accept rate。
- top confusion pairs。
- suspected overbroad templates。

## 11. 第一版实施清单

1. 新增 template-only runner 或在现有 runner 中新增 template-only mode。
2. 固定 matcher 代码和 matcher config。
3. 实现 GEPA candidate template bundle 输出 schema。
4. 实现最小 validator 和 invalid-candidate feedback。
5. 使用 `initial` 跑完整 train set，生成 sampling buckets。
6. 实现 batch size 12 的 stratified minibatch sampler。
7. 实现 routing-aware GEPA score。
8. 实现 deterministic feedback evaluator。
9. 每个 accepted proposal 跑 val，不做 per-candidate ablation。
10. 预算结束后选择 val GEPA score 最高的 `selected`。
11. 只在最终 `selected` 固定后跑 test。
12. 生成 `selected vs initial` 报告。

## 12. 后续可选升级

如果第一版证明 templates-only 有稳定收益，再考虑：

- 加入 matcher policy 优化。
- 改用 `pareto` candidate selection。
- 动态刷新 sampling buckets。
- 加入 per-candidate ablation。
- 增加 hard guard。
- 将 feedback 中的规则扩展为 backend-specific diagnosis。
- 将 `semantic_card` 接入 matcher scoring 后再开放给 GEPA。
