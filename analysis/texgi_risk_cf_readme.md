# TEXGI 风险评分与反事实解释实现说明

本文梳理平台中基于 TEXGI/TEXGISA 的风险评分（risk score）计算方式，以及专属的反事实解释（counterfactual explanations）生成路径，便于快速理解代码结构和数据流。

## 风险评分（Risk Score）
- **定义**：风险评分等于模型在验证集上的累计风险（cumulative hazard）。界面提示也强调“Risk score = cumulative hazard across intervals（区间累计的风险总和）”。
- **从生存曲线推导**：当模型返回 `Surv_Test`（随时间的生存概率 DataFrame）且未显式提供 `risk_scores` 时， `_attach_risk_summary` 会取最后一行生存概率 `S_T`，按 `1 - S_T` 转为每个样本的累计风险，并计算均值 `Mean Risk Score` 供 UI 使用。【F:pages_logic/run_models.py†L606-L622】
- **直传风险**：若模型结果中已附带 `risk_scores`，同一函数会直接对其求均值填充 `Mean Risk Score`。这一逻辑保证无论输入是生存曲线还是预先计算的风险向量，界面都有一致的风险摘要。【F:pages_logic/run_models.py†L618-L623】

## TEXGI 反事实解释生成路径
1. **入口与参数收集**（前端/逻辑层）
   - 在结果页 `_render_texgi_cf_block` 中，只有当 TEXGISA 训练结果包含 `cf_model_spec`（模型结构与权重路径）、`cf_features`（验证集特征 DataFrame）、`hazards`（区间风险矩阵）时，才启用 TEXGI 反事实模块。用户可指定病人索引、期望的生存延长时间、梯度步数、学习率、展示的特征数，以及“Lock features”列表用于冻结不可修改的变量。【F:pages_logic/run_models.py†L624-L692】

2. **目标累计风险计算**（算法层）
   - `generate_texgi_counterfactuals` 读取用户的期望生存延长 `desired_extension`，调用 `_target_cumhaz` 将其映射为目标累计风险：按时间窗数 `num_bins` 比例缩放，延长越多目标风险越低。【F:algorithm/CF.py†L22-L31】【F:algorithm/texgi_cf.py†L125-L160】

3. **约束准备与模型加载**
   - 函数将传入的特征转为 DataFrame，并按 `feature_stats` 中的最小/最大值生成逐特征的取值上下界；“Lock features” 会形成布尔掩码 `frozen_mask`，在优化过程中强制这些维度保持原值。【F:algorithm/texgi_cf.py†L125-L171】
   - `_load_model` 根据 `model_spec` 重新实例化 `MultiTaskModel` 并加载保存的权重，确保反事实搜索与训练时的 TEXGISA 模型一致。【F:algorithm/texgi_cf.py†L42-L66】

4. **逐病人梯度搜索**
   - `_optimize_single` 以选定病人的特征向量为起点，使用 Adam 迭代更新：损失由“超出目标累计风险的部分”与 L2 特征偏移正则组成；每步更新后都会重新计算模型输出的累计风险，并在达成目标（±2%）时提前终止。冻结特征在每步后被重置为原值，同时特征始终被裁剪到 cohort 上下界。【F:algorithm/texgi_cf.py†L68-L120】
   - `generate_texgi_counterfactuals` 对选定的每个病人执行上述优化，记录初始/目标/达成的累计风险，并按照特征改变量的绝对值排序，输出前 `top_k` 条可行建议，同时排除冻结或改变量为 0 的特征。【F:algorithm/texgi_cf.py†L160-L207】

5. **结果汇总与呈现**
   - 汇总表包含 `sample_id`、建议序号、特征名、当前值、建议值、改变量，以及当前/目标/达成的累计风险与估计的生存延长。界面展示表格并提供 CSV 下载，顶部提示目标与达成的累计风险以验证优化是否生效。【F:algorithm/texgi_cf.py†L168-L207】【F:pages_logic/run_models.py†L676-L705】

## 关键特性小结
- **模型一致性**：反事实搜索直接调用训练好的 TEXGISA `MultiTaskModel`，不使用简化的 hazard 缩放或近邻替代。
- **可控性**：用户可调节梯度步数、学习率、期望延长时间，并锁定任意特征，便于符合临床可行性。
- **风险透明**：全过程围绕累计风险（hazard 求和）设计，入口提示、目标设定、结果汇总均显式呈现当前/目标/达成的风险水平。
