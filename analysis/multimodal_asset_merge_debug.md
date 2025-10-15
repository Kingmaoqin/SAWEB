# Multimodal Raw Asset Merge Debugging

## Context
- 用户反馈在“Process Raw Multimodal Assets”步骤里，图像特征单独查看时有数据，但合并后的表格中所有 `img_feat_*` 列都为空白。
- 上一次改动已经统一了纯数字 ID 的规范化；这次需要进一步排查为何图像列仍然全部为缺失值。

## 调查过程
- 通读 `pages_logic/run_models.py` 中原始资产处理流程，确认合并逻辑：
  - 读取表格、图像、传感器数据后均会调用 `canonicalize_series` 规范化 ID。
  - 合并前 `_prep_for_merge` 会删除 `duration/event` 并将其余列转为数值。
- 使用 `simulate_multimodal_data.py` 生成的样例数据验证：当图像 DataFrame 的 `id` 列与表格中的 ID 一致时，合并结果正常，`img_feat_*` 列含有数值。
- 根据用户截图推断：图像单表预览中 `id` 列显示为文件路径（例如 `images/PT_0001.png`），说明用户可能将 manifest 的 `image` 列作为 ID 列上传。
  - 现有 `canonicalize_identifier` 仅处理纯数字、浮点等情况，对路径/文件名保持原样。
  - 因此图像 DataFrame 的 ID 会包含路径与扩展名，而表格 ID 为 `PT_0001`，导致合并键不匹配，最终图像列全部为空。

## 结论
- 问题根源：ID 规范化未剥离资产路径/扩展名，导致跨模态 ID 不一致。

## 修复
- 在 `canonicalize_identifier` 中增加 `_strip_asset_like_tokens`，自动移除常见资产路径前缀与文件扩展名，然后再执行原有的数值/字符串规范化逻辑。
- 新增单元测试覆盖：
  - `images/PT_0001.png → PT_0001`
  - `sensor_sequences/0005.csv → 5`
  - 其它常见扩展名场景，确保回归。

## 后续建议
- 若未来支持其它资产类型，可在 `_ASSET_EXTENSIONS` 中追加扩展名。
- UI 层仍建议提示用户优先提供显式 ID 列，以减少模糊匹配需求。
- 新增 `analysis/multimodal_alignment_debugger.py` 工具脚本，可用于快速核对各模态的规范化 ID 与资产路径是否匹配，避免再次出现合并后整列缺失的问题。
