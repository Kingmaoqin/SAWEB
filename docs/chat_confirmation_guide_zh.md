# Chat 页面确认执行流程说明

本文档用中文概述最近在“Chat with Survival Analysis Agent”页面增加的功能，以及在代码中的主要实现方式，方便后续维护和对接。

## 新增功能概览
- **显式确认再执行**：无论是快速按钮、表单提交还是聊天指令触发模型运行，都会先入队待确认动作，只有用户回复 `yes` 或 `no` 后才执行或取消。
- **聊天上下文提示**：在向代理调用时自动注入当前数据集状态和规则，提醒代理在运行训练/预览前务必询问确认，并指明可用算法及时间/事件列要求。
- **待确认提示与取消**：当存在挂起操作时，界面会显示警告，用户回复 `no` 会立即取消并记录到对话历史。
- **结果持久化展示**：直接运行完成后会缓存结果与原始数据，用于下方结果区域绘制生存曲线、特征重要性和 Kaplan–Meier。

## 关键实现节点
- **上下文构建**：`_context_string()` 根据 `DataManager` 提供的摘要拼接数据是否已加载、列名列表和算法指南，作为系统提示传递给 `sa_agent`，并强制要求“执行前询问 yes/no”。【F:pages_logic/chat_with_agent.py†L16-L55】
- **确认队列**：`_queue_confirmation()` 将待执行的函数与参数存入 `st.session_state["pending_action"]`，`_run_pending_if_confirmed()` 解析用户回复，遇到非 yes/no 会再次提示；回复 `yes` 时带 spinner 调用实际函数并保存结果，回复 `no` 则取消并写入历史。【F:pages_logic/chat_with_agent.py†L64-L105】
- **快速操作与表单**：四个快速按钮和右侧“Direct Run”表单提交时，都用 `_queue_confirmation` 包裹实际运行函数 `_run_direct`，并在对话区追加“即将执行，请回复 yes/no”的提示，保证界面触发也遵循确认流程。【F:pages_logic/chat_with_agent.py†L334-L409】【F:pages_logic/chat_with_agent.py†L438-L472】
- **聊天输入拦截**：在 `show()` 内处理 `st.chat_input` 时，若存在 `pending_action` 会优先调用 `_run_pending_if_confirmed`；否则根据文本匹配 preview/TEXGISA 的命令直接入队确认，其他自由文本则带上上下文调用代理。这样聊天触发的动作同样需要明确的 yes/no 才会运行。【F:pages_logic/chat_with_agent.py†L411-L456】
- **结果渲染与状态维护**：`_run_direct` 在运行模型后把结果和数据存入 session，以供 `_render_results` 绘制指标、TEXGISA 特征重要性、生存曲线和 KM 曲线，并在聊天历史中记录“Started: xxx”提示。【F:pages_logic/chat_with_agent.py†L288-L332】【F:pages_logic/chat_with_agent.py†L203-L287】

## 使用说明
1. 上传 CSV 后，界面会显示列名与行列数，快速按钮会使用 `duration/event` 或自动推测的列名生成默认配置。
2. 点击快速按钮或提交右侧表单后，聊天区会出现“即将执行，请回复 yes/no”的消息，用户必须回复 `yes` 才会真正运行；`no` 则取消。
3. 在聊天框输入 `run texgisa time=duration event=event` 等指令，也会先入队待确认；非命令类问题则交给代理回答。
4. 运行结束后，结果区域会显示关键指标、生存曲线与 KM 图等，可根据需要多次运行不同算法。
