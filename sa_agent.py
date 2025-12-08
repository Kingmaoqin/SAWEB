# sa_agent.py
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import os
import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import tool functions from your new sa_tools.py
from sa_tools import (
    run_survival_analysis,
    get_algorithm_explanation,
    compare_algorithms,
    explain_hyperparameter,
    get_data_summary,
    suggest_next_actions
)

# A lightweight fallback model so the app can still render when the HF token is
# missing or invalid. It simply echoes the last user message with guidance.
class OfflineFallbackChatModel(BaseChatModel):
    reason: str

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, reason: str):
        # BaseChatModel inherits from Pydantic's BaseModel and rejects unknown
        # attributes by default, so we pass `reason` through the parent
        # initializer and allow extras via Config.
        super().__init__(reason=reason)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        notice = (
            "⚠️ Running in offline mode because the Hugging Face endpoint could not be used. "
            "Please set a valid HF token in the environment or Streamlit secrets."
        )
        content = f"{notice}\nReason: {self.reason}\nLast user message: {last_user}"
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "offline-fallback"


# Function to select the LLM based on available API keys (from DRIVE project)
def get_llm():
    """Selects an appropriate LLM based on environment variables."""

    hf_token = None

    # Explicitly set the new Hugging Face router endpoint to avoid the deprecated
    # api-inference.huggingface.co URL (returns HTTP 410).
    os.environ.setdefault("HF_ENDPOINT", "https://router.huggingface.co")

    possible_keys = ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"]

    # Priority 1: Streamlit secrets
    for key in possible_keys:
        if key in st.secrets:
            hf_token = st.secrets[key]
            break

    # Priority 2: Tokens stored in the current Streamlit session (e.g., set from a UI input)
    if not hf_token and hasattr(st, "session_state"):
        for key in possible_keys:
            if st.session_state.get(key):
                hf_token = st.session_state[key]
                break

    # Priority 3: Environment variables
    if not hf_token:
        for key in possible_keys:
            if os.getenv(key):
                hf_token = os.getenv(key)
                break

    repo_id = os.getenv("HF_LLM_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
    max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    temperature = float(os.getenv("HF_TEMPERATURE", "0.7"))
    top_p = float(os.getenv("HF_TOP_P", "0.9"))

    if not hf_token:
        reason = "No Hugging Face token was provided."
        st.error("⚠️ Hugging Face Token not found. Falling back to offline mode.")
        return OfflineFallbackChatModel(reason)

    try:
        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            huggingfacehub_api_token=hf_token, # 显式传入 Token
            timeout=120,
        )

        # 关键修改：显式传入 model_id，避免 ChatHuggingFace 尝试去查询你的端点权限
        return ChatHuggingFace(llm=endpoint, model_id=repo_id)
    except Exception as exc:
        reason = f"Failed to initialize Hugging Face endpoint: {exc}"
        st.error("⚠️ Could not authenticate with Hugging Face. Falling back to offline mode.")
        return OfflineFallbackChatModel(reason)

    # Continue supporting other hosted providers in case a user explicitly sets
    # those credentials.
    if os.getenv("VLLM_ENDPOINT"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="local-model",
            openai_api_key="not-needed",
            openai_api_base=os.getenv("VLLM_ENDPOINT"),
            temperature=0.7,
            max_tokens=1024
        )
    elif os.getenv("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="llama3-8b-8192",
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=1024
        )
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, max_output_tokens=1024)
    else:
        raise ValueError(
            "No LLM API key found. Please set HF_API_TOKEN or another supported provider's key."
        )


# Define the Agent's state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define tools for the agent using the functions from sa_tools.py
@tool
def suggest_actions() -> str:
    """Checks app state and suggests what to do next. Call this first."""
    result = suggest_next_actions()
    return json.dumps(result)

@tool
def run_sa_model(
    algorithm_name: str,
    time_col: str,
    event_col: str,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    # Optional MySA/TEXGISA parameters forwarded to the underlying trainer when provided.
    lambda_smooth: float | None = None,
    lambda_expert: float | None = None,
    expert_rules_json: str = "",  # JSON string such as {"rules": [{"feature": "AGE", "min_mag": 0.01, ...}]}
    ig_steps: int | None = None,
    latent_dim: int | None = None,
    extreme_dim: int | None = None,
    gen_epochs: int | None = None,
    gen_batch: int | None = None,
    gen_lr: float | None = None,
    gen_alpha_dist: float | None = None,
    # Convenience toggle: preview-only runs skip expert penalties and reduce epochs for faster responses.
    preview_only: bool = False,
    preview_epochs: int = 50
) -> str:
    """
    Runs a survival model. algorithm_name must be one of {'CoxTime','DeepSurv','DeepHit','TEXGISA','MySA'}.
    Use preview_only=True to compute attributions quickly (lambda_expert=0, epochs=min(epochs, preview_epochs)).
    For expert priors, pass expert_rules_json as a JSON string.
    """
    extra: Dict[str, Any] = {}

    # Forward optional MySA parameters when they are provided.
    if lambda_smooth is not None: extra["lambda_smooth"] = float(lambda_smooth)
    if lambda_expert is not None: extra["lambda_expert"] = float(lambda_expert)
    if ig_steps is not None: extra["ig_steps"] = int(ig_steps)
    if latent_dim is not None: extra["latent_dim"] = int(latent_dim)
    if extreme_dim is not None: extra["extreme_dim"] = int(extreme_dim)
    if gen_epochs is not None: extra["gen_epochs"] = int(gen_epochs)
    if gen_batch is not None: extra["gen_batch"] = int(gen_batch)
    if gen_lr is not None: extra["gen_lr"] = float(gen_lr)
    if gen_alpha_dist is not None: extra["gen_alpha_dist"] = float(gen_alpha_dist)

    # Parse the expert rules JSON string if present.
    if expert_rules_json:
        try:
            extra["expert_rules"] = json.loads(expert_rules_json)
        except Exception:
            extra["expert_rules"] = {"rules": []}

    # Preview mode: disable expert penalties and cap the epoch count.
    if preview_only:
        extra["lambda_expert"] = 0.0
        epochs = min(int(epochs), int(preview_epochs))

    result = run_survival_analysis(
        algorithm_name, time_col, event_col, int(batch_size), int(epochs), float(lr),
        **extra
    )
    return json.dumps(result)

@tool
def preview_fi(algorithm_name: str, time_col: str, event_col: str,
               batch_size: int = 64, epochs: int = 100, lr: float = 0.01,
               preview_epochs: int = 50) -> str:
    """Quickly trains and computes attributions (no expert priors)."""
    result = run_survival_analysis(
        algorithm_name, time_col, event_col,
        int(batch_size), min(int(epochs), int(preview_epochs)), float(lr),
        lambda_expert=0.0
    )
    return json.dumps(result)


@tool
def explain_algorithm(algorithm_name: str) -> str:
    """Explains a specific survival analysis algorithm."""
    result = get_algorithm_explanation(algorithm_name)
    return json.dumps(result)

@tool
def compare_all_algorithms() -> str:
    """Provides a comparison table and recommendations for all algorithms."""
    result = compare_algorithms()
    return json.dumps(result)

@tool
def explain_param(param_name: str) -> str:
    """Explains a hyperparameter like 'learning_rate', 'batch_size', or 'epochs'."""
    result = explain_hyperparameter(param_name)
    return json.dumps(result)
    
@tool
def summarize_data() -> str:
    """Provides a summary of the currently loaded dataset, including columns and size."""
    result = get_data_summary()
    return json.dumps(result)

# List of all tools the agent can use
tools = [
    suggest_actions,
    run_sa_model,
    preview_fi,
    explain_algorithm,
    compare_all_algorithms,
    explain_param,
    summarize_data,
]


# System prompt to guide the LLM
SYSTEM_PROMPT = """You are an AI assistant that ONLY uses tools. You CANNOT answer any question from your own knowledge.
Your SOLE PURPOSE is to understand the user's request and call the correct tool.

**RULES:**
1.  **NEVER GUESS.** If you don't know something, or if the user asks about data, you MUST use a tool to find the answer.
2.  **TOOL-FIRST APPROACH:** To answer any question about the loaded data (like its name, columns, or summary) or to run an analysis, you MUST use one of the available tools.
3.  **DO NOT HALLUCINATE RESULTS:** Never invent analysis results like C-index scores. You can only report results that are returned from the `run_sa_model` tool.
4.  **NATURAL LANGUAGE:** Do not mention tool names in your response. Translate your action into natural language. For example, instead of "I will use the `summarize_data` tool," say, "Let me examine the data you've uploaded."
"""

def create_sa_agent():
    """Creates the conversational agent graph."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        """The primary node that calls the LLM."""
        messages = state["messages"]
        # Add system prompt to every call
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Determines the next step: call tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Define the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    # Compile and return the agent
    return workflow.compile()

# Create a single, importable instance of the agent
sa_agent = create_sa_agent()
