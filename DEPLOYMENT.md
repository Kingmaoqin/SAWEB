# Free Streamlit Cloud Deployment Plan

This document explains how to put the Survival Analysis platform online with zero
infrastructure cost. The stack relies on **Streamlit Community Cloud** (free for
public repos) and the **Hugging Face Inference API** (free tier) so that anyone
with the link can open the app without running a local LLM server.

## 1. Prepare the repository

1. Fork or clone this repository to your personal GitHub account.
2. Ensure the new `requirements.txt` is committed so Streamlit knows which
   Python dependencies to install.
3. Push the repo to GitHub. Streamlit Cloud only works with public repos in the
   free plan.

## 2. Create a free Hugging Face access token

1. Visit <https://huggingface.co/settings/tokens> and create a **read** token.
2. Copy the token value; we will inject it into Streamlit secrets so the app can
   call the hosted Llama 3 model through the Inference API.
3. (Optional) If you want to try a different hosted model, note its repository
   id (e.g., `mistralai/Mixtral-8x7B-Instruct`). The defaults live in
   `sa_agent.py` and can be overridden with environment variables later.

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io>, sign in with GitHub, and click
   **New app**.
2. Select your forked repository, pick the default branch, and set
   `main.py` as the entry point.
3. In the **Advanced settings → Secrets** panel, paste the following YAML and
   replace the token placeholder:

   ```toml
   HF_API_TOKEN = "hf_your_token_here"
   # (optional) override defaults for the hosted model
   HF_LLM_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
   HF_MAX_NEW_TOKENS = "512"
   HF_TEMPERATURE = "0.7"
   HF_TOP_P = "0.9"
   ```

4. Click **Deploy**. Streamlit will automatically install everything listed in
   `requirements.txt`, boot the app, and expose a public URL that you can share.

## 4. How it works

- `sa_agent.py` now prefers Hugging Face's free Inference API via the
  `langchain-huggingface` client. Whenever `HF_API_TOKEN` is set, the agent uses
  the hosted `meta-llama/Meta-Llama-3-8B-Instruct` chat model, so no local vLLM
  server or paid OpenAI key is required.
- If you later add other provider keys (Groq, Google, etc.) the agent still
  understands them, but the deployment works out-of-the-box with only the free
  Hugging Face token.

## 5. Sharing the public link

Once Streamlit finishes provisioning, you will receive an URL similar to
`https://your-handle-survival-analysis.streamlit.app`. Share that link with your
collaborators; they can immediately interact with the platform, upload data, and
chat with the agent through the free LLM backend.

