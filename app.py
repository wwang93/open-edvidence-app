import os
import json
import html
DEV_MODE = False
from typing import Dict, List, Tuple, Any

import streamlit as st

st.set_page_config(page_title="Open-Edvidence", layout="wide")
st.title("Open-Edvidence")
st.caption("Bridging AI reasoning and educational evidence—making research more accessible, transparent, and usable.")


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_documents(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("documents.json must be a JSON object/dict like {\"doc_001\": \"...\", ...}")

    # Ensure all values are strings
    cleaned = {str(k): str(v) for k, v in data.items()}
    return cleaned
    
@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Embeddings & retrieval
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def embed_corpus(model_name: str, doc_ids: List[str], texts: List[str]):
    # Cached by (model_name, doc_ids, texts)
    model = get_embedding_model(model_name)
    doc_embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return doc_embs


def retrieve_top_k(model_name: str, doc_ids: List[str], texts: List[str], doc_embs, query: str, k: int):
    import torch
    from sentence_transformers import util

    model = get_embedding_model(model_name)
    q_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, doc_embs)[0]

    k = min(k, len(doc_ids))
    top = torch.topk(sims, k=k)

    results = []
    for score, idx in zip(top.values, top.indices):
        i = int(idx)
        results.append((float(score.item()), doc_ids[i], texts[i]))
    return results


# ----------------------------
# Prompting (Anthropic)
# ----------------------------
def build_prompt(query: str, retrieved: List[Tuple[float, str, str]]) -> str:
    summaries = [{"score": s, "id": doc_id, "text": text} for (s, doc_id, text) in retrieved]
    payload = {"query": query, "research_summaries": summaries}

    return f"""
You are a helpful assistant. Answer the user's query using ONLY the provided research summaries.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return a single valid JSON object with this schema:
{{
  "answer": "Answer based ONLY on the provided research summaries.",
  "used_summaries": ["doc_001", "doc_014"]
}}

Rules:
- Output MUST be valid JSON only. No markdown. No code fences. No extra text.
- If the summaries are not helpful, set:
  answer: "I don't have enough evidence in the provided summaries to answer."
  used_summaries: []
- Keep the answer concise.
""".strip()


def call_anthropic(model: str, api_key: str, prompt: str, max_tokens: int):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    text = ""
    try:
        text = resp.content[0].text
    except Exception:
        text = str(resp)

    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "Failed to parse JSON from model output", "raw_text": cleaned}


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")

    #docs_path = st.text_input("documents.json path", value="documents.json")
    docs_path = st.text_input("Evidence corpus (JSON)",value="intervention_reports_chunks.json")
    embedding_model_name = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    top_k = st.slider("Top-K sources", 1, 10, 3)

    st.divider()
    st.subheader("LLM (Anthropic)")
    anthropic_model = st.text_input("Model name", value="claude-sonnet-4-5-20250929")
    max_tokens = st.slider("Max tokens", 200, 2000, 800, 100)

    st.caption("API Key from Secrets: ANTHROPIC_API_KEY")


# ----------------------------
# Main UI
# ----------------------------
col_left, col_right = st.columns([2, 1], gap="large")

docs_error = None
docs_map: Dict[str, str] = {}

try:
    docs_map = load_documents(docs_path)
except Exception as e:
    docs_error = str(e)

# Load metadata (optional; app should still work without it)
meta: Dict[str, dict] = {}
try:
    meta = load_metadata("intervention_reports_meta.json")
except Exception:
    meta = {}

with col_right:
    st.subheader("Data")
    if docs_error:
        st.error(f"Failed to load {docs_path}: {docs_error}")
        st.info("Make sure the JSON file is in the repo root, or update the path in the sidebar.")
    else:
        st.success(f"Loaded {len(docs_map)} chunks")
        st.caption(f"Metadata loaded: {len(meta)} entries" if meta else "Metadata not loaded (optional).")

        with st.expander("Preview", expanded=False):
            for doc_id in list(docs_map.keys())[:3]:
                st.markdown(f"**{doc_id}**")
                t = docs_map[doc_id]
                st.write(t[:400] + ("..." if len(t) > 400 else ""))

with col_left:
    st.subheader("Ask a question")
    query = st.text_area(
        "Query",
        value="What does the research evidence say about the effectiveness of educational interventions on student achievement?",
        height=120
    )
    run = st.button(
        "Search and Summarize",
        type="primary",
        disabled=bool(docs_error) or not query.strip()
    )

    if run:
        # --- API key ---
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        if not api_key:
            st.error("Missing ANTHROPIC_API_KEY. Please set it in Streamlit Secrets (or env vars).")
            st.stop()

        # --- Retrieval ---
        doc_ids = list(docs_map.keys())
        texts = [docs_map[i] for i in doc_ids]

        with st.spinner("Searching for relevant evidence..."):
            doc_embs = embed_corpus(embedding_model_name, doc_ids, texts)
            retrieved = retrieve_top_k(
                embedding_model_name,
                doc_ids,
                texts,
                doc_embs,
                query=query,
                k=top_k
            )

        # --- LLM ---
        prompt = build_prompt(query, retrieved)

        with st.spinner("Generating an evidence-grounded answer..."):
            result = call_anthropic(anthropic_model, api_key, prompt, max_tokens)

        # ----------------------------
        # 1) Answer (LLM first)
        # ----------------------------
        answer = result.get("answer", "")
        used = result.get("used_summaries", []) or []

        st.markdown("## Answer")
        st.markdown(
            f"""
<div style="
    padding: 1rem 1.1rem;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    background: rgba(250,250,250,0.7);
">
  <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.4rem;">
    Evidence-grounded response
  </div>
  <div style="font-size: 1.05rem; line-height: 1.55;">
    {html.escape(answer) if isinstance(answer, str) else ""}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ----------------------------
        # 1b) Nicer citations using meta
        # ----------------------------
        if used:
            st.markdown("**Cited sources**")
            for k in used:
                m = meta.get(k, {})
                report_id = m.get("report_id", k)

                # chunk_index may be None in your meta; fall back to parsing from key
                chunk_index = m.get("chunk_index", None)
                if chunk_index is None and "__chunk" in k:
                    try:
                        chunk_index = int(k.split("__chunk")[-1])
                    except Exception:
                        chunk_index = "?"

                st.write(f"- **{report_id}** (chunk {chunk_index})")

        # ----------------------------
        # 2) Evidence (retrieved sources)
        # ----------------------------
        with st.expander("Evidence (retrieved sources)", expanded=False):
            for score, doc_id, text in retrieved:
                cited_badge = " ✅ cited" if doc_id in used else ""
                st.markdown(f"**{doc_id}** — score: `{score:.4f}`{cited_badge}")
                st.write(text)


        # ----------------------------
        # 3) FOR debugging (collapsed) DEV_MODE
        # ----------------------------
        if DEV_MODE:
            with st.expander("Response JSON (debug)", expanded=False):
                st.json(result)

            with st.expander("Prompt (debug)", expanded=False):
                st.code(prompt, language="json")
