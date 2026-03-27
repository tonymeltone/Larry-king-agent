#!/usr/bin/env python3
"""
app.py
------
Streamlit chat interface for querying Larry King Live transcripts.
Uses ChromaDB for retrieval and Claude (Anthropic) for answering.
Downloads the chroma index from Hugging Face on first load.
"""

import os
import anthropic
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download

# ── Configuration ─────────────────────────────────────────────────────────────

HF_REPO    = "alloriginaltone/larry-king-chroma"  # Hugging Face dataset
CHROMA_DIR = "/tmp/chroma_db"                      # temp folder on Streamlit Cloud
N_RESULTS  = 5
MODEL      = "claude-sonnet-4-20250514"

EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

SYSTEM_PROMPT = """You are a research assistant specializing in CNN's Larry King Live.
You have access to a large archive of Larry King Live transcripts spanning many years.

When answering questions:
- Draw on the transcript excerpts provided to give accurate, specific answers
- Note the date and episode when referencing specific interviews
- Comment on tone, style, and recurring themes when relevant
- If asked about a topic, identify patterns across multiple episodes
- If the transcripts don't contain enough info to answer, say so honestly
- Be conversational but thorough

Always cite which episode(s) your answer draws from."""

# ── Setup ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_collection():
    """Download chroma index from Hugging Face if needed, then load it."""
    db_file = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(db_file):
        st.info("Downloading transcript index... this may take a minute on first load.")
        os.makedirs(CHROMA_DIR, exist_ok=True)
        hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        hf_hub_download(
            repo_id=HF_REPO,
            filename="chroma.sqlite3",
            repo_type="dataset",
            token=hf_token,
            local_dir=CHROMA_DIR,
        )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(
        name="larry_king_transcripts",
        embedding_function=EMBEDDING_FN,
    )


@st.cache_resource
def get_claude():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("No ANTHROPIC_API_KEY found. Add it to your Streamlit secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, collection, n: int = N_RESULTS) -> tuple[str, list[dict]]:
    """Search the vector DB and return formatted context + source metadata."""
    results = collection.query(query_texts=[query], n_results=n)
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    context_parts = []
    sources       = []
    for doc, meta in zip(docs, metadatas):
        context_parts.append(
            f"[Episode: {meta.get('title','?')} | Date: {meta.get('date','?')}]\n{doc}"
        )
        sources.append(meta)
    return "\n\n---\n\n".join(context_parts), sources


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Larry King Live — Transcript Research Agent",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("🎙️ Larry King Live — Transcript Research Agent")
    st.caption("Ask anything about Larry King's interviews — topics, guests, tone, themes, and more.")

    collection = get_collection()
    claude     = get_claude()

    with st.sidebar:
        st.header("About")
        st.write(
            "This agent searches thousands of Larry King Live transcripts "
            "and uses Claude AI to answer your questions."
        )
        st.metric("Transcript chunks indexed", f"{collection.count():,}")
        st.divider()
        st.header("Example questions")
        examples = [
            "What did Larry ask guests about God?",
            "How did Larry interview politicians differently than celebrities?",
            "Which guests talked about 9/11 and what did they say?",
            "What was Larry's tone when interviewing controversial figures?",
            "Did Larry ever discuss the death penalty?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{examples.index(ex)}", use_container_width=True):
                st.session_state.pending_question = ex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])} transcript excerpts)"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s.get('title', 'Unknown')}** — {s.get('date', '?')}  \n"
                            f"[View transcript]({s.get('url', '#')})"
                        )

    pending    = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about the Larry King Live transcripts...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Searching transcripts..."):
            context, sources = retrieve(user_input, collection)

        user_prompt = (
            f"Here are relevant excerpts from the Larry King Live transcripts:\n\n"
            f"{context}\n\n---\n\nUser question: {user_input}"
        )

        with st.chat_message("assistant"):
            with st.spinner("Claude is thinking..."):
                response = claude.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer = response.content[0].text

            st.markdown(answer)

            with st.expander(f"📄 Sources ({len(sources)} transcript excerpts)"):
                seen = set()
                for s in sources:
                    key = s.get("filename", "")
                    if key in seen:
                        continue
                    seen.add(key)
                    st.markdown(
                        f"**{s.get('title', 'Unknown')}** — {s.get('date', '?')}  \n"
                        f"[View on CNN Transcripts]({s.get('url', '#')})"
                    )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
