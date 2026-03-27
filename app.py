#!/usr/bin/env python3
"""
app.py
------
Streamlit chat interface for querying Larry King Live transcripts.
Uses Pinecone for retrieval and Claude (Anthropic) for answering.
"""

import os
import anthropic
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_NAME = "larry-king-transcripts"
N_RESULTS  = 5
MODEL      = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a research assistant specializing in CNN's Larry King Live.
You have access to the entire archive of Larry King Live transcripts spanning many years.

When answering questions:
- Draw on the transcript excerpts provided to give accurate, specific answers
- Note the date and episode when referencing specific interviews
- Comment on tone, style, and recurring themes when relevant
- If asked about a topic, identify patterns across multiple episodes
- If the transcripts don't contain enough info to answer, say so honestly
- Be conversational but thorough
- Consider how the questions Larry asks point back to Larry's own beliefs or perspectives

Always cite which episode(s) your answer draws from."""

# ── Setup ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_index():
    api_key = st.secrets.get("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        st.error("No PINECONE_API_KEY found. Add it to your Streamlit secrets.")
        st.stop()
    pc = Pinecone(api_key=api_key)
    return pc.Index(INDEX_NAME)


@st.cache_resource
def get_claude():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("No ANTHROPIC_API_KEY found. Add it to your Streamlit secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, index, embedder, n: int = N_RESULTS) -> tuple[str, list[dict]]:
    """Embed query, search Pinecone, return context and sources."""
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=n, include_metadata=True)

    context_parts = []
    sources       = []

    for match in results["matches"]:
        meta = match["metadata"]
        text = meta.get("text", "")
        context_parts.append(
            f"[Episode: {meta.get('title','?')} | Date: {meta.get('date','?')}]\n{text}"
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
    st.caption("Ask anything about Larry King's interviews from all his years at CNN — topics, guests, tone, themes, and more.")

    embedder = get_embedder()
    index    = get_index()
    claude   = get_claude()

    with st.sidebar:
        st.header("About")
        st.write(
            "This agent searches 3,246 Larry King Live transcripts "
            "and uses Claude AI to answer your questions."
        )
        st.divider()
        st.header("Example questions")
        examples = [
            "Which episodes and interviews touched on the meaning of life?",
            "How did Larry relate to the legacy of children?",
            "What are some of the interviews talking about God and what does Larry ask?",
            "Was his approach interviewing women seemingly different than men?",
        ]
        for i, ex in enumerate(examples):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
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
            context, sources = retrieve(user_input, index, embedder)

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
