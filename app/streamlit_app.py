import streamlit as st

from config import CHUNK_STRATEGIES, EMBED_MODELS, RETRIEVER_LADDER, ExperimentConfig
from rag.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline(chunk, embed, retr):
    pipe = RAGPipeline(ExperimentConfig(chunk, embed, retr))
    pipe.load()
    return pipe


st.title("Chat with the papers")

col1, col2, col3 = st.columns(3)
chunk = col1.selectbox("Chunking", CHUNK_STRATEGIES)
embed = col2.selectbox("Embedder", list(EMBED_MODELS))
retr = col3.selectbox("Retriever", RETRIEVER_LADDER, index=2)

query = st.text_input("Ask a question about the papers")

if query:
    pipe = get_pipeline(chunk, embed, retr)
    with st.spinner("Retrieving and answering..."):
        out, sources = pipe.ask(query, k=5)
    st.markdown(out)
    with st.expander(f"Sources ({len(sources)})"):
        for i, c in enumerate(sources, 1):
            sec = f" §{c.section}" if c.section else ""
            st.markdown(f"**[{i}] {c.paper_id} p.{c.page}{sec}**")
            st.caption(c.text)
