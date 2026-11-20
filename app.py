"""NexusRAG Interactive Web Dashboard powered by Streamlit."""

import streamlit as st
from nexusrag.pipeline import NexusEngine

st.set_page_config(
    page_title="NexusRAG | Enterprise Hybrid RAG",
    page_icon="⚡",
    layout="wide"
)

# Initialize engine in session state
if "engine" not in st.session_state:
    st.session_state.engine = NexusEngine()

engine: NexusEngine = st.session_state.engine

# Header
st.title("⚡ NexusRAG: Enterprise Multi-Agent Hybrid RAG Engine")
st.caption("Dense Semantic Vector + Okapi BM25 Lexical Search fused via Reciprocal Rank Fusion (RRF)")

# Sidebar
with st.sidebar:
    st.header("📂 Knowledge Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Documents (TXT, MD)",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uf in uploaded_files:
            text = uf.read().decode("utf-8", errors="ignore")
            chunks_num = engine.ingest_text(text, source_name=uf.name)
            st.success(f"Indexed **{uf.name}** ({chunks_num} chunks)")

    st.markdown("---")
    st.subheader("💡 Load Sample Knowledge Base")
    if st.button("Load Distributed Systems Guide"):
        sample_doc = """
        Distributed Consensus and Raft Protocol:
        Raft is a consensus algorithm designed as an alternative to Paxos. It is split into three relatively independent subproblems:
        1. Leader Election: A new leader must be chosen when an existing leader fails.
        2. Log Replication: The leader must accept log entries from clients and replicate them across the cluster, forcing other logs to agree with its own.
        3. Safety: If any server has applied a particular log entry to its state machine, then no other server may apply a different log entry for the same log index.
        
        Vector Databases and Hybrid Search:
        Modern RAG architectures suffer from semantic drift when searching exact technical keywords like error codes or identifiers.
        Combining dense embeddings with sparse lexical models like Okapi BM25 through Reciprocal Rank Fusion (RRF) resolves this contradiction.
        RRF scores documents based on reciprocal ranking: RRF = 1 / (k + rank).
        """
        engine.ingest_text(sample_doc, source_name="Raft_and_Hybrid_RAG_Guide.md")
        st.success("Loaded sample document!")

    st.markdown("---")
    stats = engine.get_stats()
    st.metric("Total Indexed Chunks", stats["total_chunks"])
    st.metric("Indexed Documents", stats["indexed_documents"])
    st.caption("Engine: Hybrid Okapi BM25 + Dense RRF")

# Main Query Area
query = st.text_input("Enter your research question:", placeholder="e.g. How does Raft handle leader election and log replication?")

if st.button("🔍 Execute Hybrid Retrieval & Synthesis", type="primary") and query:
    if not engine.all_chunks:
        st.warning("Knowledge base is currently empty. Please upload or load a sample document in the sidebar first.")
    else:
        with st.spinner("Executing Multi-Agent Retrieval & Corrective Evaluation..."):
            res = engine.query(query)

        # Metrics bar
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency", f"{res['latency_ms']} ms")
        c2.metric("Relevance", f"{res['relevance_score']*100:.0f}%")
        c3.metric("Candidate Chunks", res["chunks_count"])
        c4.metric("Synthesizer", res["model_used"])

        st.markdown("### 📝 Grounded Synthesis")
        st.markdown(res["answer"])

        if res["citations"]:
            st.markdown("#### 📚 Citations")
            for cit in res["citations"]:
                st.code(cit, language="text")

        # Diagnostics & Chunks Inspector
        with st.expander("🔬 Inspect Retrieved Chunks & RRF Scores", expanded=False):
            st.info(f"Agentic Evaluation: {res['evaluation_reasoning']}")
            for idx, item in enumerate(res["retrieved_chunks"], 1):
                st.markdown(f"**Chunk #{idx}** | Source: `{item['source']}` | **RRF Score: `{item['fused_score']}`**")
                st.caption(f"BM25 Score: {item['sparse_score']} | Dense Score: {item['dense_score']}")
                st.text_area(f"Content #{idx}", item["content"], height=100, key=f"c_{idx}")
