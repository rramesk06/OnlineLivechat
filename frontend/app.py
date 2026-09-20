import os
import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


st.set_page_config(
    page_title="RAG Graph AI",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# API Helpers
# ============================================================

def api_get(path, timeout=30):
    return requests.get(
        f"{API_URL}{path}",
        timeout=timeout
    )


def api_post(path, timeout=300, **kwargs):
    return requests.post(
        f"{API_URL}{path}",
        timeout=timeout,
        **kwargs
    )


def get_error_message(response):
    """
    Safely extract an error message from FastAPI responses.
    Supports:
      {"detail": "..."}
      {"message": "..."}
      {"error": "..."}
      plain text responses
    """
    try:
        data = response.json()

        if isinstance(data, dict):
            return (
                data.get("detail")
                or data.get("message")
                or data.get("error")
                or response.text
            )

        return str(data)

    except Exception:
        return response.text or f"HTTP {response.status_code}"


# ============================================================
# Session State
# ============================================================

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Header
# ============================================================

st.title("📚 PDF RAG + Graph RAG AI Assistant")

st.caption(
    "Hybrid Vector Search + Neo4j Knowledge Graph + "
    "Web Fallback + Guardrails"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📄 Documents")

    # --------------------------------------------------------
    # API Health
    # --------------------------------------------------------

    try:
        health_response = api_get("/health")

        if health_response.ok:

            health = health_response.json()

            st.success("API connected")

            neo4j_status = health.get("neo4j", False)
            tavily_status = health.get("tavily", False)
            guardrails_status = health.get("guardrails", False)

            st.caption(
                f"Neo4j: {'Connected' if neo4j_status else 'Not connected'}"
            )

            st.caption(
                f"Tavily: {'Configured' if tavily_status else 'Not configured'}"
            )

            st.caption(
                f"Guardrails: "
                f"{'Enabled' if guardrails_status else 'Fallback filters'}"
            )

        else:
            st.error(
                f"API unhealthy: "
                f"{get_error_message(health_response)}"
            )

    except requests.RequestException as exc:

        st.error("Cannot connect to FastAPI")

        st.caption(str(exc))


    st.divider()


    # --------------------------------------------------------
    # Load Existing Documents
    # --------------------------------------------------------

    st.subheader("Indexed PDFs")

    docs = []

    try:

        response = api_get("/documents")

        if response.ok:

            data = response.json()

            docs = data.get("documents", [])

        else:

            st.warning(
                f"Could not load documents: "
                f"{get_error_message(response)}"
            )

    except requests.RequestException as exc:

        st.warning(
            f"Could not connect to document API: {exc}"
        )


    # --------------------------------------------------------
    # Existing Document Selection
    # --------------------------------------------------------

    if docs:

        document_options = {}

        for document in docs:

            document_id = document.get("document_id")

            filename = document.get(
                "filename",
                "Unknown PDF"
            )

            chunk_count = document.get(
                "chunk_count",
                0
            )

            label = (
                f"{filename} "
                f"({chunk_count} chunks)"
            )

            document_options[label] = document_id


        selected_label = st.selectbox(
            "Select indexed PDF",
            list(document_options.keys())
        )


        if st.button(
            "Use Selected PDF",
            use_container_width=True
        ):

            selected_document_id = (
                document_options[selected_label]
            )

            st.session_state.document_id = (
                selected_document_id
            )

            st.session_state.messages = []

            st.rerun()


    else:

        st.info(
            "No indexed PDFs found. "
            "Upload a PDF below."
        )


    st.divider()


    # --------------------------------------------------------
    # Upload PDF
    # --------------------------------------------------------

    st.subheader("➕ Add PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a PDF once. Duplicate PDFs are detected automatically."
    )


    if uploaded_file:

        st.caption(
            f"Selected: **{uploaded_file.name}**"
        )

        file_size_mb = (
            len(uploaded_file.getvalue())
            / (1024 * 1024)
        )

        st.caption(
            f"Size: {file_size_mb:.2f} MB"
        )


    if uploaded_file and st.button(
        "🚀 Index PDF",
        type="primary",
        use_container_width=True
    ):

        try:

            file_bytes = uploaded_file.getvalue()

            files = {
                "file": (
                    uploaded_file.name,
                    file_bytes,
                    "application/pdf"
                )
            }


            with st.spinner(
                "Processing PDF, creating embeddings "
                "and building knowledge graph..."
            ):

                response = api_post(
                    "/documents/upload",
                    files=files,
                    timeout=300
                )


            # ------------------------------------------------
            # Successful API Response
            # ------------------------------------------------

            if response.ok:

                result = response.json()

                document_id = result.get(
                    "document_id"
                )

                status = result.get(
                    "status"
                )


                # --------------------------------------------
                # Validate document ID
                # --------------------------------------------

                if not document_id:

                    st.error(
                        "Upload response did not contain "
                        "a document_id."
                    )

                    st.json(result)

                else:

                    # ----------------------------------------
                    # Save active document
                    # ----------------------------------------

                    st.session_state.document_id = (
                        document_id
                    )

                    st.session_state.messages = []


                    # ----------------------------------------
                    # Duplicate PDF
                    # ----------------------------------------

                    if status == "duplicate":

                        message = result.get(
                            "message",
                            "This PDF is already indexed."
                        )

                        st.info(
                            f"ℹ️ {message}"
                        )


                    # ----------------------------------------
                    # New PDF
                    # ----------------------------------------

                    else:

                        message = result.get(
                            "message",
                            "PDF indexed successfully."
                        )

                        st.success(
                            f"✅ {message}"
                        )


                    # ----------------------------------------
                    # Show indexing information
                    # ----------------------------------------

                    chunk_count = result.get(
                        "chunk_count"
                    )

                    if chunk_count is not None:

                        st.caption(
                            f"Chunks indexed: "
                            f"**{chunk_count}**"
                        )


                    st.rerun()


            # ------------------------------------------------
            # API Error
            # ------------------------------------------------

            else:

                error_message = get_error_message(
                    response
                )

                st.error(
                    f"Upload failed: {error_message}"
                )


        except requests.RequestException as exc:

            st.error(
                f"Could not connect to FastAPI: {exc}"
            )

        except Exception as exc:

            st.error(
                f"Unexpected upload error: {exc}"
            )


    # ========================================================
    # Active Document Information
    # ========================================================

    if st.session_state.document_id:

        try:

            document_id = (
                st.session_state.document_id
            )

            detail_response = api_get(
                f"/documents/{document_id}"
            )


            if detail_response.ok:

                document = (
                    detail_response.json()
                )

                st.divider()

                st.subheader("📌 Active PDF")

                filename = document.get(
                    "filename",
                    "Unknown"
                )

                chunk_count = document.get(
                    "chunk_count",
                    0
                )

                st.caption(
                    f"**{filename}**"
                )

                st.caption(
                    f"Chunks: **{chunk_count}**"
                )

                st.caption(
                    f"Document ID: `{document_id}`"
                )


        except requests.RequestException:

            pass


# ============================================================
# Main Content
# ============================================================

if not st.session_state.document_id:

    st.info(
        "👈 Select an indexed PDF or upload a new PDF "
        "from the sidebar to start chatting."
    )

    st.markdown(
        """
        ### What this application supports

        - 📄 PDF document ingestion
        - ✂️ Intelligent document chunking
        - 🔢 Sentence Transformer embeddings
        - 🔎 FAISS vector search
        - 🕸️ Neo4j Knowledge Graph
        - 🔀 Hybrid Vector + Graph retrieval
        - 🧠 LLM-based answer generation
        - 🌐 Tavily web fallback
        - 🛡️ Guardrails
        - 📊 RAG evaluation with DeepEval
        """
    )

    st.stop()


# ============================================================
# Chat History
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )


    with st.chat_message(role):

        st.markdown(content)


        source = message.get(
            "source"
        )


        if source:

            if source == "hybrid":

                st.caption(
                    "📚 Source: Hybrid RAG "
                    "(FAISS + Neo4j)"
                )

            elif source == "web":

                st.caption(
                    "🌐 Source: Web Search "
                    "(Tavily)"
                )

            elif source == "none":

                st.caption(
                    "⚠️ Source: No sufficient evidence"
                )

            else:

                st.caption(
                    f"Source: {source}"
                )


# ============================================================
# Chat Input
# ============================================================

question = st.chat_input(
    "Ask a question about the PDF..."
)


if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Call Chat API
    # --------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching FAISS + Neo4j and generating answer..."
            ):

                response = api_post(
                    f"/documents/"
                    f"{st.session_state.document_id}"
                    f"/chat",
                    json={
                        "question": question,
                        "top_k": 5
                    },
                    timeout=300
                )


            # =================================================
            # Successful Response
            # =================================================

            if response.ok:

                result = response.json()


                answer = result.get(
                    "answer",
                    "No answer was returned."
                )


                source = result.get(
                    "source",
                    "none"
                )


                # ---------------------------------------------
                # Answer
                # ---------------------------------------------

                st.markdown(answer)


                # ---------------------------------------------
                # Source Information
                # ---------------------------------------------

                if source == "hybrid":

                    st.success(
                        "📚 Answer generated from "
                        "document context using Hybrid RAG "
                        "(FAISS + Neo4j)."
                    )


                elif source == "web":

                    st.info(
                        "🌐 The retrieved PDF content was "
                        "not sufficiently relevant. "
                        "The answer used web search."
                    )


                elif source == "none":

                    st.warning(
                        "⚠️ No sufficient evidence was "
                        "available to answer this question."
                    )


                else:

                    st.caption(
                        f"Source: {source}"
                    )


                # =================================================
                # Retrieval Evidence
                # =================================================

                retrieval = result.get(
                    "retrieval",
                    []
                )


                with st.expander(
                    "🔎 Retrieval Evidence",
                    expanded=False
                ):

                    if not retrieval:

                        st.caption(
                            "No document passages were retrieved."
                        )

                    else:

                        st.caption(
                            f"Retrieved {len(retrieval)} "
                            "document passages."
                        )


                        for index, item in enumerate(
                            retrieval,
                            start=1
                        ):

                            page = item.get(
                                "page",
                                "?"
                            )

                            hybrid_score = item.get(
                                "hybrid_score",
                                0
                            )

                            vector_score = item.get(
                                "vector_score",
                                0
                            )

                            graph_score = item.get(
                                "graph_score",
                                0
                            )

                            st.markdown(
                                f"""
                                **{index}. Page {page}**

                                Hybrid Score: `{hybrid_score:.3f}`  
                                Vector Score: `{vector_score:.3f}`  
                                Graph Score: `{graph_score:.3f}`
                                """
                            )


                            text = item.get(
                                "text",
                                ""
                            )

                            if text:

                                st.write(text)


                            st.divider()


                # =================================================
                # Web Sources
                # =================================================

                if source == "web":

                    web_results = result.get(
                        "web_results",
                        []
                    )


                    with st.expander(
                        "🌐 Web Sources",
                        expanded=False
                    ):

                        if not web_results:

                            st.caption(
                                "No web sources were returned."
                            )

                        else:

                            for index, item in enumerate(
                                web_results,
                                start=1
                            ):

                                title = item.get(
                                    "title"
                                ) or item.get(
                                    "url"
                                ) or f"Web source {index}"


                                url = item.get(
                                    "url"
                                )


                                if url:

                                    st.markdown(
                                        f"{index}. "
                                        f"[{title}]({url})"
                                    )

                                else:

                                    st.write(
                                        f"{index}. {title}"
                                    )


                # =================================================
                # Save Assistant Message
                # =================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source": source
                    }
                )


            # =================================================
            # Chat API Error
            # =================================================

            else:

                error_message = get_error_message(
                    response
                )

                st.error(
                    f"Chat failed: {error_message}"
                )


    except requests.RequestException as exc:

        st.error(
            f"Could not connect to FastAPI: {exc}"
        )


    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )