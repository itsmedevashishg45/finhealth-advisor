import streamlit as st
from agents.financial_agent import FinancialAgent
from data_ingestion.document_processor import DocumentProcessor
from rag_pipeline.vector_store import VectorStoreManager
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="FinHealth Advisor",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    if st.button("Initialize Agent"):
        with st.spinner("Initializing Financial Agent..."):
            try:
                st.session_state.agent = FinancialAgent()
                st.success("Agent initialized!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file
                file_path = f"data/raw/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process and add to vector store
                processor = DocumentProcessor()
                docs = processor.load_documents(file_path)

                vector_store = VectorStoreManager()
                num_chunks = vector_store.add_documents(docs)

                st.success(f"Processed {num_chunks} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")

# Main interface
st.title("💰 FinHealth Advisor")
st.subheader("AI-Powered Financial & Healthcare Advisory")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about finance..."):
    if not st.session_state.agent:
        st.warning("Please initialize the agent first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.divider()
st.caption(
    "Built with LangChain, Pinecone, and Streamlit | Multi-Agent RAG System"
)
