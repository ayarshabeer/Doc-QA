"""
Document QA System
-----------------
A Streamlit application that implements a Retrieval-Augmented Generation (RAG) system
for querying documents using Sentence Transformers and Pinecone.

Author: Shabeer Ayar
Email: shabeer.ayar@gmail.com
GitHub: https://github.com/ayarshabeer
LinkedIn: https://www.linkedin.com/in/ayarshabeer/
Created: 2024-12-14
Last Modified: 2024-12-14

This application allows users to:
- Upload multiple document types (PDF, DOCX, MD)
- Process and store document embeddings in Pinecone
- Query documents using natural language
- Retrieve relevant information with source context

Requirements:
- Python 3.11+
- Dependencies listed in requirements.txt

License: MIT
"""

import os
import tempfile
from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentenceTransformerEmbeddings(Embeddings):
    """Updated embeddings class that properly inherits from LangChain's Embeddings"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text, convert_to_tensor=True).tolist()


class DocumentProcessor:
    def __init__(
        self, pinecone_api_key, pinecone_environment, index_name, openai_api_key
    ):
        try:
            # Initialize Sentence Transformers embeddings
            self.embeddings = SentenceTransformerEmbeddings()

            # Initialize Pinecone
            pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
            index = pc.Index(index_name)
            self.vector_store = PineconeVectorStore(
                index=index, embedding=self.embeddings
            )
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50, length_function=len
            )

            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0,
                openai_api_key=openai_api_key,
            )

        except Exception as e:
            raise Exception(f"Error initializing components: {str(e)}")

    def process_file(self, file, temp_dir: str) -> int:
        """Process a single file and return number of chunks processed."""
        temp_path = os.path.join(temp_dir, file.name)

        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Load and process based on file type
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.name.endswith((".docx", ".doc")):
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif file.name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file.name}")

        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)

        # Process splits to match Pinecone's expected format
        texts = [doc.page_content for doc in splits]
        metadatas = [doc.metadata for doc in splits]

        # Add documents to vector store
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        return len(splits)

    def create_qa_chain(self, k: int = 4):
        """
        Creates a QA chain using the latest LangChain syntax

        Args:
            k: Number of documents to retrieve (default=4)
        """

        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, say so, but also explain what the context tells you that's relevant.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in a detailed way. Always try to use specific information from the context. 
        If you make any assumptions or inferences, explicitly state them."""
        # Create the prompt template
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        # Create the retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Create the final chain
        qa_chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | document_chain

        return qa_chain

    def query_documents(self, query: str, k: int = 4):
        """Query the document store and return relevant results."""
        try:
            # Create and execute the QA chain
            qa_chain = self.create_qa_chain(k=k)

            # Get the retriever to fetch source documents separately
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            source_docs = retriever.invoke(query)

            # Execute query
            answer = qa_chain.invoke(query)

            # Format sources for better readability
            sources = []
            for idx, doc in enumerate(source_docs, 1):
                sources.append({"content": doc.page_content, "metadata": doc.metadata})

            return {
                "answer": answer,  # The new chain returns the answer directly
                "source_documents": sources,
            }
        except Exception as e:
            raise Exception(f"Error in query_documents: {str(e)}")


def initialize_processor(
    openai_api_key, pinecone_api_key, pinecone_environment, index_name
):
    """Initialize the DocumentProcessor if it hasn't been initialized yet."""
    if "processor" not in st.session_state:
        with st.spinner("Loading models..."):
            try:
                processor = DocumentProcessor(
                    pinecone_api_key=pinecone_api_key,
                    pinecone_environment=pinecone_environment,
                    index_name=index_name,
                    openai_api_key=openai_api_key,
                )
                st.session_state.processor = processor
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error initializing the system: {str(e)}")
                return False
    return True


def process_uploaded_files(uploaded_files):
    """Process uploaded files if they haven't been processed before."""
    # Initialize processed files set in session state if it doesn't exist
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Get new files that haven't been processed yet
    new_files = [
        file
        for file in uploaded_files
        if f"{file.name}_{file.size}" not in st.session_state.processed_files
    ]

    if new_files:
        with st.spinner("Processing new documents..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    total_chunks = 0
                    progress_bar = st.progress(0)
                    for idx, file in enumerate(new_files):
                        chunks = st.session_state.processor.process_file(file, temp_dir)
                        total_chunks += chunks
                        # Add file to processed files set
                        st.session_state.processed_files.add(f"{file.name}_{file.size}")
                        st.success(f"Processed {file.name}: {chunks} chunks")
                        progress_bar.progress((idx + 1) / len(new_files))

                st.success(
                    f"Successfully processed {len(new_files)} new files with {total_chunks} total chunks!"
                )
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    else:
        if uploaded_files:
            st.info("All files have already been processed!")


def main():
    st.set_page_config(page_title="Document QA System", layout="wide")

    st.title("📚 Document QA System")
    st.markdown("*Using Sentence Transformers (all-MiniLM-L6-v2) for embeddings*")

    # Sidebar for API keys and configuration
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")
        pinecone_environment = st.selectbox(
            "Pinecone Region",
            ["us-west-2", "us-east-1", "us-central1", "eu-west1", "ap-southeast-1"],
        )
        index_name = st.text_input("Pinecone Index Name")

        # Add a clear button to reset processed files
        # if st.button("Clear Processed Files"):
        #     if "processed_files" in st.session_state:
        #         st.session_state.processed_files = set()
        #         st.success("Cleared processed files cache!")

        st.markdown("---")
        st.markdown(
            """
        ### Model Information
        - **Embedding Model**: all-MiniLM-L6-v2
        - **Vector Dimension**: 384
        - **Similarity Metric**: Cosine
        """
        )
        st.markdown("---")
        st.markdown(
            """
        ### Supported Documents
        - pdf - PDF
        - doc / docx - Word Document
        - md - MarkDown
        """
        )

    # Main interface
    if not all([openai_api_key, pinecone_api_key, pinecone_environment, index_name]):
        st.warning("Please fill in all configuration fields in the sidebar.")
        return

    # Initialize processor only once
    if not initialize_processor(
        openai_api_key, pinecone_api_key, pinecone_environment, index_name
    ):
        return

    # File upload section
    st.header("Upload Documents")

    # Keep track of uploaded files in session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["pdf", "doc", "docx", "md"],
    )

    # Update session state and process new files
    if uploaded_files:
        process_uploaded_files(uploaded_files)

    # Display currently processed files
    if "processed_files" in st.session_state and st.session_state.processed_files:
        with st.expander("View Processed Files"):
            for file_id in st.session_state.processed_files:
                filename = file_id.split("_")[0]  # Extract filename from the ID
                st.text(f"✓ {filename}")

    # Query section
    st.header("Ask Questions")
    query = st.text_input("Enter your question about the documents")
    k = st.slider(
        "Number of relevant documents to retrieve", min_value=1, max_value=10, value=4
    )

    if query:
        with st.spinner("Searching for answer..."):
            try:
                result = st.session_state.processor.query_documents(query, k=k)

                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander("View Source Documents"):
                    for idx, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Source {idx}:**")
                        st.markdown(doc["content"])
                        if doc["metadata"]:
                            st.markdown("*Metadata:*")
                            for key, value in doc["metadata"].items():
                                st.markdown(f"- {key}: {value}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
