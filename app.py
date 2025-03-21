import tempfile
from pathlib import Path

import streamlit as st

from utils.text_utils import TextEncoder, VectorStore, process_pdf_folder


def main():
    st.title("PDF Search Engine")

    # Initialize session state for vector store and encoder
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.encoder = TextEncoder()

    # File uploader
    st.subheader("1. Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )

    # Process uploaded files
    if uploaded_files:
        if st.session_state.vector_store is None or st.button("Reload Index"):
            with st.spinner("Processing PDF files..."):
                # Create a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to temp directory
                    for uploaded_file in uploaded_files:
                        temp_path = Path(temp_dir) / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                    # Process all PDFs in the temp directory
                    texts = process_pdf_folder(temp_dir)

                    if not texts:
                        st.error("No text could be extracted from the PDFs!")
                        return

                    # Generate embeddings
                    embeddings = st.session_state.encoder.encode(texts)

                    # Initialize vector store
                    st.session_state.vector_store = VectorStore(
                        dimension=embeddings.shape[1]
                    )
                    st.session_state.vector_store.add_texts(texts, embeddings)

                    st.success(f"Successfully processed {len(texts)} text segments!")

    # Search interface
    if st.session_state.vector_store is not None:
        st.subheader("2. Search Documents")

        # Search input and parameters
        query = st.text_input("Enter your search query")
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)

        # Search button
        if query and st.button("Search"):
            with st.spinner("Searching..."):
                # Generate query embedding
                query_embedding = st.session_state.encoder.encode(query)

                # Get results
                results = st.session_state.vector_store.search(
                    query_embedding, k=num_results
                )

                # Display results
                st.subheader("Search Results")
                for i, (text, score) in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {score:.4f})"):
                        st.write(text)
    else:
        st.info("Please upload PDF files to start searching.")


if __name__ == "__main__":
    main()
