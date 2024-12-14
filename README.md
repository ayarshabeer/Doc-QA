
# Document QA System with RAG

This is a Streamlit app that lets you ask questions about your documents using RAG (Retrieval Augmented Generation).

## Setup

1. Clone the repo:
```bash
git clone https://github.com/yourusername/doc-qa
cd doc-qa
```

2. Install the requirements:
```bash
pip install -r requirements
```
Want a faster install? Try UV - a new Python package installer that's much faster than pip. Check out my guide on using UV here: [UV Making Python Package Management Fast and Simple](https://www.ayarshabeer.com/p/uv-making-python-package-management)

3. Set up Pinecone:
- Create a free account at [Pinecone](https://www.pinecone.io/)
- Create an index with:
  - Dimension: 384
  - Metric: cosine

## How to Use

1. Run the app:
```bash
streamlit run app.py
```

2. In the sidebar:
- Enter your OpenAI API key
- Enter your Pinecone API key
- Select your Pinecone region
- Enter your Pinecone index name

3. Upload documents:
- Supports PDF, Word (.doc/.docx), and Markdown files
- Click "Browse files" or drag and drop

4. Ask questions:
- Type your question in the text box
- Use the slider to adjust how many documents to check
- View the answer and source documents

That's it! You can now ask questions about your documents.

Note: Make sure your Pinecone index is properly set up with dimension 384 to match the all-MiniLM-L6-v2 embeddings model.
