# Azure RAG Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using Azure Cognitive Search and Azure OpenAI.

## Features

- **Multi-format document support**: TXT, PDF, DOCX, CSV, PPTX, XLSX, XLS
- Document indexing with vector embeddings
- Vector similarity search using Azure Cognitive Search
- Context-aware responses using Azure OpenAI
- Improved text chunking with configurable size and overlap
- Proper error handling and logging
- Environment variable configuration

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**New Dependencies Added:**
- `python-docx`: For Microsoft Word document processing
- `pandas`: For CSV and Excel file processing
- `python-pptx`: For PowerPoint presentation processing
- `openpyxl`: For Excel file support

### 2. Configure Environment Variables

Create a `.env` file in the project root with your Azure service credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME=text-embedding-ada-002

# Azure Cognitive Search Configuration
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_INDEX_NAME=rag-index
AZURE_SEARCH_ADMIN_KEY=your_search_admin_key_here
```

### 3. Azure Services Setup

Ensure you have the following Azure services configured:

- **Azure OpenAI Service**: Deploy `gpt-35-turbo` for chat completions and `text-embedding-ada-002` for embeddings
- **Azure Cognitive Search**: Create a search service with vector search capabilities

## Usage

### Basic Usage

```python
python main.py
```

This will:
1. Create/update the search index with vector search configuration
2. Process and index all supported files in the `data/` directory
3. Start an interactive query session

### Custom Usage

```python
from main import create_search_index, index_file, query_rag

# Create the search index
create_search_index()

# Index your documents
index_file("your_document.txt")

# Query the system
answer = query_rag("Your question here")
print(answer)
```

## Key Functions

- `create_search_index()`: Creates or updates the search index with vector search
- `process_and_upload_files(directory_path)`: Processes all supported files in a directory
- `extract_text_from_file(filepath)`: Extracts text from various file formats
- `chunk_text(text, chunk_size=800, overlap=100)`: Splits text into overlapping chunks
- `perform_vector_search(query, k=3)`: Performs vector similarity search
- `get_rag_response(query, search_results)`: Generates RAG response using retrieved context

## Supported File Formats

| Format | Extension | Library Used | Notes |
|--------|-----------|--------------|-------|
| Text | `.txt` | Built-in | UTF-8 encoding |
| PDF | `.pdf` | PyPDF2 | Extracts text from all pages |
| Word | `.docx` | python-docx | Extracts paragraphs and text |
| CSV | `.csv` | pandas | Converts to text representation |
| PowerPoint | `.pptx` | python-pptx | Extracts text from slides |
| Excel | `.xlsx`, `.xls` | pandas/openpyxl | Converts to text representation |

## Notes

- The system uses 1536-dimensional embeddings (OpenAI standard)
- Default chunk size is 800 characters with 100-character overlap
- Vector search uses HNSW algorithm with cosine similarity
- All API calls include proper error handling and logging
- Multi-format processing automatically detects file types and applies appropriate extraction methods

