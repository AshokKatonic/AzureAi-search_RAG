import os
import PyPDF2
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
)
from azure.search.documents.models import VectorizableTextQuery

# --- 1. Load Environment Variables ---
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

# Local data directory
DATA_DIR = "data"

# --- 2. Initialize Clients ---
# Initialize the Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)

# Initialize the Azure AI Search clients
search_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=search_credential)
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=search_credential)

# --- 3. Indexing Functions ---

def create_search_index():
    """Creates a new search index if it doesn't already exist."""
    try:
        index_client.get_index(name=AZURE_SEARCH_INDEX_NAME)
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists.")
    except Exception:
        print(f"Creating index '{AZURE_SEARCH_INDEX_NAME}'...")
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="chunk", type=SearchFieldDataType.String),
            SearchField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=1536, vector_search_profile_name="my-hnsw-profile")
        ]
        # Configure HNSW algorithm with custom parameters
        hnsw_config = HnswVectorSearchAlgorithmConfiguration(
            name="my-hnsw-config",
            # Parameters for HNSW algorithm optimization
            parameters={
                "m": 4,  # Number of bi-directional links created for every new element (default: 4)
                "efConstruction": 400,  # Size of dynamic candidate list (default: 400)
                "efSearch": 500,  # Size of dynamic candidate list for search (default: 500)
                "metric": "cosine"  # Distance metric: "cosine", "euclidean", or "dotProduct"
            }
        )
        
        vector_search = VectorSearch(
            algorithms=[hnsw_config],
            profiles=[{"name": "my-hnsw-profile", "algorithm_configuration_name": "my-hnsw-config"}]
        )
        index = SearchIndex(name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
        index_client.create_or_update_index(index)
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")

def chunk_text(text, max_tokens=512, overlap=64):
    """Splits text into chunks of a specified size with overlap."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(encoding.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks

def process_and_index_files():
    """Processes local files, generates embeddings, and indexes them."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created a '{DATA_DIR}' directory. Please add your .txt or .pdf files there.")
        return

    print("--- Starting File Indexing ---")
    documents_to_upload = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath):
            print(f"Processing file: {filename}...")
            content = ""
            if filename.lower().endswith(".pdf"):
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        content += page.extract_text() or ""
            elif filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f"  - Skipping unsupported file type: {filename}")
                continue
            
            text_chunks = chunk_text(content)
            
            for i, chunk in enumerate(text_chunks):
                embedding = get_embedding(chunk, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
                document = {
                    "chunk_id": f"{filename}-{i}",
                    "title": filename,
                    "chunk": chunk,
                    "text_vector": embedding
                }
                documents_to_upload.append(document)

                if len(documents_to_upload) >= 100:
                    search_client.upload_documents(documents=documents_to_upload)
                    print(f"  - Uploaded a batch of {len(documents_to_upload)} documents.")
                    documents_to_upload = []

    if documents_to_upload:
        search_client.upload_documents(documents=documents_to_upload)
        print(f"  - Uploaded the final batch of {len(documents_to_upload)} documents.")

    print("--- File Indexing Complete ---\n")

# --- 4. RAG Query Functions ---

def get_embedding(text, model):
    """Generates an embedding for the given text."""
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

def search_documents(query_text):
    """Performs a vector search on the Azure AI Search index."""
    search_results = search_client.search(
        search_text=None,
        vector_queries=[
            VectorizableTextQuery(
                text=query_text,
                k_nearest_neighbors=3,
                fields="text_vector"
            )
        ],
        top=3
    )
    return search_results

def generate_response(user_question, context):
    """Generates a response using Azure OpenAI."""
    system_message = f"""
    You are an AI assistant. Answer the user's question based only on the provided context.
    If the context doesn't contain the answer, state that the information is not available in the documents.

    Context:
    {context}
    """
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question}
        ]
    )
    return response.choices[0].message.content

# --- 5. Main Application Flow ---

def main():
    """Main function to run the RAG pipeline."""
    # Step 1: Create the search index if it doesn't exist
    create_search_index()

    # Step 2: Process and index local files
    process_and_index_files()

    # Step 3: Run the query pipeline
    print("--- Starting RAG Query ---")
    user_question = "What is included in my Northwind Health Plus plan that is not in standard?"
    print(f"User Question: {user_question}\n")

    print("Searching for relevant documents...")
    search_results = search_documents(user_question)

    context = ""
    print("Search Results:")
    for result in search_results:
        print(f"  - Title: {result['title']}")
        print(f"    Chunk: {result['chunk'][:150]}...")
        context += result['chunk'] + "\n\n"
    
    if not context:
        print("\nNo relevant documents found. Cannot generate an answer.")
        return

    print("\nGenerating response...")
    answer = generate_response(user_question, context)
    print("\n--- Generated Answer ---")
    print(answer)
    print("------------------------\n")

if __name__ == '__main__':
    main()