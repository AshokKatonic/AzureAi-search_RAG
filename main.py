"""
Azure RAG (Retrieval-Augmented Generation) Application

This module implements a complete RAG system using Azure AI Search and Azure OpenAI.
It processes documents, generates embeddings, performs vector search, and provides
intelligent responses based on retrieved context.

Author: Your Name
Date: 2024
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)
from openai import AzureOpenAI

load_dotenv()

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_API_VERSION = "2024-02-01"

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

search_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=search_credential)
search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=search_credential)


def create_search_index():
    """
    Creates a new search index in Azure AI Search if it doesn't already exist.
    
    The index is configured with:
    - Vector search using HNSW algorithm with cosine similarity
    - Semantic search capabilities
    - Fields for document content, filepath, and embeddings
    
    Fields:
        - id: Unique document identifier (key field)
        - content: Searchable text content (retrievable)
        - filepath: Source file path (filterable, retrievable)
        - embedding: Vector embeddings for similarity search (1536 dimensions)
    
    HNSW Parameters:
        - m: 4 (bi-directional links per element)
        - efConstruction: 400 (construction candidate list size)
        - efSearch: 500 (search candidate list size)
        - metric: cosine (distance metric)
    
    Returns:
        None
        
    Raises:
        Exception: If index creation fails
    """
    if AZURE_SEARCH_INDEX_NAME in index_client.list_index_names():
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists. Skipping creation.")
        return

    print(f"Creating index '{AZURE_SEARCH_INDEX_NAME}'...")
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(name="filepath", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-search-profile",
        ),
    ]

    hnsw_config = HnswAlgorithmConfiguration(
        name="my-hnsw-config",
        parameters={
            "m": 4,
            "efConstruction": 400,
            "efSearch": 500,
            "metric": "cosine"
        }
    )
    
    vector_search = VectorSearch(
        profiles=[{"name": "my-vector-search-profile", "algorithm_configuration_name": "my-hnsw-config"}],
        algorithms=[hnsw_config],
    )
    
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )

    index = SearchIndex(name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    index_client.create_index(index)
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")


def generate_embeddings(text):
    """
    Generates vector embeddings for the given text using Azure OpenAI's embedding model.
    
    Args:
        text (str): The input text to generate embeddings for
        
    Returns:
        list: A list of 1536 float values representing the text embedding
        
    Raises:
        Exception: If embedding generation fails
    """
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME
    )
    return response.data[0].embedding


def process_and_upload_files(directory_path):
    """
    Processes all text files in a given directory and uploads them to the search index.
    
    This function:
    1. Scans the directory for .txt files
    2. Reads each file's content
    3. Splits content into chunks by paragraphs (double newlines)
    4. Generates embeddings for each chunk
    5. Uploads documents to Azure AI Search
    
    Args:
        directory_path (str): Path to the directory containing text files
        
    Returns:
        None
        
    Note:
        - Only processes .txt files
        - Skips empty chunks
        - Document IDs are generated as filename_chunk_index (dots replaced with underscores)
    """
    print(f"Processing files in directory: {directory_path}")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            chunks = content.split('\n\n')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    document = {
                        "id": f"{filename.replace('.', '_')}_{i}",
                        "content": chunk,
                        "filepath": filename,
                        "embedding": generate_embeddings(chunk)
                    }
                    documents.append(document)
    
    if documents:
        search_client.upload_documents(documents=documents)
        print(f"Uploaded {len(documents)} documents to the index.")
    else:
        print("No new documents to upload.")


def perform_vector_search(query, k=3):
    """
    Performs a vector similarity search on the index for the given query.
    
    Args:
        query (str): The search query text
        k (int, optional): Number of nearest neighbors to retrieve. Defaults to 3.
        
    Returns:
        list: List of search results containing document information
        
    Note:
        - Converts query to embeddings before searching
        - Uses vector similarity search only (no text search)
        - Returns results ordered by similarity score
    """
    vectorized_query = VectorizedQuery(
        vector=generate_embeddings(query),
        k_nearest_neighbors=k,
        fields="embedding"
    )

    results = search_client.search(
        search_text="",
        vector_queries=[vectorized_query],
    )
    
    return [result for result in results]


def get_rag_response(query, search_results):
    """
    Generates a response using the RAG (Retrieval-Augmented Generation) pattern.
    
    This function:
    1. Combines search results into a context string
    2. Creates a system prompt for the AI assistant
    3. Formats the user query with context
    4. Calls Azure OpenAI chat completion model
    5. Returns the generated response
    
    Args:
        query (str): The user's question
        search_results (list): List of relevant documents from vector search
        
    Returns:
        str: The AI-generated response based on retrieved context
        
    Note:
        - Uses a conservative system prompt to avoid hallucination
        - Handles missing fields gracefully with default values
        - Limits response to 800 tokens
    """
    system_prompt = """
    You are an intelligent assistant. You answer user questions based on the context provided.
    If the information is not in the context, say that you cannot answer.
    """
    
    context_parts = []
    for result in search_results:
        filepath = result.get('filepath', 'Unknown file')
        content = result.get('content', 'No content available')
        context_parts.append(f"From {filepath}:\n{content}")

    context = "\n\n".join(context_parts)

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

    message_text = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
        messages=message_text,
        max_completion_tokens=800,
    )
    
    return completion.choices[0].message.content


def main():
    """
    Main application entry point that orchestrates the RAG pipeline.
    
    This function:
    1. Creates the search index if it doesn't exist
    2. Sets up the data directory with sample content
    3. Processes and uploads documents to the index
    4. Starts an interactive Q&A session
    
    The interactive session allows users to:
    - Ask questions about the indexed documents
    - Get AI-generated responses based on retrieved context
    - Exit by typing 'exit' or using Ctrl+C
    """
    create_search_index()
    
    data_directory = "data" 
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        with open(os.path.join(data_directory, "sample.txt"), "w") as f:
            f.write("Azure AI Search is a fully managed search-as-a-service. It provides a rich search experience to custom applications.\n\n")
            f.write("Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM-generated responses. It grounds the model on external sources of knowledge to supplement the LLM's internal representation of information.")
            
    process_and_upload_files(data_directory)
    
    print("\n--- Ready to answer questions ---")
    try:
        while True:
            user_query = input("Enter your question (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
            
            search_results = perform_vector_search(user_query)
            
            if not search_results:
                print("I couldn't find any relevant information in the documents.")
                continue

            response = get_rag_response(user_query, search_results)
            
            print("\nAnswer:")
            print(response)
            print("\n------------------\n")
            
    except KeyboardInterrupt:
        print("\nExiting application.")


if __name__ == "__main__":
    main()
