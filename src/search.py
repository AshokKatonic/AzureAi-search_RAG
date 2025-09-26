from azure.search.documents.models import VectorizedQuery
from .clients import get_search_client, openai_client
from .embeddings import generate_embeddings
from .config import AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME


def perform_vector_search(org_id, query, k=3):
    """
    Performs a vector search on the organization's index for a given query.
    
    Args:
        org_id (str): Organization identifier
        query (str): The search query
        k (int): Number of results to return
    
    Returns:
        list: List of search results
    """
    org_search_client = get_search_client(org_id)
    
    vectorized_query = VectorizedQuery(
        vector=generate_embeddings(query),
        k_nearest_neighbors=k,
        fields="embedding"
    )

    results = org_search_client.search(
        search_text="",
        vector_queries=[vectorized_query],
        select=["content", "filepath", "organization_id", "blob_name"],
        filter=f"organization_id eq '{org_id}'"
    )
    
    return [result for result in results]


def get_rag_response(org_id, query, search_results):
    """
    Generates a response using the RAG pattern for organization-specific data.
    
    Args:
        org_id (str): Organization identifier
        query (str): The user's question
        search_results (list): List of retrieved documents
    
    Returns:
        str: The generated response
    """
    system_prompt = f"""
    You are an intelligent assistant for organization '{org_id}'. 
    You answer user questions based on the context provided from the organization's documents.
    If the information is not in the context, say that you cannot answer based on the available documents.
    Always be helpful and provide accurate information based on the organization's data.
    """

    context_parts = []
    for result in search_results:
        filepath = result.get('filepath', 'Unknown file')
        content = result.get('content', 'No content available')
        context_parts.append(f"From {filepath}:\n{content}")

    context = "\n\n".join(context_parts)

    user_prompt = f"Context from organization '{org_id}' documents:\n{context}\n\nQuestion:\n{query}"

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


def search_documents(org_id, query, k=3):
    """
    Performs a complete search and RAG response for organization documents.
    
    Args:
        org_id (str): Organization identifier
        query (str): The search query
        k (int): Number of results to return
    
    Returns:
        str: The generated RAG response
    """
    search_results = perform_vector_search(org_id, query, k)
    
    if not search_results:
        return f"I couldn't find any relevant information in organization '{org_id}' documents."
    
    return get_rag_response(org_id, query, search_results)