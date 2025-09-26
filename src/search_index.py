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
from .clients import get_index_client, sanitize_azure_name
from .config import (
    VECTOR_SEARCH_DIMENSIONS,
    VECTOR_SEARCH_PROFILE_NAME,
    HNSW_CONFIG_NAME,
    SEMANTIC_CONFIG_NAME,
    HNSW_PARAMETERS
)


def create_search_index(org_id):
    """
    Creates a new search index for a specific organization in Azure AI Search.
    
    Args:
        org_id (str): Organization identifier
    """
    index_client = get_index_client()
    safe_org_id = sanitize_azure_name(org_id)
    org_index_name = safe_org_id
    
    if org_index_name in index_client.list_index_names():
        print(f"Index '{org_index_name}' already exists for organization '{org_id}'. Skipping creation.")
        return

    print(f"Creating index '{org_index_name}' for organization '{org_id}'...")
    
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(name="filepath", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(name="organization_id", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(name="blob_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=VECTOR_SEARCH_DIMENSIONS,
            vector_search_profile_name=VECTOR_SEARCH_PROFILE_NAME,
        ),
    ]

    hnsw_config = HnswAlgorithmConfiguration(
        name=HNSW_CONFIG_NAME,
        parameters=HNSW_PARAMETERS
    )
    
    vector_search = VectorSearch(
        profiles=[{"name": VECTOR_SEARCH_PROFILE_NAME, "algorithm_configuration_name": HNSW_CONFIG_NAME}],
        algorithms=[hnsw_config],
    )
    
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )

    index = SearchIndex(
        name=org_index_name, 
        fields=fields, 
        vector_search=vector_search, 
        semantic_search=semantic_search
    )
    
    index_client.create_index(index)
    print(f"Index '{org_index_name}' created successfully for organization '{org_id}'.")


def delete_search_index(org_id):
    """
    Deletes the search index for a specific organization.
    
    Args:
        org_id (str): Organization identifier
    """
    index_client = get_index_client()
    safe_org_id = sanitize_azure_name(org_id)
    org_index_name = safe_org_id
    
    try:
        if org_index_name in index_client.list_index_names():
            index_client.delete_index(index=org_index_name)
            print(f"Index '{org_index_name}' deleted successfully for organization '{org_id}'.")
        else:
            print(f"Index '{org_index_name}' does not exist for organization '{org_id}'.")
    except Exception as e:
        print(f"Could not delete index for organization '{org_id}': {e}")


def list_indexes():
    """
    Lists all organization-specific indexes.
    
    Returns:
        list: List of organization index names
    """
    index_client = get_index_client()
    all_indexes = index_client.list_index_names()
    
    # Filter for organization-specific indexes (all indexes are now org-specific)
    return all_indexes