import logging
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from .config import (
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    AZURE_OPENAI_API_VERSION,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_CONNECTION_STRING
)

logging.getLogger('azure').setLevel(logging.ERROR)


def sanitize_azure_name(name):
    """
    Sanitizes a name to be Azure-compatible for containers and indexes.
    
    Args:
        name (str): Original name
        
    Returns:
        str: Azure-compatible name
    """
    # Convert to lowercase
    sanitized = name.lower()
    # Replace invalid characters with dashes
    sanitized = sanitized.replace('_', '-')
    sanitized = sanitized.replace(' ', '-')
    # Remove any other invalid characters (keep only alphanumeric and dashes)
    sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)
    # Remove multiple consecutive dashes
    sanitized = re.sub(r'-+', '-', sanitized)
    # Remove leading/trailing dashes
    sanitized = sanitized.strip('-')
    # Ensure it doesn't exceed Azure limits (128 chars)
    if len(sanitized) > 128:
        sanitized = sanitized[:128].rstrip('-')
    
    return sanitized


openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


search_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, 
    credential=search_credential
)


search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, 
    index_name="default", 
    credential=search_credential
)


def get_blob_service_client():
    """
    Creates and returns a BlobServiceClient for Azure Blob Storage.
    
    Returns:
        BlobServiceClient: Configured blob service client
    """
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    elif AZURE_STORAGE_ACCOUNT_NAME:
        account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        return BlobServiceClient(account_url=account_url, credential=credential)
    else:
        raise ValueError("Either AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME must be set")



def get_container_client(org_id):
    """
    Gets or creates a container client for a specific organization.
    
    Args:
        org_id (str): Unique identifier for the organization
        
    Returns:
        ContainerClient: Container client for the organization's documents
    """
    blob_service_client = get_blob_service_client()
    safe_org_id = sanitize_azure_name(org_id)
    container_name = safe_org_id
    
    try:
        container_client = blob_service_client.create_container(container_name)
        print(f"Created container '{container_name}' for organization '{org_id}'")
    except ResourceExistsError:
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Using existing container '{container_name}' for organization '{org_id}'")
    
    return container_client



def get_search_client(org_id):
    """
    Gets a search client for a specific organization's index.
    
    Args:
        org_id (str): Unique identifier for the organization
        
    Returns:
        SearchClient: Search client for the organization's index
    """
    safe_org_id = sanitize_azure_name(org_id)
    org_index_name = safe_org_id
    return SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, 
        index_name=org_index_name, 
        credential=search_credential
    )



def get_index_client():
    """
    Gets the index client for managing organization-specific indexes.
    
    Returns:
        SearchIndexClient: Index client for managing indexes
    """
    return index_client