from .clients import get_blob_service_client, sanitize_azure_name
from .search_index import create_search_index, delete_search_index, list_indexes
from .document_processor import list_org_blobs, delete_org_blob
from azure.core.exceptions import ResourceNotFoundError


def create_org(org_id):
    """
    Creates a new organization with blob container and search index.
    
    Args:
        org_id (str): Organization identifier
        
    Returns:
        dict: Status of the creation process
    """
    try:
        # Create search index
        create_search_index(org_id)
        
        # Create blob container (this happens automatically when first blob is uploaded)
        from .clients import get_container_client
        get_container_client(org_id)
        
        return {
            "status": "success",
            "message": f"Organization '{org_id}' created successfully",
            "org_id": org_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create organization '{org_id}': {str(e)}",
            "org_id": org_id
        }


def delete_org(org_id):
    """
    Deletes an organization and all its associated resources.
    
    Args:
        org_id (str): Organization identifier
        
    Returns:
        dict: Status of the deletion process
    """
    try:
        # Delete search index
        delete_search_index(org_id)
        
        # Delete blob container
        safe_org_id = sanitize_azure_name(org_id)
        container_name = safe_org_id
        
        blob_service_client = get_blob_service_client()
        try:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.delete_container()
            print(f"Deleted container '{container_name}' for organization '{org_id}'")
        except ResourceNotFoundError:
            print(f"Container '{container_name}' not found for organization '{org_id}'")
        
        return {
            "status": "success",
            "message": f"Organization '{org_id}' deleted successfully",
            "org_id": org_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete organization '{org_id}': {str(e)}",
            "org_id": org_id
        }


def list_orgs():
    """
    Lists all organizations based on existing containers and indexes.
    
    Returns:
        list: List of organization identifiers
    """
    blob_service_client = get_blob_service_client()
    
    # Get organizations from blob containers (all containers are now org-specific)
    containers = list(blob_service_client.list_containers())
    org_containers = [c.name for c in containers]
    
    # Get organizations from search indexes
    org_indexes = list_indexes()
    
    # Combine and deduplicate
    org_ids = set()
    
    # Extract org IDs from container names
    for container_name in org_containers:
        org_id = container_name
        org_ids.add(org_id)
    
    # Extract org IDs from index names
    for index_name in org_indexes:
        org_id = index_name
        org_ids.add(org_id)
    
    return list(org_ids)


def get_org_info(org_id):
    """
    Gets information about a specific organization.
    
    Args:
        org_id (str): Organization identifier
        
    Returns:
        dict: Organization information
    """
    try:
        # Get blob count
        blob_count = len(list_org_blobs(org_id))
        
        # Check if search index exists
        from .clients import get_index_client
        
        index_client = get_index_client()
        safe_org_id = sanitize_azure_name(org_id)
        org_index_name = safe_org_id
        index_exists = org_index_name in index_client.list_index_names()
        
        return {
            "org_id": org_id,
            "blob_count": blob_count,
            "search_index_exists": index_exists,
            "status": "active" if (blob_count > 0 or index_exists) else "inactive"
        }
    except Exception as e:
        return {
            "org_id": org_id,
            "status": "error",
            "message": str(e)
        }