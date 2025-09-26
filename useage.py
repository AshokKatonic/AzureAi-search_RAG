from src.manage_org import create_org, delete_org, list_orgs, get_org_info
from src.document_processor import process_and_upload_org_files
from src.search import search_documents

# org_manager = OrganizationManager()

org_id = "company-abc"
result = create_org(org_id)
print(result)

process_and_upload_org_files(org_id, "/path/to/documents")

response = search_documents(org_id, "What are our company policies?")
print(response)

orgs = list_orgs()
print(f"Available organizations: {orgs}")