from src.search_index import create_search_index, delete_search_index
from src.document_processor import process_and_upload_org_files
from src.search import search_documents
from src.manage_org import create_org, delete_org, list_orgs, get_org_info
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.utils import ensure_data_directory
import traceback


def main():
    """
    Main application function for organization-based RAG workflow.
    """
    print("Starting Organization-based RAG application...")

    # Initialize organization manager
    # org_manager = OrganizationManager()
    
    try:
        test_org_id = "org_33EBz4so6OdUfUM2dPOveY3WRnF"
        
        print(f"Setting up organization: {test_org_id}")
        
        result = create_org(test_org_id)
        print(f"Organization creation result: {result}")
        
        data_directory = ensure_data_directory()
        process_and_upload_org_files(
            test_org_id, 
            data_directory, 
            chunk_size=DEFAULT_CHUNK_SIZE, 
            overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        print(f"\nInteractive mode for organization '{test_org_id}'")
        print("Enter your questions about the organization's documents (or 'exit' to quit):")
        
        while True:
            user_query = input(f"[{test_org_id}] Enter your question: ")
            if user_query.lower() == 'exit':
                break
            
            try:
                print(f"Processing query for organization '{test_org_id}': '{user_query}'")
                response = search_documents(test_org_id, user_query)
                
                print("\nAnswer:")
                print(response)
                print("\n------------------\n")
                
            except Exception as e:
                print(f"Error processing query: {e}")
                traceback.print_exc()
                print("Please try again or check your configuration.\n")
            
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()