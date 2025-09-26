import os


def ensure_data_directory():
    """
    Ensures the data directory exists and creates sample files if needed.
    
    Returns:
        str: Path to the data directory
    """
    data_directory = "data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created data directory: {data_directory}")
        
        sample_files = {
            "sample.txt": "Azure AI Search is a fully managed search-as-a-service. It provides a rich search experience to custom applications.\n\nRetrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM-generated responses. It grounds the model on external sources of knowledge to supplement the LLM's internal representation of information.",
            "sample_data.txt": "Sample Data for Multi-Format RAG Testing\n\nThis document contains sample data to test the multi-format file processing capabilities of the Azure RAG implementation.\n\nKey Features:\n- PDF text extraction\n- DOCX document processing\n- CSV data parsing\n- PPTX presentation content extraction\n- XLSX spreadsheet processing\n\nThe system now supports comprehensive document ingestion for enterprise knowledge bases.",
            "sample.csv": "Name,Age,Department,Salary\nJohn Doe,30,Engineering,75000\nJane Smith,28,Marketing,65000\nBob Johnson,35,Sales,70000\nAlice Brown,32,Engineering,80000\nCharlie Wilson,29,Marketing,60000"
        }
        
        for filename, content in sample_files.items():
            filepath = os.path.join(data_directory, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created sample file: {filename}")
    
    return data_directory


def get_file_info(filepath):
    """
    Gets information about a file.
    
    Args:
        filepath (str): Path to the file
    
    Returns:
        dict: File information including size, extension, etc.
    """
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    return {
        "filename": os.path.basename(filepath),
        "size": stat.st_size,
        "extension": os.path.splitext(filepath)[1].lower(),
        "modified": stat.st_mtime
    }


def format_file_size(size_bytes):
    """
    Formats file size in human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
    
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"
