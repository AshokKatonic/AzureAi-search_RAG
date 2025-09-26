from .clients import openai_client
from .config import AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME


def generate_embeddings(text):
    """
    Generates embeddings for a given text using Azure OpenAI's embedding model.
    
    Args:
        text (str): The text to generate embeddings for
    
    Returns:
        list: The embedding vector
    """
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME
    )
    return response.data[0].embedding