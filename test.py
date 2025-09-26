# test_embedding.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

print(f"🔍 Environment variable: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME')}")

openai_client = AzureOpenAI(
    api_key="3lDUZwn4FGGz4UFp8xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    api_version="2024-02-01",
    azure_endpoint="https://xxxxx-openai-serxxxxxxxxxxxxxxx"
)

try:
    response = openai_client.embeddings.create(
        input=["test"],
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    )
    print("✅ Embedding model works!")
except Exception as e:

    print(f"❌ Error: {e}")
