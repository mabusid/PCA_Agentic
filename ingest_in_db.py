# import basics
import time
import os
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiate supabase db
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# initiate embeddings model
# project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",
    # project=project_id
)

# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader("documents")

# split the documents in multiple chunks
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# store chunks in vector store
# vector_store = SupabaseVectorStore.from_documents(
#     docs,
#     embeddings,
#     client=supabase_client,
#     table_name="documents",
#     query_name="match_documents",
#     chunk_size=1000,
# )

# Split documents into small batches to avoid embedding quota limits
batch_size = 25  # Process only n documents at a time
batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
print(f"Processing in {len(batches)} batches of ~{batch_size} documents each")

# Create vector store instance
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
)

# Process each batch with delays to avoid quota limits
for i, batch in enumerate(batches):
    print(f"\n--- Processing batch {i+1}/{len(batches)} ({len(batch)} documents) ---")
    
    try:
        # Add documents to vector store (this will generate embeddings)
        vector_store.add_documents(batch)
        print(f"✅ Successfully processed batch {i+1}")
    except Exception as e:
        print(f"❌ Error processing batch {i+1}: {e}")
        print("Continuing with next batch...")
        continue
    
    # Wait between batches to avoid embedding quota limits
    if i < len(batches) - 1:  # Don't sleep after the last batch
        print("Waiting 60 seconds before next batch...")
        time.sleep(60)