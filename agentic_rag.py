# import basics
import os
import json
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
# from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain import hub

from supabase.client import Client, create_client
from langchain_core.tools import tool

from google.oauth2 import service_account

# load environment variables
load_dotenv()  

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

credentials = None
credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if credentials_json:
    # Parse the JSON string
    service_account_info = json.loads(credentials_json)
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

# initiate embeddings model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",
    # project=project_id
    credentials=credentials
)

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (temperature = 0)
llm = ChatVertexAI(
    model="gemini-2.5-pro", 
    temperature=0,
    credentials=credentials
)

# fetch the prompt from the prompt hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "What is the mission of DPR. Provide sources for your answer with page numbers."})

# put the result on the screen
print(response["output"])