from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
model = SentenceTransformer('all-MiniLM-L6-v2')

class SearchRequest(BaseModel):
    user_query: str

@app.post("/search")
async def search(request: SearchRequest):
    query_vector = model.encode(request.user_query)
    search_results = client.search(
        collection_name="qapilot-chatbot",
        query_vector=query_vector,
        limit=1
    )

    result = search_results[0].payload
    return result