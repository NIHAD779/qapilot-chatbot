from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()


client = QdrantClient(
    url="https://0ed5ff2c-d055-4459-aa08-8ff72d9a4f29.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="UImA32LIcFUAtmobuAsSDKleHka1FG2riKCUYsU2ayK5zqGs7lYOpQ"
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