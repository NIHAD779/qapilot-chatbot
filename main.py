from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uvicorn
from supabase import create_client, Client
from datetime import datetime, timezone
import uuid
from zoneinfo import ZoneInfo

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant client configuration
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


model = SentenceTransformer('all-MiniLM-L6-v2')


supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

class SearchRequest(BaseModel):
    user_query: str
    max_num_results: int = 3
    upper_threshold: float = 0.75
    lower_threshold: float = 0.50


class SearchResult(BaseModel):
    matched_query_id: int
    matched_query: str
    matched_intent: str
    answer: str
    score: float

class SearchResponse(BaseModel):
    request_id: str
    user_query: str
    match_found: bool
    talk_to_agent: bool
    answers: list[SearchResult]

class Point(BaseModel):
    id:int
    matching_query: str
    intent: str
class AddDataRequest(BaseModel):
    collection_name: str
    data: list[Point]

class AddPointsRequest(BaseModel):
    collection_name: str
    points: list[Point]

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Step 1: Get query vector from SentenceTransformer
        query_vector = model.encode(request.user_query)

        # Step 2: Search in Qdrant
        search_results = client.search(
            collection_name="qapilot-chatbot",
            query_vector=query_vector,
            limit=request.max_num_results
        )

        # Process top result
        top_result_score = search_results[0].score
        answers = []
        added_intents = set()

        if top_result_score > request.upper_threshold:
            answers.append(SearchResult(
                matched_query_id=str(search_results[0].id),
                matched_query=search_results[0].payload['matching_query'],
                matched_intent=search_results[0].payload['intent'],
                answer="answer string",
                score=search_results[0].score
            ))
            added_intents.add(search_results[0].payload['intent'])
            match_found = True
            talk_to_agent = False
        elif request.lower_threshold < top_result_score <= request.upper_threshold:
            for res in search_results:
                if request.lower_threshold < res.score <= request.upper_threshold:
                    if res.payload['intent'] not in added_intents:
                        answers.append(SearchResult(
                            matched_query_id=str(res.id),
                            matched_query=res.payload['matching_query'],
                            matched_intent=res.payload['intent'],
                            answer="answer string",
                            score=res.score
                        ))
                        added_intents.add(res.payload['intent'])
            match_found = bool(answers)
            talk_to_agent = False
        else:
            match_found = False
            talk_to_agent = True

        # Step 3: Create final response
        final_response = SearchResponse(
            request_id=request_id,
            user_query=request.user_query,
            match_found=match_found,
            talk_to_agent=talk_to_agent,
            answers=answers
        )

        # Step 4: Log data to Supabase
        qdrant_response = [result.model_dump() for result in search_results]  

        response = supabase.table("search_logs").insert({
            "request_id": request_id,
            "user_query": request.user_query,
            "max_num_results": request.max_num_results,
            "upper_threshold": request.upper_threshold,
            "lower_threshold": request.lower_threshold,
            "qdrant_response": qdrant_response,
            "final_response": final_response.model_dump(),
            "created_at": datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
        }).execute()

        return final_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_data", status_code=201)
async def add_data(request: AddDataRequest):
    try:
        client.upsert(
            collection_name=request.collection_name,
            points=[{
                "id":point.id,
                "vector":model.encode(point.matching_query),
                "payload":{
                    "matching_query": point.matching_query,
                    "intent": point.intent
                }
            } for point in request.data]
        )
        return {"message": "Data added successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)