from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.database import get_db
from src.schemas import ProjectAllocationRequest
import json
import os
from datetime import datetime
import socket
import requests
import mysql.connector
from sqlalchemy.exc import OperationalError
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD
import logging

from src.services.team_allocation import EventTeamAllocator, PROJECT_HISTORY_FILE
from src.services.spcl_request_classifier import classify_task, HISTORY_PATH
from src.services.task_scheduler import predict_categories, PredictRequest



logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
app = FastAPI()
allocator = EventTeamAllocator()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)


@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Hello World"}

@app.on_event("startup")
async def startup_event():
    logger.info(f"DB_HOST: {DATABASE_HOST}")
    logger.info(f"DB_PORT: {DATABASE_PORT}")
    logger.info(f"DB_NAME: {DATABASE_NAME}")
    logger.info(f"DB_USER: {DATABASE_USER}")

@app.get("/test")
async def network_test():
    try:
        # Attempting a database connection to test MySQL server
        connection = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            database=DATABASE_NAME,
            port=DATABASE_PORT,
            connection_timeout=5
        )
        connection.close()
        return {"message": f"Successfully connected to MySQL server at {DATABASE_HOST}:{DATABASE_PORT}"}
    except mysql.connector.Error as e:
        logger.error(f"MySQL Error: {e}")
        return {"error": f"Could not reach MySQL server: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}
    
@app.post("/allocate-teams")
def allocate_teams(request: ProjectAllocationRequest, db=Depends(get_db)):
    logger.info("Received allocation request: %s", request)
    
    # 🔥 Fix: Validate date range before calling `allocate_teams()`
    start_dt = datetime.strptime(request.start, '%Y-%m-%d')
    end_dt = datetime.strptime(request.end, '%Y-%m-%d')

    if end_dt < start_dt:
        return {"success": False, "error": "End date cannot be before the start date."}

    try:
        result = allocator.allocate_teams(
            db,
            request.project_name,
            request.package_id,
            request.start,
            request.end
        )
        logger.info("Allocation result: %s", result)
        return result  # ✅ Return correct result format
    except Exception as e:
        logger.error("Error during team allocation: %s", str(e))
        return {"success": False, "error": str(e)}


@app.get("/allocated-teams/{project_name}")
def get_allocated_teams(project_name: str):
    allocated_teams = allocator.allocated_teams.get(project_name, [])

    if not allocated_teams:
        return {"success": True, "allocated_teams": []}  # ✅ Always return a valid JSON structure

    return {"success": True, "allocated_teams": allocated_teams}

@app.get("/project-history")
def get_project_history():
    try:
        if os.path.exists(PROJECT_HISTORY_FILE):
            with open(PROJECT_HISTORY_FILE, "r") as file:
                history = json.load(file)
            return {"message": "Project history fetched successfully", "data": history}
        else:
            return {"message": "No project history available"}
    except Exception as e:
        logger.error("Error fetching project history: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error fetching project history: {str(e)}")

@app.post("/classify-task")
def classify_task_endpoint(request: dict):
    if "task" not in request:
        raise HTTPException(status_code=400, detail="Missing 'task' in request body")
    
    result = classify_task(request["task"])
    return result

@app.get("/classify-history")
def classify_history_endpoint():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as file:
            history = json.load(file)
        return history
    return []

@app.post("/predict_categories")
def predict_endpoint(request: PredictRequest):
    return predict_categories(request.project_name, request.start, request.end)









# ├── src
# │   ├── config.py
# │   ├── database.py
# │   ├── models.py
# │   ├── schemas.py
# │   ├── services
# │   │   ├── team_allocation.py
# ├── main.py
# ├── requirements.txt
