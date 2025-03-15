from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import json
import os
from datetime import datetime
import logging
import mysql.connector
from sqlalchemy.exc import OperationalError
from src.database import get_db
from src.schemas import ProjectAllocationRequest
from src.config import DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, LARAVEL_URL

from src.services.team_allocation import process_allocation_request, load_allocation_history, HISTORY_FILE

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

def verify_origin(request: Request):
    allowed_origins = [
        LARAVEL_URL.rstrip("/"),
    ]
    origin = request.headers.get("origin") or request.headers.get("referer")

    if origin is None:
        return True  

    if not any(origin.startswith(allowed) for allowed in allowed_origins):
        logger.warning(f"Unauthorized request from origin: {origin}")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid request source")

    return True


@app.get("/", dependencies=[Depends(verify_origin)])
async def health_check():
    return {"status": "healthy", "message": "Hello World"}

@app.get("/test", dependencies=[Depends(verify_origin)])
async def network_test():
    try:
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

@app.post("/allocate-teams", dependencies=[Depends(verify_origin)])
def allocate_teams(request: ProjectAllocationRequest):
    logger.info("Received allocation request: %s", request.dict())

    try:
        start_dt = datetime.strptime(request.start, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end, "%Y-%m-%d")
        if end_dt < start_dt:
            logger.error("Invalid date range: Start %s, End %s", start_dt, end_dt)
            raise HTTPException(status_code=400, detail="End date cannot be before start date.")

        result = process_allocation_request({
            "project_id": request.project_id,
            "package_id": request.package_id,
            "start": request.start,
            "end": request.end,
        })

        if result["message"] == "failed":
            logger.error("Allocation failed: %s", result)
            raise HTTPException(status_code=500, detail=result["reason"])

        return result

    except ValueError as ve:
        logger.error("Invalid date format: %s", str(ve))
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    except Exception as e:
        logger.error("Error during team allocation: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/allocated-teams/{project_id}", dependencies=[Depends(verify_origin)])
def get_allocated_teams(project_id: int):
    history = load_allocation_history()
    allocated_teams = [entry for entry in history if entry["project_id"] == project_id]
    return {"success": True, "allocated_teams": allocated_teams}

@app.get("/project-history", dependencies=[Depends(verify_origin)])
def get_project_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as file:
                history = json.load(file)
            return {"message": "Project history fetched successfully", "data": history}
        return {"message": "No project history available"}
    except Exception as e:
        logger.error("Error fetching project history: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)
