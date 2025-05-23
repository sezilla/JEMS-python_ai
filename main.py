import traceback
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session
import json
import os
from datetime import datetime
import logging
import atexit
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.sql import text
import time
from functools import wraps

from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, 
    DATABASE_PASSWORD, LARAVEL_URL, USE_SSH_TUNNEL
)
from src.services.task_allocation import allocate_tasks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("main")

from src.database import get_db, close_ssh_tunnel, test_connection, ssh_tunnel

from src.schemas import (
    TaskAllocationRequest,
    TeamAllocationRequest, 
    SpecialRequest,
    SpecialRequestResponse,
    TaskScheduleRequest,
    ScheduleResponse
    )
from src.services.team_allocation import allocate_team, sync_db_to_json, ALLOCATION_HISTORY
from src.services.special_request import generate_special_request, save_special_request, HISTORY_SPECIAL_REQUEST
from src.services.task_scheduler import create_schedule, sync_trello_tasks_to_json, SCHEDULE_HISTORY

app = FastAPI(title="Team Allocation API")

atexit.register(close_ssh_tunnel)

def with_db_retry(max_retries=3, retry_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    if USE_SSH_TUNNEL and ssh_tunnel:
                        if not ssh_tunnel.check_is_alive():
                            logger.warning("SSH tunnel was down, restarted it")
                    
                    return await func(*args, **kwargs)
                
                except OperationalError as e:
                    retries += 1
                    last_error = e
                    logger.warning(f"Database connection error (attempt {retries}/{max_retries}): {str(e)}")
                    
                    time.sleep(retry_delay)
                    
                    if retries == max_retries - 1 and USE_SSH_TUNNEL and ssh_tunnel:
                        logger.info("Attempting to restart SSH tunnel before final retry")
                        ssh_tunnel.stop()
                        time.sleep(1)
                        ssh_tunnel.start()
                
                except SQLAlchemyError as e:
                    logger.error(f"Database error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            
            logger.error(f"Failed after {max_retries} attempts: {str(last_error)}")
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        return wrapper
    return decorator

async def verify_origin(request: Request):
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

@app.on_event("startup")
async def startup_event():
    """Perform startup tasks"""
    logger.info("Application starting up...")
    logger.info(f"API configured with {'SSH tunnel' if USE_SSH_TUNNEL else 'direct'} database connection")

@app.on_event("shutdown")
async def shutdown_event():
    """Perform shutdown tasks"""
    logger.info("Application shutting down...")
    close_ssh_tunnel()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Team Allocation API", "timestamp": datetime.now().isoformat()}

@app.get("/test")
@with_db_retry(max_retries=3)
async def test_database_connection():
    """
    Test the database connection and return connection details.
    This endpoint is useful for diagnostic purposes.
    """
    logger.info("Testing database connection...")
    result = test_connection()
    
    if result["status"] == "error":
        logger.error(f"Database connection test failed: {result['error']}")
        raise HTTPException(
            status_code=503, 
            detail={
                "message": "Database connection failed",
                "details": result
            }
        )
    
    logger.info(f"Database connection test successful: {result['message']}")
    return {
        "message": "Database connection test",
        "connection": result
    }



# SERVICES

# TEAM ALLOCATION
@app.post("/allocate-teams", dependencies=[Depends(verify_origin)])
def allocate_teams(request: TeamAllocationRequest):
    logger.info("Received allocation request: %s", request.model_dump())

    try:
        sync_db_to_json()

        start_date_str = str(request.start)
        end_date_str = str(request.end)

        if end_date_str < start_date_str:
            logger.error("Invalid date range: Start %s, End %s", start_date_str, end_date_str)
            raise HTTPException(status_code=400, detail="End date cannot be before start date.")

        try:
            result = allocate_team(request)

            if not result.success:
                logger.error("Allocation failed")
                raise HTTPException(status_code=500, detail="Failed to allocate teams")

            sync_db_to_json()

            return result.model_dump()

        except ValueError as ve:
            logger.error("Invalid date format: %s", str(ve))
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Error during team allocation: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/project-history", dependencies=[Depends(verify_origin)])
def get_project_history():
    try:
        if os.path.exists(ALLOCATION_HISTORY):
            with open(ALLOCATION_HISTORY, "r") as file:
                history = json.load(file)
            return {"message": "Project history fetched successfully", "data": history}
        return {"message": "No project history available"}
    except Exception as e:
        logger.error("Error fetching project history: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# SPECIAL REQUESTS
@app.post("/special-request", dependencies=[Depends(verify_origin)])
def generate_special_request_endpoint(request: SpecialRequest) -> SpecialRequestResponse:
    try:
        special_request = generate_special_request(request)
        save_special_request(special_request)
        logger.info("Generated Special Request: %s", json.dumps(special_request.dict(), indent=4))
        return special_request
    except Exception as e:
        logger.error("Error generating special request: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate special request")
    
@app.get("/special-request-history", dependencies=[Depends(verify_origin)])
def get_special_request_history():
    try:
        if os.path.exists(HISTORY_SPECIAL_REQUEST):
            with open(HISTORY_SPECIAL_REQUEST, "r") as file:
                history = json.load(file)
            return {"message": "Special request history fetched successfully", "data": history}
        return {"message": "No special request history available"}
    except Exception as e:
        logger.error("Error fetching special request history: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    

# TASK SCHEDULER
@app.post("/generate-schedule", dependencies=[Depends(verify_origin)])
def generate_schedule_endpoint(request: TaskScheduleRequest):
    try:
        sync_trello_tasks_to_json()

        if not request.project_id:
            raise HTTPException(status_code=400, detail="Invalid request parameters")

        schedule = create_schedule(request)

        logger.info("Generated Schedule: %s", json.dumps(schedule.model_dump(), indent=4))
        return schedule

    except ValueError as ve:
        logger.error("JSON parsing error: %s", str(ve))
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

    except Exception as e:
        logger.error("Error generating schedule: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    
# TASK ASSIGNMENT to users
@app.post("/allocate-user-to-task")
async def allocate_user_to_task(data: TaskAllocationRequest):
    try:
        print(f"Received allocation for project_id: {data.project_id}")

        response = allocate_tasks(data)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error or "Failed to allocate tasks")
        
        response_dict = response.model_dump()
        
        return {
            "success": True,
            "message": "Users allocated successfully.",
            "allocation": response_dict["checklists"]  # Changed from checklist_id to checklists
        }

    except Exception as e:
        print(f"Error in allocation endpoint: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)