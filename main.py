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

# Import configuration before other modules
from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, 
    DATABASE_PASSWORD, LARAVEL_URL, USE_SSH_TUNNEL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("main")

# Set up database connection and SSH tunnel
from src.database import get_db, close_ssh_tunnel, test_connection, ssh_tunnel

# Import schemas and services
from src.schemas import ProjectAllocationRequest
from src.services.team_allocation import process_allocation_request, load_allocation_history, HISTORY_FILE

# Create the FastAPI app
app = FastAPI(title="Team Allocation API")

# Register the SSH tunnel close function to run when application exits
atexit.register(close_ssh_tunnel)

# Function to retry database operations
def with_db_retry(max_retries=3, retry_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    # Check if SSH tunnel is still alive when in production mode
                    if USE_SSH_TUNNEL and ssh_tunnel:
                        if not ssh_tunnel.check_is_alive():
                            logger.warning("SSH tunnel was down, restarted it")
                    
                    # Try the operation
                    return await func(*args, **kwargs)
                
                except OperationalError as e:
                    retries += 1
                    last_error = e
                    logger.warning(f"Database connection error (attempt {retries}/{max_retries}): {str(e)}")
                    
                    # Wait before retrying
                    time.sleep(retry_delay)
                    
                    # If this is our last retry, check if SSH tunnel needs restart
                    if retries == max_retries - 1 and USE_SSH_TUNNEL and ssh_tunnel:
                        logger.info("Attempting to restart SSH tunnel before final retry")
                        ssh_tunnel.stop()
                        time.sleep(1)
                        ssh_tunnel.start()
                
                except SQLAlchemyError as e:
                    # Other database errors we won't retry
                    logger.error(f"Database error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            
            # If we've exhausted retries, raise the last error
            logger.error(f"Failed after {max_retries} attempts: {str(last_error)}")
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        return wrapper
    return decorator

# Middleware to verify origin
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

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Perform startup tasks"""
    logger.info("Application starting up...")
    # Database connection is initialized in the database module
    logger.info(f"API configured with {'SSH tunnel' if USE_SSH_TUNNEL else 'direct'} database connection")

@app.on_event("shutdown")
async def shutdown_event():
    """Perform shutdown tasks"""
    logger.info("Application shutting down...")
    close_ssh_tunnel()  # Close SSH tunnel if it exists

# Add middleware for connection status and timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Team Allocation API", "timestamp": datetime.now().isoformat()}

# Database connection test endpoint using our database module
# Database connection test endpoint
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

# Run the application directly when this file is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)