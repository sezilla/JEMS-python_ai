import traceback
import json
import os
import time
import atexit
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any
import logging

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.sql import text

# Import configurations
from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, 
    DATABASE_PASSWORD, LARAVEL_URL, USE_SSH_TUNNEL
)

# Import services
from src.services.task_allocation import allocate_tasks
from src.services.team_allocation import allocate_team, sync_db_to_json, ALLOCATION_HISTORY
from src.services.special_request import generate_special_request, save_special_request, HISTORY_SPECIAL_REQUEST
from src.services.task_scheduler import create_schedule, sync_trello_tasks_to_json, SCHEDULE_HISTORY

# Import database utilities
from src.database import get_db, close_ssh_tunnel, test_connection, ssh_tunnel

# Import schemas
from src.schemas import (
    TaskAllocationRequest,
    TeamAllocationRequest, 
    SpecialRequest,
    SpecialRequestResponse,
    TaskScheduleRequest,
    ScheduleResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger("main")

# Initialize FastAPI app
app = FastAPI(
    title="Team Allocation API",
    description="API for team allocation, task scheduling, and special requests",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[LARAVEL_URL.rstrip("/"), "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Register cleanup function
atexit.register(close_ssh_tunnel)

# Enhanced database retry decorator
def with_db_retry(max_retries: int = 3, retry_delay: int = 1):
    """Decorator to retry database operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    # Check SSH tunnel health if enabled
                    if USE_SSH_TUNNEL and ssh_tunnel:
                        if not ssh_tunnel.check_is_alive():
                            logger.warning("SSH tunnel was down, attempting to restart...")
                            try:
                                ssh_tunnel.start()
                                time.sleep(2)  # Give tunnel time to establish
                            except Exception as tunnel_error:
                                logger.error(f"Failed to restart SSH tunnel: {tunnel_error}")
                    
                    return await func(*args, **kwargs)
                
                except OperationalError as e:
                    retries += 1
                    last_error = e
                    logger.warning(f"Database connection error (attempt {retries}/{max_retries}): {str(e)}")
                    
                    if retries < max_retries:
                        delay = retry_delay * (2 ** (retries - 1))  # Exponential backoff
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        
                        # Try to restart SSH tunnel on second-to-last attempt
                        if retries == max_retries - 1 and USE_SSH_TUNNEL and ssh_tunnel:
                            logger.info("Attempting to restart SSH tunnel before final retry")
                            try:
                                ssh_tunnel.stop()
                                time.sleep(1)
                                ssh_tunnel.start()
                                time.sleep(2)
                            except Exception as tunnel_error:
                                logger.error(f"Failed to restart SSH tunnel: {tunnel_error}")
                
                except SQLAlchemyError as e:
                    logger.error(f"Database error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
            
            logger.error(f"Failed after {max_retries} attempts: {str(last_error)}")
            raise HTTPException(status_code=503, detail="Database service unavailable after multiple retry attempts")
        
        return wrapper
    return decorator

# Enhanced origin verification
async def verify_origin(request: Request) -> bool:
    """Verify request origin for security"""
    try:
        allowed_origins = [
            LARAVEL_URL.rstrip("/"),
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
        
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        
        # Allow requests without origin (direct API calls, Postman, etc.)
        if not origin and not referer:
            logger.info("Request without origin/referer - allowing (likely API tool)")
            return True
        
        # Check origin
        if origin and any(origin.startswith(allowed) for allowed in allowed_origins):
            return True
            
        # Check referer as fallback
        if referer and any(referer.startswith(allowed) for allowed in allowed_origins):
            return True
        
        logger.warning(f"Unauthorized request from origin: {origin}, referer: {referer}")
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Invalid request source"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in origin verification: {str(e)}")
        # In case of error, allow the request but log it
        return True

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Middleware for request logging and timing
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request logging and timing"""
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming {request.method} request to {request.url}")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(f"Response {response.status_code} for {request.method} {request.url} - {process_time:.4f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error processing {request.method} {request.url} - {process_time:.4f}s: {str(e)}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Perform startup tasks"""
    logger.info("🚀 Team Allocation API starting up...")
    logger.info(f"Database connection: {'SSH tunnel' if USE_SSH_TUNNEL else 'direct'}")
    
    try:
        # Test database connection on startup
        connection_result = test_connection()
        if connection_result["status"] == "success":
            logger.info("✅ Database connection test successful")
        else:
            logger.error(f"❌ Database connection test failed: {connection_result}")
    except Exception as e:
        logger.error(f"❌ Error testing database connection on startup: {str(e)}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Perform shutdown tasks"""
    logger.info("🛑 Team Allocation API shutting down...")
    try:
        close_ssh_tunnel()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")

# Utility function for safe file operations
def safe_file_operation(file_path: str, operation: str = "read") -> Optional[Dict[str, Any]]:
    """Safely perform file operations with error handling"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
            
        if operation == "read":
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"File operation error for {file_path}: {str(e)}")
        return None

# HEALTH CHECK ENDPOINTS

@app.get("/", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "message": "Team Allocation API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with system status"""
    try:
        db_status = test_connection()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "running",
                "database": db_status["status"],
                "ssh_tunnel": "active" if USE_SSH_TUNNEL and ssh_tunnel and ssh_tunnel.check_is_alive() else "inactive"
            },
            "database": db_status
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/test", tags=["Health"])
@with_db_retry(max_retries=3)
async def test_database_connection():
    """Test database connection with retry logic"""
    logger.info("Testing database connection...")
    
    try:
        result = test_connection()
        
        if result["status"] == "error":
            logger.error(f"Database connection test failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=503, 
                detail={
                    "message": "Database connection failed",
                    "details": result,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        logger.info(f"Database connection test successful: {result.get('message', 'Success')}")
        return {
            "message": "Database connection test successful",
            "connection": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Database test failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# TEAM ALLOCATION ENDPOINTS

@app.post("/allocate-teams", tags=["Team Allocation"], dependencies=[Depends(verify_origin)])
async def allocate_teams(request: TeamAllocationRequest):
    """Allocate teams for projects"""
    logger.info(f"Received team allocation request: {request.model_dump()}")

    try:
        # Validate date range
        start_date_str = str(request.start)
        end_date_str = str(request.end)

        if end_date_str < start_date_str:
            logger.error(f"Invalid date range: Start {start_date_str}, End {end_date_str}")
            raise HTTPException(
                status_code=400, 
                detail="End date cannot be before start date"
            )

        # Sync database to JSON
        try:
            sync_db_to_json()
        except Exception as e:
            logger.warning(f"Failed to sync database to JSON: {str(e)}")

        # Perform allocation
        result = allocate_team(request)

        if not result.success:
            logger.error("Team allocation failed")
            raise HTTPException(
                status_code=500, 
                detail=result.error or "Failed to allocate teams"
            )

        # Sync back to database
        try:
            sync_db_to_json()
        except Exception as e:
            logger.warning(f"Failed to sync back to database: {str(e)}")

        logger.info("Team allocation completed successfully")
        return result.model_dump()

    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during team allocation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during team allocation: {str(e)}"
        )

@app.get("/project-history", tags=["Team Allocation"], dependencies=[Depends(verify_origin)])
async def get_project_history():
    """Get project allocation history"""
    try:
        history = safe_file_operation(ALLOCATION_HISTORY, "read")
        
        if history is not None:
            return {
                "success": True,
                "message": "Project history fetched successfully",
                "data": history,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "message": "No project history available",
            "data": [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching project history: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch project history: {str(e)}"
        )

# SPECIAL REQUEST ENDPOINTS

@app.post("/special-request", tags=["Special Requests"], dependencies=[Depends(verify_origin)])
async def generate_special_request_endpoint(request: SpecialRequest) -> SpecialRequestResponse:
    """Generate a special request"""
    try:
        logger.info(f"Generating special request: {request.model_dump()}")
        
        special_request = generate_special_request(request)
        
        # Save the special request
        try:
            save_special_request(special_request)
        except Exception as e:
            logger.warning(f"Failed to save special request: {str(e)}")
        
        logger.info("Special request generated successfully")
        return special_request
        
    except Exception as e:
        logger.error(f"Error generating special request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate special request: {str(e)}"
        )
    
@app.get("/special-request-history", tags=["Special Requests"], dependencies=[Depends(verify_origin)])
async def get_special_request_history():
    """Get special request history"""
    try:
        history = safe_file_operation(HISTORY_SPECIAL_REQUEST, "read")
        
        if history is not None:
            return {
                "success": True,
                "message": "Special request history fetched successfully",
                "data": history,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "message": "No special request history available",
            "data": [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching special request history: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch special request history: {str(e)}"
        )

# TASK SCHEDULER ENDPOINTS

@app.post("/generate-schedule", tags=["Task Scheduler"], dependencies=[Depends(verify_origin)])
async def generate_schedule_endpoint(request: TaskScheduleRequest):
    """Generate a task schedule"""
    try:
        logger.info(f"Generating schedule for project: {request.project_id}")
        
        # Validate request
        if not request.project_id:
            raise HTTPException(
                status_code=400, 
                detail="Project ID is required"
            )

        # Sync Trello tasks
        try:
            sync_trello_tasks_to_json()
        except Exception as e:
            logger.warning(f"Failed to sync Trello tasks: {str(e)}")

        # Generate schedule
        schedule = create_schedule(request)

        logger.info("Schedule generated successfully")
        return schedule

    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"JSON parsing error: {str(ve)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to parse AI response"
        )
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during schedule generation: {str(e)}"
        )

# TASK ALLOCATION ENDPOINTS

@app.post("/allocate-user-to-task", tags=["Task Allocation"])
async def allocate_user_to_task(data: TaskAllocationRequest):
    """Allocate users to specific tasks"""
    try:
        logger.info(f"Received task allocation for project_id: {data.project_id}")

        # Validate request
        if not data.project_id:
            raise HTTPException(
                status_code=400,
                detail="Project ID is required"
            )

        response = allocate_tasks(data)
        
        if not response.success:
            error_msg = response.error or "Failed to allocate tasks"
            logger.error(f"Task allocation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        response_dict = response.model_dump()
        
        logger.info("Users allocated to tasks successfully")
        return {
            "success": True,
            "message": "Users allocated successfully",
            "allocation": response_dict.get("checklists", []),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in task allocation endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during task allocation: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Team Allocation API server...")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=3000,
        reload=True,
        log_level="info",
        access_log=True
    )