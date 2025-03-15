import logging
import time
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from src.config import (
    DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, 
    DATABASE_USER, DATABASE_PASSWORD, USE_SSH_TUNNEL,
    SSH_HOST, SSH_USER, SSH_PRIVATE_KEY_PATH
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# Create SSH tunnel instance if in production mode
ssh_tunnel = None
local_port = 13306  # Use a different local port to avoid conflicts

# Global DATABASE_URL variable for reuse
DATABASE_URL = None
engine = None
SessionLocal = None

def initialize_connection(max_retries=3):
    """Initialize the database connection and SSH tunnel if needed"""
    global ssh_tunnel, DATABASE_URL, engine, SessionLocal
    
    if USE_SSH_TUNNEL:
        try:
            # Import here to prevent circular imports
            from src.utils.ssh_tunnel import SSHTunnel
            
            logger.info("Initializing SSH tunnel...")
            ssh_tunnel = SSHTunnel(
                ssh_host=SSH_HOST,
                ssh_user=SSH_USER,
                ssh_pkey=SSH_PRIVATE_KEY_PATH,
                remote_host=DATABASE_HOST,
                remote_port=DATABASE_PORT,
                local_port=local_port
            )
            
            # Start the SSH tunnel
            logger.info("Starting SSH tunnel...")
            ssh_tunnel.start()
            
            # Wait a moment for the tunnel to establish
            time.sleep(1)
            
            if ssh_tunnel.tunnel_is_up:
                logger.info(f"SSH tunnel established successfully on local port {local_port}")
                
                # When using SSH tunnel, we connect to localhost through the tunnel
                DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@127.0.0.1:{local_port}/{DATABASE_NAME}"
            else:
                logger.error("SSH tunnel failed to establish")
                
                # Fall back to direct connection as a last resort
                logger.warning("Falling back to direct connection as SSH tunnel failed")
                DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
        except Exception as e:
            logger.error(f"Error setting up SSH tunnel: {str(e)}")
            
            # Fall back to direct connection as a last resort
            logger.warning("Falling back to direct connection as SSH tunnel setup failed")
            DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    else:
        # Direct connection without SSH tunnel
        DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

    logger.info(f"Creating database engine with URL: {DATABASE_URL}")

    # Create the SQLAlchemy engine with optimized connection parameters
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=1800,  # 30 minutes
        max_overflow=3,
        pool_size=5,
        pool_timeout=10,
        connect_args={
            "connect_timeout": 5,
            "use_pure": True
        }
    )

    # Test the connection before proceeding
    retries = 0
    connected = False
    last_error = None
    
    while retries < max_retries and not connected:
        try:
            # Test the connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                connected = True
                logger.info("Database connection test successful")
        except OperationalError as e:
            retries += 1
            last_error = e
            logger.warning(f"Database connection test failed (attempt {retries}/{max_retries}): {str(e)}")
            time.sleep(1)
    
    if not connected:
        logger.error(f"Failed to connect to database after {max_retries} attempts: {last_error}")
        # Continue anyway, let the application decide whether to fail or retry later
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return DATABASE_URL

# Initialize the connection
try:
    DATABASE_URL = initialize_connection()
except Exception as e:
    logger.error(f"Failed to initialize database connection: {str(e)}")
    # Set up a minimal configuration to let the app start anyway
    DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Get a database session and handle cleanup properly.
    This function is used as a dependency in FastAPI endpoints.
    """
    global engine, SessionLocal
    
    # If the session factory doesn't exist yet, create it
    if SessionLocal is None:
        logger.warning("SessionLocal was None, creating new session factory")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = None
    try:
        db = SessionLocal()
        # Test connection before returning
        db.execute(text("SELECT 1")).fetchone()
        yield db
    except OperationalError as e:
        logger.error(f"Database session error: {str(e)}")
        # Try to reinitialize connection
        if db:
            db.close()
        logger.info("Attempting to reinitialize database connection")
        initialize_connection(max_retries=1)
        
        # Create a new session with the new engine
        if SessionLocal:
            db = SessionLocal()
            yield db
        else:
            raise
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if db:
            db.close()

def test_connection():
    """
    Test the database connection and return server version.
    """
    db = None
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT VERSION()")).fetchone()
        version = result[0] if result else "Unknown"
        return {
            "status": "success",
            "message": f"Successfully connected to MySQL server with user '{DATABASE_USER}'",
            "version": version,
            "using_ssh_tunnel": USE_SSH_TUNNEL and ssh_tunnel and ssh_tunnel.tunnel_is_up,
            "database": DATABASE_NAME
        }
    except Exception as e:
        logger.error(f"Database connection test error: {str(e)}")
        return {
            "status": "error",
            "error": f"Could not reach MySQL server: {str(e)}",
            "attempted_connection": {
                "user": DATABASE_USER,
                "database": DATABASE_NAME,
                "using_ssh_tunnel": USE_SSH_TUNNEL and ssh_tunnel and ssh_tunnel.tunnel_is_up
            }
        }
    finally:
        if db:
            db.close()

def close_ssh_tunnel():
    """
    Helper function to close the SSH tunnel when shutting down the application.
    This should be called during application shutdown.
    """
    if ssh_tunnel and ssh_tunnel.tunnel_is_up:
        logger.info("Closing SSH tunnel...")
        ssh_tunnel.stop()
        logger.info("SSH tunnel closed")