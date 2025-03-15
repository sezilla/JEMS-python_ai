from dotenv import load_dotenv
import os
import logging
import platform
import pathlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

# Load environment variables
logger.info("Loading environment variables from .env file")
load_dotenv()

# Database configuration
DB_CONNECTION = os.getenv("DB_CONNECTION", "local")
DATABASE_HOST = os.getenv("DB_HOST", "127.0.0.1")
DATABASE_PORT = int(os.getenv("DB_PORT", "3306"))
DATABASE_NAME = os.getenv("DB_DATABASE", "forge")
DATABASE_USER = os.getenv("DB_USERNAME", "forge")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "")

# Log the loaded values for debugging
logger.info(f"DB_CONNECTION: {DB_CONNECTION}")
logger.info(f"DATABASE_HOST: {DATABASE_HOST}")
logger.info(f"DATABASE_PORT: {DATABASE_PORT}")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"DATABASE_USER: {DATABASE_USER}")
logger.info(f"DATABASE_PASSWORD: {'*' * len(DATABASE_PASSWORD) if DATABASE_PASSWORD else 'Not set'}")

# SSH configuration
SSH_HOST = os.getenv("SSH_HOST")
SSH_USER = os.getenv("SSH_USERNAME")
SSH_PRIVATE_KEY_PATH = os.getenv("SSH_PRIVATE_KEY")

# Normalize SSH key path and add fallback options
if SSH_PRIVATE_KEY_PATH:
    # Convert to platform-appropriate path format
    if platform.system() == "Windows":
        SSH_PRIVATE_KEY_PATH = SSH_PRIVATE_KEY_PATH.replace('/', '\\')
    else:
        SSH_PRIVATE_KEY_PATH = SSH_PRIVATE_KEY_PATH.replace('\\', '/')
    
    # Try to find the key file
    if not os.path.isfile(SSH_PRIVATE_KEY_PATH):
        logger.warning(f"SSH private key not found at path: {SSH_PRIVATE_KEY_PATH}")
        
        # Try common fallback locations
        potential_paths = [
            # Original path
            SSH_PRIVATE_KEY_PATH,
            # Current directory
            os.path.join(os.getcwd(), "id_rsa"),
            # Relative to project root
            os.path.join(os.getcwd(), "secret", "id_rsa"),
            # Home directory
            os.path.join(str(pathlib.Path.home()), ".ssh", "id_rsa"),
        ]
        
        # Look for the key in potential locations
        for path in potential_paths:
            if os.path.isfile(path):
                logger.info(f"Found SSH key at alternate location: {path}")
                SSH_PRIVATE_KEY_PATH = path
                break

# API tokens
ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LARAVEL_URL = os.getenv("LARAVEL_HOST_URL", "http://127.0.0.1:8000/")

# Determine database connection type with auto-fallback
USE_SSH_TUNNEL = DB_CONNECTION == "production"

# Auto-fallback: Disable SSH tunnel if key not found
if USE_SSH_TUNNEL and (not SSH_PRIVATE_KEY_PATH or not os.path.isfile(SSH_PRIVATE_KEY_PATH)):
    logger.warning("SSH key not found. Automatically switching to direct database connection!")
    USE_SSH_TUNNEL = False
    
    # If we're switching to direct connection but still using remote credentials,
    # warn that this might not work unless the database allows direct access
    if SSH_HOST != DATABASE_HOST:
        logger.warning(
            f"Using direct connection to database at {DATABASE_HOST} with production credentials. "
            "This may fail if the database server doesn't accept direct connections."
        )

logger.info(f"Final database connection mode: {'SSH tunnel' if USE_SSH_TUNNEL else 'direct'}")

# Prepare database URL for direct connection
if not USE_SSH_TUNNEL:
    DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    logger.info(f"Direct database connection configured to: {DATABASE_HOST}:{DATABASE_PORT}")