from dotenv import load_dotenv
import os
import logging
import platform
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

logger.info("Loading environment variables from .env file")
load_dotenv()

DB_CONNECTION = os.getenv("DB_CONNECTION", "local")
DATABASE_HOST = os.getenv("DB_HOST", "127.0.0.1")
DATABASE_PORT = int(os.getenv("DB_PORT", "3306"))
DATABASE_NAME = os.getenv("DB_DATABASE", "MultiProjectBasedEventManagement")
DATABASE_USER = os.getenv("DB_USERNAME", "root")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "")

logger.info(f"DB_CONNECTION: {DB_CONNECTION}")
logger.info(f"DATABASE_HOST: {DATABASE_HOST}")
logger.info(f"DATABASE_PORT: {DATABASE_PORT}")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"DATABASE_USER: {DATABASE_USER}")
logger.info(f"DATABASE_PASSWORD: {'*' * len(DATABASE_PASSWORD) if DATABASE_PASSWORD else 'Not set'}")

SSH_HOST = os.getenv("SSH_HOST")
SSH_USER = os.getenv("SSH_USERNAME")
SSH_PRIVATE_KEY_PATH = os.getenv("SSH_PRIVATE_KEY")

if SSH_PRIVATE_KEY_PATH:
    if platform.system() == "Windows":
        SSH_PRIVATE_KEY_PATH = SSH_PRIVATE_KEY_PATH.replace('/', '\\')
    else:
        SSH_PRIVATE_KEY_PATH = SSH_PRIVATE_KEY_PATH.replace('\\', '/')
    
    if not os.path.isfile(SSH_PRIVATE_KEY_PATH):
        logger.warning(f"SSH private key not found at path: {SSH_PRIVATE_KEY_PATH}")
        
        potential_paths = [
            SSH_PRIVATE_KEY_PATH,
            os.path.join(os.getcwd(), "id_rsa"),
            os.path.join(os.getcwd(), "secret", "id_rsa"),
            os.path.join(str(pathlib.Path.home()), ".ssh", "id_rsa"),
        ]
        
        for path in potential_paths:
            if os.path.isfile(path):
                logger.info(f"Found SSH key at alternate location: {path}")
                SSH_PRIVATE_KEY_PATH = path
                break

# API tokens
ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LARAVEL_URL = os.getenv("LARAVEL_HOST_URL", "http://127.0.0.1:8000/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

USE_SSH_TUNNEL = DB_CONNECTION == "production"

if USE_SSH_TUNNEL and (not SSH_PRIVATE_KEY_PATH or not os.path.isfile(SSH_PRIVATE_KEY_PATH)):
    logger.warning("SSH key not found. Automatically switching to direct database connection!")
    USE_SSH_TUNNEL = False
    
    if SSH_HOST != DATABASE_HOST:
        logger.warning(
            f"Using direct connection to database at {DATABASE_HOST} with production credentials. "
            "This may fail if the database server doesn't accept direct connections."
        )

logger.info(f"Final database connection mode: {'SSH tunnel' if USE_SSH_TUNNEL else 'direct'}")

if not USE_SSH_TUNNEL:
    DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    logger.info(f"Direct database connection configured to: {DATABASE_HOST}:{DATABASE_PORT}")