from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mysql.connector
import os
import logging
from urllib.parse import urlparse, parse_qs
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("database")

# Get environment variables
DB_URL = os.getenv("DB_URL")
PRIVATE_KEY_PATH = os.getenv("SSH_PRIVATE_KEY")

# Default database values (fallback)
DATABASE_HOST = os.getenv("DB_HOST", "127.0.0.1")
DATABASE_PORT = int(os.getenv("DB_PORT", "3306"))
DATABASE_NAME = os.getenv("DB_DATABASE", "forge")
DATABASE_USER = os.getenv("DB_USERNAME", "forge")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "")

SSH_HOST = None
SSH_USER = None

# Ensure PRIVATE_KEY_PATH is not None
if not PRIVATE_KEY_PATH:
    logger.error("SSH_PRIVATE_KEY environment variable is not set.")
    exit(1)

# Parse DB_URL if it exists
if DB_URL:
    try:
        logger.debug(f"Parsing DB_URL: {DB_URL}")
        parsed_url = urlparse(DB_URL)
        query_params = parse_qs(parsed_url.query)

        ssh_part, db_part = parsed_url.netloc.split("@")
        SSH_USER, SSH_HOST = ssh_part.split("@")
        DATABASE_HOST = db_part.split("/")[0]  # Extract MySQL host

        logger.info(f"Using SSH Tunnel: {SSH_USER}@{SSH_HOST} → {DATABASE_HOST}")
    except Exception as e:
        logger.error(f"Failed to parse DB_URL: {e}")
        exit(1)

# Function to start SSH tunnel
def create_ssh_tunnel():
    try:
        logger.debug(f"Starting SSH tunnel: {SSH_USER}@{SSH_HOST} with key {PRIVATE_KEY_PATH}")

        tunnel = SSHTunnelForwarder(
            (SSH_HOST, 22),
            ssh_username=SSH_USER,
            ssh_pkey=PRIVATE_KEY_PATH,
            remote_bind_address=(DATABASE_HOST, 3306),
            local_bind_address=("127.0.0.1", 3307)
        )
        tunnel.start()
        logger.info(f"SSH tunnel established: 127.0.0.1:3307 → {DATABASE_HOST}:3306")
        return tunnel
    except Exception as e:
        logger.error(f"SSH Tunnel Error: {e}")
        return None

# Start SSH tunnel
ssh_tunnel = create_ssh_tunnel()
if ssh_tunnel:
    DATABASE_HOST = "127.0.0.1"
    DATABASE_PORT = 3307
else:
    logger.error("SSH Tunnel failed. Cannot proceed with MySQL connection.")
    exit(1)

# Construct SQLAlchemy connection URL
DATABASE_URL = f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Test connection
def test_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            database=DATABASE_NAME,
            port=DATABASE_PORT,
            connection_timeout=5
        )
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        logger.info(f"Connected to MySQL version: {version}")
        cursor.close()
        return {"status": "success", "version": version}
    except mysql.connector.Error as e:
        logger.error(f"MySQL Error: {e}")
        return {"status": "failure", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return {"status": "failure", "error": str(e)}
    finally:
        if 'connection' in locals():
            connection.close()

# Clean up SSH tunnel on exit
import atexit
if ssh_tunnel:
    atexit.register(ssh_tunnel.stop)