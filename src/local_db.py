from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import mysql.connector
from src.config import (
    DATABASE_HOST,
    DATABASE_PORT,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_PASSWORD,
)
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Build the MySQL DATABASE_URL
DATABASE_URL = (
    f"mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def test_mysql_connection():
    """
    Function to test a direct connection to MySQL using mysql.connector.
    """
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

def get_db():
    """
    Dependency for database session handling.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Error with DB session: {e}")
        raise e
    finally:
        db.close()
