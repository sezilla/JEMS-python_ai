import pymysql
import os
from dotenv import load_dotenv

DATABASE_HOST = os.getenv("DB_HOST")
DATABASE_PORT = int(os.getenv("DB_PORT", "3306"))
DATABASE_NAME = os.getenv("DB_DATABASE")
DATABASE_USER = os.getenv("DB_USERNAME")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")

try:
    connection = pymysql.connect(
        host=DATABASE_HOST,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
        database=DATABASE_NAME,
        port=DATABASE_PORT
    )
    print("Connection successful!")
except pymysql.MySQLError as e:
    print(f"Connection failed: {e}")
finally:
    if 'connection' in locals() and connection.open:
        connection.close()
