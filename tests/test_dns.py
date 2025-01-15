import socket
from dotenv import load_dotenv
import os

DATABASE_HOST = os.getenv("DB_HOST")

try:
    ip = socket.gethostbyname(DATABASE_HOST)
    print(f"Resolved {DATABASE_HOST} to IP: {ip}")
except socket.gaierror as e:
    print(f"Hostname resolution failed: {e}")
