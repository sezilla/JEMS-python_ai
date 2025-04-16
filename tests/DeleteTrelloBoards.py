import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("TRELLO_API_KEY")
API_TOKEN = os.getenv("TRELLO_API_TOKEN")
ORGANIZATION_ID = os.getenv("TRELLO_ORGANIZATION_ID")

get_boards_url = f"https://api.trello.com/1/organizations/{ORGANIZATION_ID}/boards"
query = {
    'key': API_KEY,
    'token': API_TOKEN
}

response = requests.get(get_boards_url, params=query)

if response.status_code == 200:
    boards = response.json()
    for board in boards:
        board_id = board['id']
        board_name = board['name']
        print(f"Deleting board: {board_name} (ID: {board_id})")

        delete_board_url = f"https://api.trello.com/1/boards/{board_id}"
        delete_response = requests.delete(delete_board_url, params=query)

        if delete_response.status_code == 200:
            print(f"Successfully deleted board: {board_name}")
        else:
            print(f"Failed to delete board: {board_name}. Status code: {delete_response.status_code}")
else:
    print(f"Failed to fetch boards. Status code: {response.status_code}")
    print(response.text)
