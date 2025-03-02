# jems-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip freeze



alt:::
uvicorn main:app --host 127.0.0.1 --port 3000