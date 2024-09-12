from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

class Item(BaseModel):
    name: str
    Description: str | None = None


app = FastAPI()


@app.post("/test")
async def create_item(item: Dict):
    print(item)
    return f"Hello {item['name']}"


@app.get("/")
async def derp():
    return f"Hello butthead"

from pyngrok import ngrok
ngrok.set_auth_token("2iyevSimFN1Libp8js4qy5DALJo_75UiXh1p3PRoQ8z12no2R")

port = 8000
public_url = ngrok.connect(port).public_url

print(f"ngrok tunnel {public_url} -> http://127.0.0.1:{port}")