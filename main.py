from fastapi import FastAPI
import requests
import dotenv
from algorithm import find_approximate_needed_charger_locations_newcentroid, find_existing_chargers
from geojson_pydantic import Feature, FeatureCollection, Point
from typing import List

dotenv.load_dotenv()

app = FastAPI()

@app.get("/")
async def root() -> dict:
    return {"message": "hello world"}

@app.get("/existing-chargers")
async def existing_chargers() -> FeatureCollection:
    return find_existing_chargers(format_='geojson')

@app.get("/recommended-chargers")
async def recommended_chargers(quantity: int = None) -> List[Point]:
    return find_approximate_needed_charger_locations_newcentroid(quantity=quantity)