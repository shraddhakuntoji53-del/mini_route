from fastapi import FastAPI
import requests, json

app = FastAPI()

# Convert place → coordinates
def geocode(place):
    url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&limit=1"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})
    return (float(r.json()[0]["lat"]),float(r.json()[0]["lon"])) if r.json() else None


@app.get("/routes")
def get_routes(start:str,end:str,alternatives:int=3):

    s = geocode(start)
    e = geocode(end)

    if not s or not e:
        return {"error":"location_not_found"}

    slat,slon = s; elat,elon = e

    url = f"http://router.project-osrm.org/route/v1/driving/{slon},{slat};{elon},{elat}?overview=full&alternatives=true&steps=true&geometries=geojson"
    res = requests.get(url).json()

    if "routes" not in res:
        return {"error":"no_route"}

    results=[]

    for r in res["routes"][:alternatives]:
        coords=[(c[1],c[0]) for c in r["geometry"]["coordinates"]]
        steps=[step["maneuver"]["instruction"] for leg in r["legs"] for step in leg["steps"]]

        results.append({
            "distance": r["distance"]/1000,
            "duration": r["duration"]/60,
            "coordinates": coords,
            "steps": steps
        })

    return {"routes": results}
