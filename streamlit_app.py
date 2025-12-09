# smart_routes_ev_traffic.py
import streamlit as st
from streamlit_folium import st_folium
import folium
import requests, time, math, hashlib, random
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="Smart Routes — Traffic Priority + EV Stops", layout="wide")

# ---------------- SESSION STATE INITIALIZATION ----------------
if "trip_log" not in st.session_state:
    st.session_state.trip_log = []

if "source" not in st.session_state:
    st.session_state.source = ""

if "destination" not in st.session_state:
    st.session_state.destination = ""

# Optional: store last recorded timestamp to avoid duplicates on reruns
if "last_trip_timestamp" not in st.session_state:
    st.session_state.last_trip_timestamp = None

# ---------------- Helpers ----------------
USER_AGENT = "smart-routes-ev/1.0"

def geocode(place):
    if not place or not place.strip():
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        r = requests.get(url, params={"q": place, "format": "json", "limit": 1},
                         headers={"User-Agent": USER_AGENT}, timeout=8)
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return float(d["lat"]), float(d["lon"])
    except Exception:
        pass
    # fallback geopy
    try:
        geo = Nominatim(user_agent=USER_AGENT, timeout=8)
        loc = geo.geocode(place)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None

def generate_query_variants(slat, slon, elat, elon):
    variants = []
    variants.append(f"{slon},{slat};{elon},{elat}")
    shifts = [0.0025, -0.0025, 0.005, -0.005]  # small shifts (200-500m)
    for d in shifts:
        variants.append(f"{slon + d},{slat + d};{elon},{elat}")
        variants.append(f"{slon},{slat};{elon + d},{elat + d}")
    # dedupe keeping order
    seen = set(); out=[]
    for q in variants:
        if q not in seen:
            seen.add(q); out.append(q)
    return out

def unique_key(route_obj):
    try:
        coords = route_obj["geometry"]["coordinates"]
        return f"{len(coords)}_{int(route_obj.get('distance',0))}"
    except Exception:
        return None

def route_struct(route_obj):
    coords = [(c[1], c[0]) for c in route_obj["geometry"]["coordinates"]]
    distance_km = route_obj.get("distance",0)/1000.0
    duration_min = route_obj.get("duration",0)/60.0
    # collect simple text steps
    steps=[]
    for leg in route_obj.get("legs",[]):
        for stp in leg.get("steps",[]):
            instr = (stp.get("maneuver") or {}).get("instruction") or stp.get("name") or stp.get("maneuver",{}).get("type") or "Proceed"
            steps.append(instr)
    return {"coords": coords, "distance_km": round(distance_km,3), "duration_min": round(duration_min,1), "osrm": route_obj, "steps": steps}

def collect_alternatives(s_coord, e_coord, n=3):
    slat, slon = s_coord; elat, elon = e_coord
    queries = generate_query_variants(slat, slon, elat, elon)
    found = {}
    for q in queries:
        try:
            url = "http://router.project-osrm.org/route/v1/driving/" + q
            r = requests.get(url, params={"overview":"full","geometries":"geojson","steps":"true"}, timeout=12)
            if r.status_code != 200:
                continue
            js = r.json()
            if js.get("code") != "Ok" or not js.get("routes"):
                continue
            ro = js["routes"][0]
            key = unique_key(ro)
            if key and key not in found:
                found[key] = ro
        except Exception:
            pass
        if len(found) >= n:
            break
        time.sleep(0.2)
    return [route_struct(r) for r in found.values()]

def seed_base(coords, extra=0):
    if not coords:
        return 0
    s = f"{coords[0]}_{coords[-1]}_{len(coords)}_{extra}"
    return int(hashlib.sha1(s.encode()).hexdigest()[:8],16)

def haversine_km(a,b):
    R=6371.0
    lat1,lon1=math.radians(a[0]),math.radians(a[1])
    lat2,lon2=math.radians(b[0]),math.radians(b[1])
    dlat=lat2-lat1; dlon=lon2-lon1
    aa=math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R*2*math.asin(min(1, math.sqrt(aa)))

def simulate_traffic(coords, hour_offset=0):
    seed = seed_base(coords, 101) + int((datetime.utcnow().timestamp()//3600) + round(hour_offset))
    rnd = random.Random(seed)
    levels=[]
    for i in range(max(1, len(coords)-1)):
        dist = haversine_km(coords[i], coords[i+1])
        base = max(0.05, min(0.75, 0.32 - 0.03*dist))
        val = min(1.0, max(0.0, base + rnd.gauss(0,0.15)))
        levels.append(round(val,3))
    return levels

def simulate_road_quality(coords):
    rnd = random.Random(seed_base(coords, 202))
    quals = []
    for i in range(max(1, len(coords)-1)):
        q = min(1.0, max(0.0, rnd.gauss(0.85,0.15)))
        quals.append(round(q,3))
    return quals

def compute_fuel_and_co2(distance_km, mileage_kmpl, fuel_price, co2_per_l=2.31):
    liters = distance_km / mileage_kmpl
    cost = liters * fuel_price
    co2 = liters * co2_per_l
    return round(liters,2), round(cost,2), round(co2,2)

def find_chargers_near(lat, lon, radius=3000, max_items=50):
    q = f"""
    [out:json][timeout:25];
    node(around:{radius},{lat},{lon})[amenity=charging_station];
    out center {max_items};
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=20)
        if r.status_code != 200: return []
        js = r.json()
        out=[]
        for el in js.get("elements",[])[:max_items]:
            name = el.get("tags",{}).get("name") or "Charging"
            out.append({"lat": el.get("lat"), "lon": el.get("lon"), "name": name})
        return out
    except Exception:
        return []

def plan_ev_stops_along_route(route_coords, ev_range_km, search_radius=2500):
    if ev_range_km <= 0:
        return []
    total_dist = 0.0
    stops=[]
    accumulated=0.0
    for i in range(1, len(route_coords)):
        seg = haversine_km(route_coords[i-1], route_coords[i])
        accumulated += seg
        total_dist += seg
        if accumulated >= ev_range_km*0.9:
            mid = route_coords[i]
            candidates = find_chargers_near(mid[0], mid[1], radius=search_radius, max_items=20)
            if candidates:
                cand = min(candidates, key=lambda c: haversine_km((c["lat"],c["lon"]), mid))
                stops.append({"lat": cand["lat"], "lon": cand["lon"], "name": cand["name"], "at_index": i})
                accumulated = 0.0
            else:
                stops.append({"lat": mid[0], "lon": mid[1], "name": "No charger found nearby (need manual planning)", "at_index": i})
                accumulated = 0.0
    return stops

# -------------------- UI --------------------
st.title("Smart Routes — Traffic-Priority Ranking + EV Stop Planner")

left, right = st.columns([3,1])
with left:
    # use session_state for persistent inputs
    st.session_state.source = st.text_input("Start", value=st.session_state.source, key="start_input")
    st.session_state.destination = st.text_input("End", value=st.session_state.destination, key="end_input")

    start_place = st.session_state.source or "Majestic Bangalore"
    end_place = st.session_state.destination or "Whitefield Bangalore"

    max_alts = st.number_input("Max alternatives (attempts)", 1, 6, 3)
    mileage = st.number_input("Vehicle mileage (km/L)", 18.0, step=1.0)
    fuel_price = st.number_input("Fuel price (₹/L)", 105.0, step=1.0)
    ev_range_km = st.number_input("EV range (km)", 250.0, step=10.0)
    run_button = st.button("Generate & Analyze")
with right:
    st.markdown("### Scoring (higher traffic weight => traffic-priority)")
    w_dist = st.slider("Distance weight", 0.0,1.0,0.2)
    w_traf = st.slider("Traffic weight", 0.0,1.0,0.4)
    w_quality = st.slider("Road quality weight", 0.0,1.0,0.15)
    w_fuel = st.slider("Fuel cost weight", 0.0,1.0,0.15)
    w_co2 = st.slider("CO₂ weight", 0.0,1.0,0.1)
    totw = w_dist + w_traf + w_quality + w_fuel + w_co2
    if totw == 0: totw = 1.0
    W = {"dist": w_dist/totw, "traf": w_traf/totw, "qual": w_quality/totw, "fuel": w_fuel/totw, "co2": w_co2/totw}

if run_button:
    st.session_state.pop("routes_analyzed", None)
    with st.spinner("Geocoding places..."):
        s_coord = geocode(start_place)
        e_coord = geocode(end_place)
    if not s_coord:
        st.error("Start location not found — try more specific place name.")
    elif not e_coord:
        st.error("End location not found — try more specific place name.")
    else:
        with st.spinner("Collecting alternative routes (OSRM)..."):
            alts = collect_alternatives(s_coord, e_coord, n=max_alts)
        if not alts:
            st.error("No routes found — try modifying place names (landmark/station).")
        else:
            analyzed=[]
            for r in alts:
                coords = r["coords"]
                traffic_now = simulate_traffic(coords, hour_offset=0)
                traffic_30 = simulate_traffic(coords, hour_offset=1)
                traffic_60 = simulate_traffic(coords, hour_offset=2)
                qual = simulate_road_quality(coords)
                avg_traffic = sum(traffic_now)/len(traffic_now) if traffic_now else 0
                avg_quality = sum(qual)/len(qual) if qual else 1
                liters, cost, co2 = compute_fuel_and_co2(r["distance_km"], mileage, fuel_price)
                s_dist = r["distance_km"]
                s_traf = avg_traffic*100
                s_qual = (1 - avg_quality)*100
                s_fuel = cost/10.0
                s_co2 = co2/5.0
                score = W["dist"]*s_dist + W["traf"]*s_traf + W["qual"]*s_qual + W["fuel"]*s_fuel + W["co2"]*s_co2
                ev_stops = []
                if r["distance_km"] > ev_range_km:
                    ev_stops = plan_ev_stops_along_route(coords, ev_range_km, search_radius=3000)
                analyzed.append({
                    "coords": coords,
                    "distance_km": r["distance_km"],
                    "duration_min": r["duration_min"],
                    "traffic_now": traffic_now,
                    "traffic_30": traffic_30,
                    "traffic_60": traffic_60,
                    "road_quality": qual,
                    "avg_traffic": round(avg_traffic,3),
                    "avg_quality": round(avg_quality,3),
                    "fuel_l": liters, "fuel_cost": cost, "co2": co2,
                    "score": round(score,3),
                    "ev_stops": ev_stops,
                    "steps": r.get("steps", [])
                })
            analyzed = sorted(analyzed, key=lambda x: x["score"])
            st.session_state["routes_analyzed"] = analyzed
            st.session_state["start_coord"] = s_coord
            st.session_state["end_coord"] = e_coord
            st.success(f"Analyzed {len(analyzed)} route(s). Best score: {analyzed[0]['score']}")

# ---------------- Display map & results ----------------
if "routes_analyzed" in st.session_state:
    routes = st.session_state["routes_analyzed"]
    s_coord = st.session_state["start_coord"]
    e_coord = st.session_state["end_coord"]
    colors = ["red","blue","green","purple","orange","darkred"]

    m = folium.Map(location=s_coord, zoom_start=12, tiles="OpenStreetMap")

    for i, r in enumerate(routes):
        col = colors[i % len(colors)]
        # Thinner alternative lines, best route thicker
        line_weight = 6 if i == 0 else 3
        folium.PolyLine(r["coords"], color=col, weight=line_weight, opacity=0.85,
                        tooltip=f"Route {i+1} • {r['distance_km']} km • {r['duration_min']} min • score {r['score']}").add_to(m)
        for j in range(len(r["coords"])-1):
            a = r["coords"][j]; b = r["coords"][j+1]
            mid = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
            level = r["traffic_now"][j] if j < len(r["traffic_now"]) else 0
            q = r["road_quality"][j] if j < len(r["road_quality"]) else 1
            col_marker = "green" if level < 0.33 else "orange" if level < 0.66 else "red"
            folium.CircleMarker(location=mid, radius=4, color=col_marker, fill=True, fill_opacity=0.9,
                                popup=f"Traffic:{level} • RoadQ:{q}").add_to(m)

        for stop in r["ev_stops"]:
            folium.Marker((stop["lat"], stop["lon"]), icon=folium.Icon(color="lightblue", icon="bolt"),
                          tooltip=stop["name"]).add_to(m)

    folium.Marker(s_coord, icon=folium.Icon(color="green"), tooltip="Start").add_to(m)
    folium.Marker(e_coord, icon=folium.Icon(color="red"), tooltip="End").add_to(m)

    st.subheader("🗺 Clean Alternatives With Traffic & Road Quality")
    st_folium(m, width=950, height=600)

    # Summary panel
    st.subheader("Route Summaries (sorted by composite score — lower better)")
    for idx, r in enumerate(routes, start=1):
        st.markdown(f"**Route {idx}** — {r['distance_km']} km • {r['duration_min']} min  • Score: **{r['score']}**")
        st.write(f"Avg traffic(now): {r['avg_traffic']} • Avg road quality: {r['avg_quality']}")
        st.write(f"Fuel: {r['fuel_l']} L • Cost: ₹{r['fuel_cost']} • CO₂: {r['co2']} kg")
        if r["ev_stops"]:
            st.write("Planned EV stops:")
            for s in r["ev_stops"]:
                st.write(f"- {s['name']} @ index {s['at_index']} ({round(haversine_km((s['lat'],s['lon']), e_coord),2)} km to dest)")
        else:
            st.write("No EV stops needed (range sufficient).")
        st.markdown("---")

    # Best route ETA predictions
    best = routes[0]
    st.subheader("Best route ETA predictions (simulated)")
    now_eta = best["duration_min"]
    def predict_eta(base_duration, current_levels, future_levels, factor=0.45):
        if not current_levels or not future_levels:
            return base_duration
        cur = sum(current_levels)/len(current_levels)
        fut = sum(future_levels)/len(future_levels)
        delta = max(-0.3, min(0.5, (fut - cur)))
        return round(base_duration * (1 + delta * factor), 1)
    eta_now = now_eta
    eta_30 = predict_eta(now_eta, best["traffic_now"], best["traffic_30"])
    eta_60 = predict_eta(now_eta, best["traffic_now"], best["traffic_60"], factor=0.6)
    st.write(f"Now: {eta_now} min  •  +30m predicted: {eta_30} min  •  +60m predicted: {eta_60} min")

    # Turn-by-turn
    st.subheader("Turn-by-turn (best route)")
    if best["steps"]:
        for i, s in enumerate(best["steps"][:40], start=1):
            st.write(f"{i}. {s}")
    else:
        st.write("No steps available.")
# ---------------- TRIP LOG MODULE WITH AUTO WEATHER & AUTO HEADWAY ----------------

import random
import streamlit as st
from datetime import datetime
import pandas as pd

# Init session state
if "trip_log" not in st.session_state:
    st.session_state.trip_log = []

st.subheader("🚌 Real-Time Bus Trip Recorder (Auto Weather + Auto Headway)")

# ---------------- INPUTS ----------------
bus_id = st.text_input("🚌 Bus ID", placeholder="e.g., BUS-1023")
route_id = st.text_input("🛣 Route ID", placeholder="e.g., R-21")
source = st.text_input("📍 Source", placeholder="Enter source location")
destination = st.text_input("🏁 Destination", placeholder="Enter destination")

# ---------------- AUTO WEATHER GENERATOR ----------------
def auto_weather():
    r = random.random()
    if r < 0.60:
        return "No Rain"
    elif r < 0.80:
        return "Light Rain"
    elif r < 0.95:
        return "Moderate Rain"
    else:
        return "Heavy Rain"

# ---------------- RECORD TRIPS ----------------
if st.button("📡 Record Trip Data"):
    if not bus_id or not route_id or not source or not destination:
        st.error("❌ Please fill Bus ID, Route ID, Source, and Destination.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Auto weather — NEW
        rain_status = auto_weather()

        # realistic distance & speed
        distance_km = round(random.uniform(2.5, 3.2), 3)
        speed_kmh = round(random.uniform(20, 60), 2)
        travel_time_min = round((distance_km / speed_kmh) * 60, 2)

        # Traffic level simulation
        traffic_level = round(random.uniform(0.1, 0.9), 3)

        # Auto headway — uses weather effect
        base_headway = random.randint(40, 120)

        if rain_status == "No Rain":
            rain_delay = 0
        elif rain_status == "Light Rain":
            rain_delay = random.randint(10, 25)
        elif rain_status == "Moderate Rain":
            rain_delay = random.randint(25, 45)
        else:
            rain_delay = random.randint(45, 80)

        headway_sec = base_headway + rain_delay

        # Congestion level (weather + traffic)
        rain_factor = {
            "No Rain": 0,
            "Light Rain": 0.05,
            "Moderate Rain": 0.12,
            "Heavy Rain": 0.20
        }[rain_status]

        congestion_level = round(min(1, traffic_level + rain_factor), 3)

        # Save
        st.session_state.trip_log.append({
            "timestamp": timestamp,
            "bus_id": bus_id,
            "route_id": route_id,
            "source": source,
            "destination": destination,
            "distance_km": distance_km,
            "speed_kmh": speed_kmh,
            "travel_time_min": travel_time_min,
            "traffic_level": traffic_level,
            "congestion_level": congestion_level,
            "rain": rain_status,
            "headway_sec": headway_sec
        })

        st.success("✅ Trip data recorded with automatic weather + headway!")

# ---------------- TABLE ----------------
st.markdown("### 📄 Trip Log (Auto Updated)")

if len(st.session_state.trip_log) > 0:
    df = pd.DataFrame(st.session_state.trip_log)
    st.dataframe(df, use_container_width=True)

    # Download
    st.download_button(
        "⬇️ Download Trip CSV",
        df.to_csv(index=False),
        "trip_log.csv",
        "text/csv"
    )
else:
    st.info("No trips yet.")
