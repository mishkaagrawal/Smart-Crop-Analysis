from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import uuid
import requests
import math

app = Flask(__name__)

# ---------------- CONFIG ----------------
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['RESULT_FOLDER'] = "static/results"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load trained YOLO model
model = YOLO(r"C:\Users\Mishka\OneDrive\Desktop\weed-detection\best.pt")

# OpenWeatherMap API key
WEATHER_API_KEY = "1386efe894b89f35ed2ba12db37f9c44"

# Allowed Maharashtra cities only
MAHARASHTRA_CITIES = [
    "mumbai", "pune", "nagpur", "nashik", "aurangabad",
    "solapur", "kolhapur", "satara", "sangli",
    "jalgaon", "akola", "amravati", "latur",
    "nanded", "beed", "parbhani", "hingoli",
    "wardha", "chandrapur", "gondia", "bhandara",
    "ratnagiri", "sindhudurg", "ahmednagar", "dhule"
]

# ---------------- CROP LOGIC ----------------
def recommend_crop(temp, humidity):
    if temp >= 25 and humidity >= 60:
        return [
            "Rice – Clayey / Alluvial soil",
            "Cotton – Black cotton soil",
            "Jowar – Medium to deep black soil",
            "Soybean – Well-drained black soil",
            "Bajra – Sandy loam soil"
        ]
    elif 10 <= temp < 25 and humidity < 60:
        return [
            "Wheat – Loamy soil",
            "Gram – Sandy loam soil",
            "Mustard – Alluvial soil",
            "Safflower – Deep black soil"
        ]
    elif temp >= 21:
        return [
            "Sugarcane – Deep black soil (irrigated)",
            "Sunflower – Well-drained loamy soil",
            "Banana – Rich alluvial soil",
            "Turmeric – Sandy loam soil"
        ]
    else:
        return ["Vegetables – Loamy soil"]

# ---------------- WATER BODY FUNCTIONS ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_nearby_water_details(lat, lon):
    query = f"""
    [out:json];
    (
      way["natural"="water"](around:5000,{lat},{lon});
      relation["natural"="water"](around:5000,{lat},{lon});
      way["waterway"](around:5000,{lat},{lon});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data=query)

    water_bodies = []
    seen_names = set()

    if response.status_code == 200:
        data = response.json()
        for elem in data["elements"]:
            name = elem.get("tags", {}).get("name")
            if not name or name in seen_names:
                continue

            width = elem.get("tags", {}).get("width")
            if width:
                try:
                    if float(width) < 3:
                        continue
                except:
                    pass

            if "center" in elem:
                w_lat = elem["center"]["lat"]
                w_lon = elem["center"]["lon"]
            elif "lat" in elem:
                w_lat = elem["lat"]
                w_lon = elem["lon"]
            else:
                continue

            distance = round(haversine(lat, lon, w_lat, w_lon), 2)
            water_bodies.append({
                "name": name,
                "distance": distance,
                "lat": w_lat,
                "lon": w_lon
            })
            seen_names.add(name)

    return water_bodies

def water_management_advice(rainfall, water_bodies):
    if rainfall > 150:
        status = "High Rainfall"
        irrigation = "No irrigation required"
        warning = "Ensure drainage to prevent waterlogging"
    elif 50 <= rainfall <= 150:
        status = "Moderate Rainfall"
        irrigation = "Supplementary irrigation if required"
        warning = "Monitor soil moisture"
    else:
        status = "Low Rainfall"
        irrigation = "Drip or sprinkler irrigation recommended"
        warning = "Use water-saving techniques"

    source = "Nearby water bodies available" if water_bodies else "No nearby water source – Harvest rainwater"

    return {
        "status": status,
        "irrigation": irrigation,
        "warning": warning,
        "source": source,
        "details": water_bodies
    }

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    weather_data = None
    crops = []
    water_advice = None

    if request.method == "POST":

        # -------- WEED DETECTION --------
        if "image" in request.files:
            file = request.files["image"]
            if file.filename:
                filename = str(uuid.uuid4()) + ".jpg"
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_path)

                model.predict(input_path, save=True, project=app.config['RESULT_FOLDER'], name=filename)
                result_dir = os.path.join(app.config['RESULT_FOLDER'], filename)
                result_image = f"/static/results/{filename}/" + os.listdir(result_dir)[0]

        # -------- WEATHER & WATER --------
        city = request.form.get("city")
        if city:
            city = city.lower().strip()
            if city not in MAHARASHTRA_CITIES:
                weather_data = {"error": "Enter Maharashtra city only"}
            else:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city},MH,IN&appid={WEATHER_API_KEY}&units=metric"
                res = requests.get(url)

                if res.status_code == 200:
                    data = res.json()
                    temp = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    rainfall = data.get("rain", {}).get("1h", 0)
                    lat = data["coord"]["lat"]
                    lon = data["coord"]["lon"]

                    crops = recommend_crop(temp, humidity)
                    water_bodies = get_nearby_water_details(lat, lon)
                    water_advice = water_management_advice(rainfall, water_bodies)

                    weather_data = {
                        "city": city.title(),
                        "temp": temp,
                        "humidity": humidity,
                        "rainfall": rainfall
                    }

    return render_template(
        "index.html",
        result=result_image,
        weather=weather_data,
        crops=crops,
        water=water_advice
    )

if __name__ == "__main__":
    app.run(debug=True)
