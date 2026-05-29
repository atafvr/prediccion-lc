import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "Smoking": 6,
    "Air Pollution": 5,
    "Alcohol use": 4,
    "Obesity": 5,
    "Genetic Risk": 6,
}

response = requests.post(url, json=payload, timeout=10)
print(response.status_code)
print(response.json())
