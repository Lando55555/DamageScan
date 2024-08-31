import requests

def retrieve_weather_and_humidity(location, api_key):
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data['current']['temp_c'],
            "humidity": data['current']['humidity']
        }
    except requests.RequestException as e:
        raise Exception(f"Weather API request failed: {str(e)}")