# ─────────────────────────────────────────────────────────────────────────────
# MCP Server — real weather data via Open-Meteo
#
# What this does:
#   Exposes one tool: get_weather(city)
#   Claude sees this tool, decides when to call it, gets the result back
#
# How MCP works:
#   1. This server runs as a local process
#   2. Claude Code connects to it via stdio (standard input/output)
#   3. When you ask "what's the weather in Chennai?" Claude calls get_weather
#   4. The server calls Open-Meteo (free, no API key needed) for real data
#   5. Claude uses that data to form its response
#
# APIs used (both free, no key required):
#   - geocoding: https://geocoding-api.open-meteo.com  (city name → lat/lon)
#   - weather:   https://api.open-meteo.com            (lat/lon → conditions)
# ─────────────────────────────────────────────────────────────────────────────

import requests

from mcp.server.fastmcp import FastMCP  # FastMCP is the simple way to build MCP servers

# create the MCP server — give it a name (shows up in Claude's tool list)
mcp = FastMCP("weather-server")

# WMO Weather Interpretation Codes → human-readable descriptions
# https://open-meteo.com/en/docs#weathervariables
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight showers", 81: "moderate showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "heavy thunderstorm with hail",
}


def _fetch_json(url: str) -> dict:
    """Simple HTTP GET that returns parsed JSON."""
    return requests.get(url, timeout=10).json()


# ── define a tool ─────────────────────────────────────────────────────────────
# @mcp.tool() registers this function as a callable tool for the LLM
# Claude can read the function name, docstring, and type hints to understand it

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather for a city using real-time data from Open-Meteo.
    Returns temperature, humidity, wind speed, and conditions.
    """

    # ── step 1: geocode the city name to lat/lon ──────────────────────────────
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo = requests.get(geo_url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=10).json()

    if not geo.get("results"):
        return f"Could not find location data for '{city}'."

    result = geo["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    full_name = f"{result['name']}, {result.get('country', '')}"

    # ── step 2: fetch current weather for that lat/lon ────────────────────────
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        f"&timezone=auto"
    )
    data = _fetch_json(weather_url)
    current = data["current"]

    temp      = current["temperature_2m"]
    humidity  = current["relative_humidity_2m"]
    wind      = current["wind_speed_10m"]
    condition = WMO_CODES.get(current["weather_code"], "unknown conditions")

    return (
        f"Weather in {full_name}: {temp}°C, {condition}. "
        f"Humidity {humidity}%, wind {wind} km/h."
    )


# ── run the server ────────────────────────────────────────────────────────────
# transport="stdio" means Claude communicates with this server via stdin/stdout
# This is the standard MCP transport for local servers

if __name__ == "__main__":
    print("MCP weather server starting...")  # confirm it's running
    mcp.run(transport="stdio")  # hand control to the MCP event loop
