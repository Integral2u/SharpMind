using System.Text.Json;
using SharpMind;

namespace SharpMind.AgentTools;

public class WeatherTool
{
    private static readonly HttpClient _httpClient = new();

    [ToolDesc("Gets the current weather for a specified city.")]
    public async Task<string> GetCurrentWeather([ToolDesc("The name of the city.")] string city)
    {
        try
        {
            // 1. Geocoding: City Name -> Lat/Lon
            var geoUrl = $"https://geocoding-api.open-meteo.com/v1/search?name={Uri.EscapeDataString(city)}&count=1&language=en&format=json";
            var geoResponse = await _httpClient.GetStringAsync(geoUrl);
            using var geoDoc = JsonDocument.Parse(geoResponse);
            var results = geoDoc.RootElement.GetProperty("results");

            if (results.GetArrayLength() == 0)
                return $"Could not find coordinates for city: {city}";

            var first = results[0];
            double lat = first.GetProperty("latitude").GetDouble();
            double lon = first.GetProperty("longitude").GetDouble();
            string name = first.GetProperty("name").GetString() ?? city;
            string country = first.TryGetProperty("country", out var c) ? c.GetString() : "";

            // 2. Weather: Lat/Lon -> Current Weather
            var weatherUrl = $"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true";
            var weatherResponse = await _httpClient.GetStringAsync(weatherUrl);
            using var weatherDoc = JsonDocument.Parse(weatherResponse);
            var current = weatherDoc.RootElement.GetProperty("current_weather");

            double temp = current.GetProperty("temperature").GetDouble();
            int weatherCode = current.GetProperty("weathercode").GetInt32();
            double windSpeed = current.GetProperty("windspeed").GetDouble();

            string condition = WeatherCodeToDescription(weatherCode);

            return $"The current weather in {name}, {country} is {temp}°C with {condition} and a wind speed of {windSpeed} km/h.";
        }
        catch (Exception ex)
        {
            return $"Error fetching weather for {city}: {ex.Message}";
        }
    }

    private static string WeatherCodeToDescription(int code)
    {
        return code switch
        {
            0 => "clear sky",
            1 => "mainly clear",
            2 => "partly cloudy",
            3 => "overcast",
            45 => "fog",
            48 => "depositing rime fog",
            51 => "light drizzle",
            53 => "moderate drizzle",
            55 => "dense drizzle",
            61 => "slight rain",
            63 => "moderate rain",
            65 => "heavy rain",
            80 => "slight rain showers",
            // ... simplified for brevity
            _ => $"weather code {code}"
        };
    }
}
