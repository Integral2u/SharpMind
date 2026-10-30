using System.Globalization;
using System.Text.Json;

namespace SharpMind.Core.AgentTools;

public class WeatherTool : INetworkToolService
{
    private static readonly HttpClient _httpClient = new() { Timeout = TimeSpan.FromSeconds(15) };

    [ToolDesc("Gets the current weather for a city, with temperature, conditions and wind speed. Pass the city name; add the country when the name is ambiguous, e.g. 'Paris, France'.")]
    public static async Task<string> GetCurrentWeather(
        [ToolDesc("The city name, optionally with the country, e.g. 'Paris, France'.")] string city,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(city))
            return "Error: no city given. Pass the city name.";

        try
        {
            // 1. Geocoding: city name -> latitude and longitude
            var geoUrl = $"https://geocoding-api.open-meteo.com/v1/search?name={Uri.EscapeDataString(city.Trim())}&count=1&language=en&format=json";
            using var geoDoc = JsonDocument.Parse(await _httpClient.GetStringAsync(geoUrl, cancellationToken));

            // The geocoder omits "results" entirely when nothing matches, so read it optionally.
            if (!geoDoc.RootElement.TryGetProperty("results", out var results)
                || results.ValueKind != JsonValueKind.Array
                || results.GetArrayLength() == 0)
            {
                return $"City not found: '{city}'. Try adding the country, e.g. '{city}, <country>'.";
            }

            var first = results[0];
            double lat = first.GetProperty("latitude").GetDouble();
            double lon = first.GetProperty("longitude").GetDouble();
            string name = first.TryGetProperty("name", out var n) ? n.GetString() ?? city : city;
            string country = first.TryGetProperty("country", out var c) ? c.GetString() ?? "" : "";

            // 2. Current conditions at those coordinates
            var weatherUrl = string.Create(CultureInfo.InvariantCulture,
                $"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true");
            using var weatherDoc = JsonDocument.Parse(await _httpClient.GetStringAsync(weatherUrl, cancellationToken));
            if (!weatherDoc.RootElement.TryGetProperty("current_weather", out var current))
                return $"Weather data is not available for {name} right now.";

            double temp = current.GetProperty("temperature").GetDouble();
            int weatherCode = current.GetProperty("weathercode").GetInt32();
            double windSpeed = current.GetProperty("windspeed").GetDouble();

            string where = string.IsNullOrEmpty(country) ? name : $"{name}, {country}";
            return $"Current weather in {where}: {temp.ToString(CultureInfo.InvariantCulture)}°C, {WeatherCodeToDescription(weatherCode)}, wind {windSpeed.ToString(CultureInfo.InvariantCulture)} km/h.";
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return $"Error fetching weather for {city}: the weather service did not respond in time.";
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException or KeyNotFoundException or InvalidOperationException)
        {
            return $"Error fetching weather for {city}: {ex.Message}";
        }
    }

    /// <summary>WMO weather interpretation codes (Open-Meteo), as text.</summary>
    internal static string WeatherCodeToDescription(int code) => code switch
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
        56 => "light freezing drizzle",
        57 => "dense freezing drizzle",
        61 => "slight rain",
        63 => "moderate rain",
        65 => "heavy rain",
        66 => "light freezing rain",
        67 => "heavy freezing rain",
        71 => "slight snow fall",
        73 => "moderate snow fall",
        75 => "heavy snow fall",
        77 => "snow grains",
        80 => "slight rain showers",
        81 => "moderate rain showers",
        82 => "violent rain showers",
        85 => "slight snow showers",
        86 => "heavy snow showers",
        95 => "thunderstorm",
        96 => "thunderstorm with slight hail",
        99 => "thunderstorm with heavy hail",
        _ => $"weather code {code}"
    };
}
