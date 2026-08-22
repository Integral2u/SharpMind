using System.Globalization;
using SharpMind.Core;

namespace SharpMind.Extensions.Tools;

/// <summary>
/// Provides date and time information — current time, time zone conversions,
/// and formatting.
/// </summary>
public class DateTimeTool
{
    [ToolDesc("Returns the current date and time in UTC and the local time zone, plus the day of the week.")]
    public static string GetNow()
    {
        var utc = DateTime.UtcNow;
        var local = DateTime.Now;
        var tz = TimeZoneInfo.Local;

        return $"UTC:     {utc:yyyy-MM-dd HH:mm:ss}\n" +
               $"Local:   {local:yyyy-MM-dd HH:mm:ss} ({tz.DisplayName})\n" +
               $"Day:     {utc:dddd}";
    }

    [ToolDesc("Converts a UTC timestamp to a target time zone. Use Windows names (e.g. 'Eastern Standard Time', 'GMT Standard Time').")]
    public static string ConvertTime(
        [ToolDesc("The UTC timestamp to convert (ISO 8601, e.g. '2026-01-15T14:30:00Z').")] string utcTimestamp,
        [ToolDesc("The target time zone name (e.g. 'Eastern Standard Time').")] string targetTimeZone)
    {
        try
        {
            if (!DateTime.TryParse(utcTimestamp, null, DateTimeStyles.RoundtripKind, out var dt))
                return $"Could not parse timestamp: '{utcTimestamp}'. Use ISO 8601 format.";

            if (dt.Kind != DateTimeKind.Utc)
                dt = DateTime.SpecifyKind(dt, DateTimeKind.Utc);

            TimeZoneInfo tz;
            try
            {
                tz = TimeZoneInfo.FindSystemTimeZoneById(targetTimeZone);
            }
            catch (TimeZoneNotFoundException)
            {
                tz = TimeZoneInfo.GetSystemTimeZones()
                    .FirstOrDefault(t => string.Equals(t.Id, targetTimeZone, StringComparison.OrdinalIgnoreCase))
                    ?? throw new TimeZoneNotFoundException($"Time zone not found: '{targetTimeZone}'.");
            }

            var converted = TimeZoneInfo.ConvertTimeFromUtc(dt, tz);
            return $"{dt:yyyy-MM-dd HH:mm:ss} UTC -> {converted:yyyy-MM-dd HH:mm:ss} ({tz.DisplayName})";
        }
        catch (Exception ex)
        {
            return $"Error converting time: {ex.Message}";
        }
    }

    [ToolDesc("Returns the list of available time zone IDs on this system.")]
    public static string ListTimeZones()
    {
        var zones = TimeZoneInfo.GetSystemTimeZones()
            .Select(z => $"{z.Id} — {z.DisplayName}")
            .Take(50);
        return $"Showing first 50 of {TimeZoneInfo.GetSystemTimeZones().Count} time zones:\n" + string.Join("\n", zones);
    }
}
