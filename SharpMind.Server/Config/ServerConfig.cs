namespace SharpMind.Server.Config;

/// <summary>
/// Configuration POCO that maps to the "SharpMind" section of appsettings.json.
/// Builder options override these values.
/// </summary>
public sealed class ServerConfig
{
    /// <summary>
    /// Directory to scan for model files. Empty = use runtime default (~/SharpMind/Models).
    /// </summary>
    public string ModelsDir { get; set; } = "";

    /// <summary>
    /// HTTP port. Defaults to 11435.
    /// </summary>
    public int Port { get; set; } = 11435;
}
