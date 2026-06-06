namespace SharpMind.Inference.Agent;

/// <summary>
/// Marker interface. Apply to any tool host class whose methods read or write
/// the local file system. ChatSession will route calls through the File
/// permission gate when this interface is detected on the host object.
/// </summary>
/// <example>
/// <code>
/// public class FileTools : IFileToolService
/// {
///     [ToolDesc("Read a text file and return its contents.")]
///     public string ReadFile([ToolDesc("Absolute path to the file.")] string path)
///         => File.ReadAllText(path);
/// }
/// </code>
/// </example>
public interface IFileToolService { }

/// <summary>
/// Marker interface. Apply to any tool host class whose methods make outbound
/// network calls. ChatSession will route calls through the Network permission
/// gate when this interface is detected on the host object.
/// </summary>
/// <example>
/// <code>
/// public class WebTools : INetworkToolService
/// {
///     [ToolDesc("Fetch the body of a URL.")]
///     public async Task&lt;string&gt; GetAsync([ToolDesc("URL to fetch.")] string url)
///         => await new HttpClient().GetStringAsync(url);
/// }
/// </code>
/// </example>
public interface INetworkToolService { }
