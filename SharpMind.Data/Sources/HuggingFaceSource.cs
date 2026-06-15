using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace SharpMind.Data.Sources;

/// <summary>
/// Streams documents from the HuggingFace datasets-server REST API.
/// No NuGet dependency — uses only <see cref="HttpClient"/> and <c>System.Text.Json</c>.
///
/// Supports public and private datasets (pass a token for private repos).
/// Pages through the <c>/rows</c> endpoint automatically until all rows
/// are exhausted or the requested row limit is reached.
///
/// Usage:
/// <code>
///   var source = new HuggingFaceSource("allenai/c4", split: "train", textField: "text");
///   var source = new HuggingFaceSource("openai/gsm8k", split: "train",
///                                       config: "main", textField: "question");
/// </code>
/// </summary>
public sealed class HuggingFaceSource : IDataSource
{
    private const string BaseUrl = "https://datasets-server.huggingface.co";
    private const int DefaultPage = 100;

    private readonly HttpClient _http;
    private readonly bool _ownsClient;
    private readonly string _dataset;
    private readonly string _config;
    private readonly string _split;
    private readonly string[] _fieldPath;
    private readonly int _pageSize;
    private readonly long? _maxRows;

    // Construction

    /// <param name="dataset">Dataset repository id, e.g. <c>"allenai/c4"</c>.</param>
    /// <param name="split">Dataset split: <c>"train"</c>, <c>"validation"</c>, <c>"test"</c>.</param>
    /// <param name="config">
    /// Dataset config/subset name. Defaults to <c>"default"</c>.
    /// Required for datasets with multiple configs (e.g. <c>"en"</c> for C4).
    /// </param>
    /// <param name="textField">
    /// Field name containing the document text. Dot notation supported.
    /// Defaults to <c>"text"</c>.
    /// </param>
    /// <param name="tokenEnvVar">
    /// Name of the environment variable holding the HuggingFace API token.
    /// Defaults to <c>"HF_TOKEN"</c> — the standard HuggingFace convention.
    /// Required for private or gated datasets.
    /// </param>
    /// <param name="tokenOverride">
    /// Pass a token value directly. Not recommended — prefer the environment
    /// variable so tokens are never in source code or config files.
    /// Takes precedence over <paramref name="tokenEnvVar"/> when set.
    /// </param>
    /// <param name="pageSize">Rows per API call. Max 100 per HuggingFace limits.</param>
    /// <param name="maxRows">
    /// Maximum rows to stream. Null streams the entire split.
    /// Useful for quick experiments without loading a full 100GB+ split.
    /// </param>
    /// <param name="httpClient">
    /// Optional pre-configured client. When null a new client is created and
    /// disposed with this instance.
    /// </param>
    public HuggingFaceSource(
        string dataset,
        string split = "train",
        string config = "default",
        string textField = "text",
        string tokenEnvVar = "HF_TOKEN",
        string? tokenOverride = null,
        int pageSize = DefaultPage,
        long? maxRows = null,
        HttpClient? httpClient = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataset);
        ArgumentException.ThrowIfNullOrWhiteSpace(split);
        ArgumentException.ThrowIfNullOrWhiteSpace(config);
        ArgumentException.ThrowIfNullOrWhiteSpace(textField);

        _dataset = dataset;
        _config = config;
        _split = split;
        _fieldPath = textField.Split('.', StringSplitOptions.RemoveEmptyEntries);
        _pageSize = Math.Clamp(pageSize, 1, 100);
        _maxRows = maxRows;

        if (httpClient is not null)
        {
            _http = httpClient;
            _ownsClient = false;
        }
        else
        {
            _http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
            _ownsClient = true;
        }

        _http.DefaultRequestHeaders.Remove("User-Agent");
        _http.DefaultRequestHeaders.Add("User-Agent", "SharpMind/1.0");

        // Resolve token: direct override wins, then environment variable
        string? token = tokenOverride ?? Environment.GetEnvironmentVariable(tokenEnvVar);
        if (!string.IsNullOrWhiteSpace(token))
        {
            _http.DefaultRequestHeaders.Remove("Authorization");
            _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {token}");
        }
    }

    // IDataSource

    public long? EstimatedCount => _maxRows;

    public string Description =>
        $"HuggingFace({_dataset}/{_config}/{_split}, field={string.Join('.', _fieldPath)}" +
        (_maxRows.HasValue ? $", max={_maxRows}" : "") + ")";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        long offset = 0;
        long yielded = 0;
        long limit = _maxRows ?? long.MaxValue;

        while (yielded < limit)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int pageSize = (int)Math.Min(_pageSize, limit - yielded);
            var page = await FetchPageAsync(offset, pageSize, cancellationToken)
                                      .ConfigureAwait(false);

            if (page is null || page.Rows.Count == 0)
                yield break;

            foreach (var row in page.Rows)
            {
                cancellationToken.ThrowIfCancellationRequested();

                string? text = ExtractField(row.RowData);
                if (!string.IsNullOrWhiteSpace(text))
                {
                    yield return text;
                    if (++yielded >= limit) yield break;
                }
            }

            offset += page.Rows.Count;

            // HF API returns fewer rows than requested when we've hit the end
            if (page.Rows.Count < pageSize)
                yield break;
        }
    }

    public ValueTask DisposeAsync()
    {
        if (_ownsClient) _http.Dispose();
        return ValueTask.CompletedTask;
    }

    // API helpers

    private async Task<HuggingFacePage?> FetchPageAsync(
        long offset, int length, CancellationToken cancellationToken)
    {
        string url = $"{BaseUrl}/rows" +
                     $"?dataset={Uri.EscapeDataString(_dataset)}" +
                     $"&config={Uri.EscapeDataString(_config)}" +
                     $"&split={Uri.EscapeDataString(_split)}" +
                     $"&offset={offset}" +
                     $"&length={length}";

        using var response = await _http.GetAsync(url, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return await response.Content
                             .ReadFromJsonAsync<HuggingFacePage>(cancellationToken: cancellationToken)
                             .ConfigureAwait(false);
    }

    private string? ExtractField(JsonElement row)
    {
        JsonElement current = row;
        foreach (string key in _fieldPath)
        {
            if (current.ValueKind != JsonValueKind.Object ||
                !current.TryGetProperty(key, out current))
                return null;
        }
        return current.ValueKind == JsonValueKind.String ? current.GetString() : null;
    }

    // Response DTOs (internal — not part of the public API)

    private sealed class HuggingFacePage
    {
        public List<HuggingFaceRow> Rows { get; init; } = [];
    }

    private sealed class HuggingFaceRow
    {
        [System.Text.Json.Serialization.JsonPropertyName("row")]
        public JsonElement RowData { get; init; }
    }
}
