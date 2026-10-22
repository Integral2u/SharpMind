using SharpMind.Data.Sources;
using Xunit;

namespace SharpMind.Tests.Data;

public sealed class FusechatSourceTests
{
    [Fact]
    public async Task ReadAsync_StreamsRootArrayEntries()
    {
        string path = Path.Combine(Path.GetTempPath(), "sm-fusechat-" + Guid.NewGuid().ToString("N") + ".json");
        await File.WriteAllTextAsync(path, """
            [
              { "conversations": [ { "from": "human", "value": "hi" }, { "from": "gpt", "value": "hello" } ] },
              { "conversations": [ { "from": "human", "value": "again" }, { "from": "gpt", "value": "ok" } ] }
            ]
            """);

        try
        {
            var source = new FusechatSource(path);
            var documents = new List<string>();
            await foreach (string document in source.ReadAsync())
                documents.Add(document);

            Assert.Equal(2, documents.Count);
            Assert.Contains("human: hi", documents[0]);
            Assert.Contains("gpt: hello", documents[0]);
            Assert.Contains("human: again", documents[1]);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
