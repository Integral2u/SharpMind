using SharpMind.Model.Vision;
using Xunit;

namespace SharpMind.Tests.Vision;

public sealed class Lfm3VlRuntimeTests
{
    [Theory]
    [InlineData(512, 512, 256)]
    [InlineData(512, 256, 128)]
    [InlineData(256, 256, 64)]
    [InlineData(32, 32, 1)]
    public void ImageTokens_IsThePatchGridDividedByTheMergeArea(int height, int width, int expected)
    {
        Assert.Equal(expected, Lfm3VlRuntime.ImageTokens(height, width));
    }

    [Fact]
    public void ImageTokens_RequiresAWholeMergeBlock()
    {
        // A 32-pixel image is the smallest size that yields one merged token.
        Assert.Equal(0, Lfm3VlRuntime.ImageTokens(16, 16));
    }
}
