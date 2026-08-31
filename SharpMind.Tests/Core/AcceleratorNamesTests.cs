using SharpMind.Core.Plugins;

namespace SharpMind.Tests.Core;

/// <summary>
/// The legacy-name contract both resolvers and the CUI selectors rely on: a stored pre-rename
/// <c>"cuda"</c> must canonicalize to today's <c>"ilgpu"</c> (case-insensitive) so old saves and
/// presets keep resolving instead of breaking on load.
/// </summary>
public sealed class AcceleratorNamesTests
{
    [Theory]
    [InlineData("cuda", "ilgpu")]
    [InlineData("CUDA", "ilgpu")]
    [InlineData("Cuda", "ilgpu")]
    public void Canonicalize_MapsLegacyCudaToIlgpu_AnyCasing(string name, string expected)
        => Assert.Equal(expected, AcceleratorNames.Canonicalize(name));

    [Theory]
    [InlineData("ilgpu")]
    [InlineData("ILGPU")]
    [InlineData("cpu")]
    [InlineData("metal")]
    [InlineData("")]
    public void Canonicalize_LeavesNonAliasedNamesUnchanged(string name)
        => Assert.Equal(name, AcceleratorNames.Canonicalize(name));

    [Fact]
    public void Matches_AcceptsTheLegacyAliasAndAnyCasing()
    {
        Assert.True(AcceleratorNames.Matches("cuda", "ilgpu"));
        Assert.True(AcceleratorNames.Matches("CUDA", "ilgpu"));
        Assert.True(AcceleratorNames.Matches("ilgpu", "ilgpu"));
        Assert.True(AcceleratorNames.Matches("ILGPU", "ilgpu"));
        Assert.False(AcceleratorNames.Matches("cuda", "cuda")); // canonical target is "ilgpu", not "cuda"
        Assert.False(AcceleratorNames.Matches("metal", "ilgpu"));
    }
}
