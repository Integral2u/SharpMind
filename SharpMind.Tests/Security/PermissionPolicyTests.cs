using SharpMind.Inference.Agent;

namespace SharpMind.Tests.Security;

/// <summary>
/// Covers the path-aware file permission rule and the permission decision logic:
/// approved sub-directories are assumed accessible; access that escapes to a parent
/// directory triggers the Ask flow; embedded-plugin tools are forced through Ask
/// even when the global default is Always, while Never still blocks.
/// </summary>
public class PermissionPolicyTests
{
    private static readonly string[] Roots =
    [
        @"C:\proj",
        @"C:\models",
    ];

    // ── PermissionPathPolicy ─────────────────────────────────────────────

    [Theory]
    [InlineData(@"C:\proj")]
    [InlineData(@"C:\proj\")]
    [InlineData(@"C:\proj\sub")]
    [InlineData(@"C:\proj\sub\file.txt")]
    public void IsUnderRoot_DirectChild_IsInside(string path)
    {
        Assert.True(PermissionPathPolicy.IsUnderRoot(path, Roots));
    }

    [Theory]
    [InlineData(@"C:\proj2")]
    [InlineData(@"C:\proj-other")]
    [InlineData(@"C:\")]
    [InlineData(@"C:\Windows")]
    [InlineData(@"C:\models2\file.txt")]
    [InlineData(@"C:\other\sub")]
    public void IsUnderRoot_ParentOrSibling_IsOutside(string path)
    {
        Assert.False(PermissionPathPolicy.IsUnderRoot(path, Roots));
    }

    [Theory]
    [InlineData(@"C:\proj\sub\..\file.txt", true)]   // escapes then re-enters the root
    [InlineData(@"C:\proj\..\other\file.txt", false)] // escapes to a sibling directory
    [InlineData(@"C:\proj\..\..\Windows\file.txt", false)] // escapes all the way out
    public void IsResourceInsideRoots_EscapesAreDetected(string resource, bool expected)
    {
        Assert.Equal(expected, PermissionPathPolicy.IsResourceInsideRoots(resource, Roots));
    }

    [Fact]
    public void IsResourceInsideRoots_RelativePaths_ProbeAgainstEachRoot()
    {
        // Relative resource resolves inside C:\proj\sub -> inside.
        Assert.True(PermissionPathPolicy.IsResourceInsideRoots(@"sub\file.txt", Roots));
        // Relative escape cannot resolve inside any root.
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(@"..\..\Windows\file.txt", Roots));
    }

    [Fact]
    public void IsResourceInsideRoots_NoRoots_NothingIsInside()
    {
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(@"C:\anything", []));
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(@"", Roots));
    }

    // ── PermissionPolicy.Resolve ─────────────────────────────────────────

    [Fact]
    public void Resolve_FileInsideApprovedRoot_InAskMode_IsAlways()
    {
        var result = PermissionPolicy.Resolve(
            "ListFiles", ToolCategory.File, @"C:\proj\sub\file.txt",
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Always, result);
    }

    [Fact]
    public void Resolve_FileOutsideApprovedRoot_InAskMode_Prompts()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, @"C:\Windows\file.txt",
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Ask, result);
    }

    [Fact]
    public void Resolve_FileInAskMode_AlwaysRespected()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, @"C:\Windows\file.txt",
            ToolPermission.Always, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Always, result);
    }

    [Fact]
    public void Resolve_FileInAskMode_NeverRespected()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, @"C:\proj\sub\file.txt",
            ToolPermission.Never, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Never, result);
    }

    [Fact]
    public void Resolve_Network_IgnoresPathRule()
    {
        // Network resources are never auto-allowed by the path rule, even "inside" a root.
        var result = PermissionPolicy.Resolve(
            "Fetch", ToolCategory.Network, @"C:\proj\file.txt",
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Ask, result);
    }

    // ── Restricted (embedded-plugin) tools ──────────────────────────────

    [Fact]
    public void Resolve_RestrictedTool_ForcesAskEvenWhenAlways()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "EmbeddedTool", ToolCategory.File, @"C:\Windows\file.txt",
            ToolPermission.Always, ToolPermission.Always, Roots, restricted);
        Assert.Equal(ToolPermission.Ask, result);
    }

    [Fact]
    public void Resolve_RestrictedTool_NeverStillBlocks()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "EmbeddedTool", ToolCategory.File, @"C:\proj\file.txt",
            ToolPermission.Never, ToolPermission.Ask, Roots, restricted);
        Assert.Equal(ToolPermission.Never, result);
    }

    [Fact]
    public void Resolve_RestrictedTool_AskModePrompts()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "EmbeddedTool", ToolCategory.Network, "https://example.com",
            ToolPermission.Ask, ToolPermission.Ask, Roots, restricted);
        Assert.Equal(ToolPermission.Ask, result);
    }

    [Fact]
    public void Resolve_NonRestrictedTool_UsesConfiguredDefault()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "NormalTool", ToolCategory.Network, "https://example.com",
            ToolPermission.Ask, ToolPermission.Always, Roots, restricted);
        Assert.Equal(ToolPermission.Always, result);
    }
}
