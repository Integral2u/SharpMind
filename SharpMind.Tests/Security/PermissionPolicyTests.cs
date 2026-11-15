using SharpMind.Inference.Agent;

namespace SharpMind.Tests.Security;

/// <summary>
/// Covers the path-aware file permission rule and the permission decision logic:
/// approved sub-directories are assumed accessible; access that escapes to a parent
/// directory triggers the Ask flow; embedded-plugin tools are forced through Ask
/// even when the global default is Always, while Never still blocks.
/// </summary>
/// <remarks>
/// The roots are synthesized under the temp directory rather than hard-coded as
/// <c>C:\...</c> strings so the same expectations hold on every OS: a literal
/// Windows path is not rooted on Linux, where it silently becomes a child of the
/// current directory and the "escapes the root" cases stop escaping.
/// </remarks>
public class PermissionPolicyTests
{
    private static readonly string Base =
        Path.Combine(Path.GetTempPath(), "sharpmind-permission");

    private static readonly string Proj = Path.Combine(Base, "proj");
    private static readonly string Models = Path.Combine(Base, "models");

    private static readonly string[] Roots = [Proj, Models];

    // ── PermissionPathPolicy ─────────────────────────────────────────────

    public static TheoryData<string> InsidePaths => new()
    {
        Proj,
        Proj + Path.DirectorySeparatorChar,
        Path.Combine(Proj, "sub"),
        Path.Combine(Proj, "sub", "file.txt"),
    };

    public static TheoryData<string> OutsidePaths => new()
    {
        Proj + "2",                              // shares the prefix but is a sibling
        Proj + "-other",
        Base,                                    // parent of both roots
        Path.Combine(Base, "Windows"),
        Models + "2",
        Path.Combine(Base, "other", "sub"),
    };

    public static TheoryData<string, bool> EscapePaths => new()
    {
        { Path.Combine(Proj, "sub", "..", "file.txt"), true },    // escapes then re-enters the root
        { Path.Combine(Proj, "..", "other", "file.txt"), false }, // escapes to a sibling directory
        { Path.Combine(Proj, "..", "..", "Windows", "file.txt"), false }, // escapes all the way out
    };

    [Theory]
    [MemberData(nameof(InsidePaths))]
    public void IsUnderRoot_DirectChild_IsInside(string path)
    {
        Assert.True(PermissionPathPolicy.IsUnderRoot(path, Roots));
    }

    [Theory]
    [MemberData(nameof(OutsidePaths))]
    public void IsUnderRoot_ParentOrSibling_IsOutside(string path)
    {
        Assert.False(PermissionPathPolicy.IsUnderRoot(path, Roots));
    }

    [Theory]
    [MemberData(nameof(EscapePaths))]
    public void IsResourceInsideRoots_EscapesAreDetected(string resource, bool expected)
    {
        Assert.Equal(expected, PermissionPathPolicy.IsResourceInsideRoots(resource, Roots));
    }

    [Fact]
    public void IsResourceInsideRoots_RelativePaths_ProbeAgainstEachRoot()
    {
        // Relative resource resolves inside <root>/sub -> inside.
        Assert.True(PermissionPathPolicy.IsResourceInsideRoots(
            Path.Combine("sub", "file.txt"), Roots));
        // Relative escape cannot resolve inside any root.
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(
            Path.Combine("..", "..", "Windows", "file.txt"), Roots));
    }

    [Fact]
    public void IsResourceInsideRoots_NoRoots_NothingIsInside()
    {
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(
            Path.Combine(Base, "anything"), []));
        Assert.False(PermissionPathPolicy.IsResourceInsideRoots(@"", Roots));
    }

    [Fact]
    public void IsUnderRoot_CaseSensitivityFollowsPlatform()
    {
        // Windows paths are case-insensitive, so a differently-cased child must
        // still count as inside. Unix paths are case-sensitive, so it must not.
        string upper = Path.Combine(Base, "PROJ", "file.txt");
        Assert.Equal(
            OperatingSystem.IsWindows(),
            PermissionPathPolicy.IsUnderRoot(upper, [Proj]));
    }

    // ── PermissionPolicy.Resolve ─────────────────────────────────────────

    [Fact]
    public void Resolve_FileInsideApprovedRoot_InAskMode_IsAlways()
    {
        var result = PermissionPolicy.Resolve(
            "ListFiles", ToolCategory.File, Path.Combine(Proj, "sub", "file.txt"),
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Always, result);
    }

    [Fact]
    public void Resolve_FileOutsideApprovedRoot_InAskMode_Prompts()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, Path.Combine(Base, "Windows", "file.txt"),
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Ask, result);
    }

    [Fact]
    public void Resolve_FileInAskMode_AlwaysRespected()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, Path.Combine(Base, "Windows", "file.txt"),
            ToolPermission.Always, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Always, result);
    }

    [Fact]
    public void Resolve_FileInAskMode_NeverRespected()
    {
        var result = PermissionPolicy.Resolve(
            "ReadFile", ToolCategory.File, Path.Combine(Proj, "sub", "file.txt"),
            ToolPermission.Never, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Never, result);
    }

    [Fact]
    public void Resolve_Network_IgnoresPathRule()
    {
        // Network resources are never auto-allowed by the path rule, even "inside" a root.
        var result = PermissionPolicy.Resolve(
            "Fetch", ToolCategory.Network, Path.Combine(Proj, "file.txt"),
            ToolPermission.Ask, ToolPermission.Ask, Roots, restrictedToolNames: null);
        Assert.Equal(ToolPermission.Ask, result);
    }

    // ── Restricted (embedded-plugin) tools ──────────────────────────────

    [Fact]
    public void Resolve_RestrictedTool_ForcesAskEvenWhenAlways()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "EmbeddedTool", ToolCategory.File, Path.Combine(Base, "Windows", "file.txt"),
            ToolPermission.Always, ToolPermission.Always, Roots, restricted);
        Assert.Equal(ToolPermission.Ask, result);
    }

    [Fact]
    public void Resolve_RestrictedTool_NeverStillBlocks()
    {
        IReadOnlySet<string> restricted = new HashSet<string> { "EmbeddedTool" };
        var result = PermissionPolicy.Resolve(
            "EmbeddedTool", ToolCategory.File, Path.Combine(Proj, "file.txt"),
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
