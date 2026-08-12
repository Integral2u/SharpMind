using System.Reflection;
using SharpMind.Core;

namespace SharpMind.Tests.Core;

/// <summary>
/// Verifies <see cref="PluginAssemblyResolver"/>: dependency DLLs embedded as
/// manifest resources in a plugin are keyed by their real assembly name (read
/// from their own metadata) and served on request; on-disk sibling DLLs probe
/// from a registered directory; registration is idempotent; nothing resolves
/// when nothing is registered.
/// </summary>
public sealed class PluginAssemblyResolverTests
{
    /// <summary>The test assembly embeds SharpMind.Core.dll as a resource (see csproj).</summary>
    private static Assembly TestAssembly => typeof(PluginAssemblyResolverTests).Assembly;

    [Fact]
    public void RegisterEmbeddedResources_RegistersRealAssemblyName()
    {
        try
        {
            PluginAssemblyResolver.RegisterEmbeddedResources(TestAssembly);

            var names = PluginAssemblyResolver.RegisteredNames.ToHashSet(StringComparer.OrdinalIgnoreCase);

            // The embedded bytes are SharpMind.Core, so that — not the resource
            // file name — is the key that must be registered.
            Assert.Contains("SharpMind.Core", names);
        }
        finally
        {
            PluginAssemblyResolver.Reset();
        }
    }

    [Fact]
    public void EmbeddedResource_ResolvesRegisteredAssembly()
    {
        try
        {
            PluginAssemblyResolver.RegisterEmbeddedResources(TestAssembly);

            var asm = PluginAssemblyResolver.Resolve(new AssemblyName("SharpMind.Core"));

            Assert.NotNull(asm);
            Assert.Equal("SharpMind.Core", asm!.GetName().Name);
        }
        finally
        {
            PluginAssemblyResolver.Reset();
        }
    }

    [Fact]
    public void EmbeddedResource_DoesNotResolveUnregisteredName()
    {
        try
        {
            PluginAssemblyResolver.RegisterEmbeddedResources(TestAssembly);

            var asm = PluginAssemblyResolver.Resolve(new AssemblyName("NoSuchDependency"));

            Assert.Null(asm);
        }
        finally
        {
            PluginAssemblyResolver.Reset();
        }
    }

    [Fact]
    public void RegisterDirectory_ResolvesSiblingDll()
    {
        using var dir = new TempDirectory();
        // Probe looks for "<name>.dll" — copy under the dependency's real name.
        File.Copy(typeof(PluginAssemblyResolver).Assembly.Location, Path.Combine(dir.Path, "SharpMind.Core.dll"));

        try
        {
            PluginAssemblyResolver.RegisterDirectory(dir.Path);

            var asm = PluginAssemblyResolver.Resolve(new AssemblyName("SharpMind.Core"));

            Assert.NotNull(asm);
        }
        finally
        {
            PluginAssemblyResolver.Reset();
        }
    }

    [Fact]
    public void RegisterDirectory_IsIdempotent_AndNoOpForMissingFolder()
    {
        try
        {
            // Missing folder: silent no-op.
            PluginAssemblyResolver.RegisterDirectory(Path.Combine(Path.GetTempPath(), "does-not-exist-999"));

            using var dir = new TempDirectory();
            PluginAssemblyResolver.RegisterDirectory(dir.Path);
            PluginAssemblyResolver.RegisterDirectory(dir.Path); // duplicate must not throw

            // Registration still functional after the duplicate call.
            File.Copy(typeof(PluginAssemblyResolver).Assembly.Location, Path.Combine(dir.Path, "SharpMind.Core.dll"));
            var asm = PluginAssemblyResolver.Resolve(new AssemblyName("SharpMind.Core"));
            Assert.NotNull(asm);
        }
        finally
        {
            PluginAssemblyResolver.Reset();
        }
    }

    [Fact]
    public void NoRegistration_DoesNotResolveAnything()
    {
        var asm = PluginAssemblyResolver.Resolve(new AssemblyName("TotallyMissing.Assembly"));
        Assert.Null(asm);
    }
}
