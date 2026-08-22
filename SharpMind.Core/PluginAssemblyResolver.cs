using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.Loader;

namespace SharpMind.Core;

/// <summary>
/// Dependency resolution for plugin assemblies loaded by a host (folder plugins,
/// training data components, DLLs embedded inside an .SMM model). Hooks the
/// default load context so a plugin's referenced assemblies can be served either
/// from sibling DLLs in its own directory or from <c>*.dll</c> resources embedded
/// in the plugin assembly itself.
///
/// Two registration modes:
/// <list type="bullet">
/// <item><see cref="RegisterDirectory"/> — resolves <c>&lt;name&gt;.dll</c> from an
/// on-disk directory. Intended for folder-based plugin scans (chat <c>plugins/</c>,
/// the training wizard's plugins folder), where the folder is an approved root.</item>
/// <item><see cref="RegisterEmbeddedResources"/> — resolves dependency names from
/// manifest resources ending in <c>.dll</c> inside the given assembly. The primary
/// mechanism for SMM-embedded plugins, which are stored as bare assembly bytes and
/// cannot carry sibling files.</item>
/// </list>
///
/// Both modes hook <see cref="AssemblyLoadContext.Default"/>. Resolving and
/// <see cref="AppDomain.AssemblyResolve"/>; a handler returns null when it cannot
/// answer (so other resolvers still run) and registrations are idempotent.
/// </summary>
public static class PluginAssemblyResolver
{
    private static readonly Lock _sync = new();

    private sealed record DirectoryProbe(string Directory);
    private sealed record EmbeddedSource(Assembly Assembly);

    private static readonly List<DirectoryProbe> _directories = [];
    private static readonly List<EmbeddedSource> _embeddedSources = [];
    private static readonly Dictionary<string, byte[]> _embeddedCache = new(StringComparer.OrdinalIgnoreCase);
    private static volatile bool _hooked;

    /// <summary>
    /// Serves <c>&lt;name&gt;.dll</c> from <paramref name="directory"/> for any
    /// assembly the default load context fails to resolve. No-op when the
    /// directory is empty or already registered. Idempotent across calls and
    /// threads.
    /// </summary>
    public static void RegisterDirectory(string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return;

        lock (_sync)
        {
            if (_directories.Any(d => string.Equals(d.Directory, directory, StringComparison.OrdinalIgnoreCase)))
                return;
            _directories.Add(new DirectoryProbe(directory));
            Hook();
        }
    }

    /// <summary>
    /// Serves dependency assemblies from <c>*.dll</c> manifest resources embedded
    /// inside <paramref name="assembly"/>. Each resource's real assembly name is
    /// read from its own CLI metadata and used as the resolution key, so the
    /// resource file name doesn't matter (e.g. <c>libs/Newtonsoft.Json.dll</c>
    /// answers requests for the <c>Newtonsoft.Json</c> assembly). Resources that
    /// aren't managed assemblies are skipped.
    /// </summary>
    public static void RegisterEmbeddedResources(Assembly? assembly)
    {
        if (assembly is null) return;

        lock (_sync)
        {
            if (_embeddedSources.Any(s => ReferenceEquals(s.Assembly, assembly)))
                return;

            foreach (var resource in GetDllResourceNames(assembly))
            {
                byte[] bytes = ReadResource(assembly, resource);
                if (TryReadAssemblyName(bytes) is { Length: > 0 } realName)
                    _embeddedCache[realName] = bytes;
            }

            _embeddedSources.Add(new EmbeddedSource(assembly));
            Hook();
        }
    }

    /// <summary>
    /// Resolves the named assembly from any registered source, returning null
    /// when it cannot be provided (the default load context / other resolvers
    /// remain in charge). Used by the <see cref="AssemblyLoadContext.Default"/>
    /// and <see cref="AppDomain"/> resolution hooks. Exposed for tests; callers
    /// should go through the load context hooks.
    /// </summary>
    internal static Assembly? Resolve(AssemblyName request)
        => ResolveFromEmbedded(request) ?? ResolveFromDirectories(request);

    private static Assembly? ResolveFromEmbedded(AssemblyName request)
    {
        if (string.IsNullOrWhiteSpace(request.Name))
            return null;

        byte[]? bytes;
        lock (_sync)
        {
            if (!_embeddedCache.TryGetValue(request.Name, out bytes))
                return null;
        }

        try
        {
            return Assembly.Load(bytes);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Resolves <c>&lt;name&gt;.dll</c> from a registered on-disk directory, or null.</summary>
    private static Assembly? ResolveFromDirectories(AssemblyName request)
    {
        if (string.IsNullOrWhiteSpace(request.Name)) return null;

        foreach (var probe in _directories)
        {
            string path = Path.Combine(probe.Directory, request.Name + ".dll");
            if (File.Exists(path))
            {
                try { return Assembly.LoadFrom(path); }
                catch { }
            }
        }
        return null;
    }

    private static void Hook()
    {
        if (_hooked) return;
        _hooked = true;
        AppDomain.CurrentDomain.AssemblyResolve += OnAppDomainResolve;
        AssemblyLoadContext.Default.Resolving += OnContextResolving;
    }

    private static Assembly? OnAppDomainResolve(object? sender, ResolveEventArgs args)
    {
        var name = new AssemblyName(args.Name);
        return Resolve(name);
    }

    private static Assembly? OnContextResolving(AssemblyLoadContext context, AssemblyName name)
        => Resolve(name);

    internal static IReadOnlyCollection<string> RegisteredNames
    {
        get
        {
            lock (_sync)
            {
                return [.. _directories.Select(d => d.Directory), .. _embeddedCache.Keys];
            }
        }
    }

    /// <summary>For tests: clears all registrations so state doesn't leak between cases.</summary>
    internal static void Reset()
    {
        lock (_sync)
        {
            _directories.Clear();
            _embeddedSources.Clear();
            _embeddedCache.Clear();
            _hooked = false;
        }
    }

    private static IEnumerable<string> GetDllResourceNames(Assembly assembly)
    {
        try
        {
            return assembly.GetManifestResourceNames().Where(n => n.EndsWith(".dll", StringComparison.OrdinalIgnoreCase));
        }
        catch
        {
            return [];
        }
    }

    private static byte[] ReadResource(Assembly assembly, string resource)
    {
        using var stream = assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidOperationException($"Embedded resource '{resource}' not found.");
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Reads an assembly's simple name from its CLI metadata without loading it.
    /// Returns null when the bytes aren't a loadable managed assembly.
    /// </summary>
    private static string? TryReadAssemblyName(byte[] bytes)
    {
        try
        {
            using var stream = new MemoryStream(bytes, writable: false);
            using var pe = new PEReader(stream);
            if (!pe.HasMetadata) return null;
            var reader = pe.GetMetadataReader();
            if (!reader.IsAssembly) return null;
            var name = reader.GetString(reader.GetAssemblyDefinition().Name);
            return string.IsNullOrWhiteSpace(name) ? null : name;
        }
        catch
        {
            return null;
        }
    }
}