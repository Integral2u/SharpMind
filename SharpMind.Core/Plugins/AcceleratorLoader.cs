using System.Reflection;

namespace SharpMind.Core.Plugins;

/// <summary>
/// Finds <see cref="IAcceleratorPlugin"/> implementations in a plugins folder.
/// The scan is recursive so a plugin can ship in its own sub-folder with its
/// native dependencies beside it. Failures are collected as warnings — one bad
/// DLL never aborts the load.
/// </summary>
public static class AcceleratorLoader
{
    /// <summary>
    /// Loads every <c>*.dll</c> under <paramref name="directory"/> (recursively) and
    /// instantiates each public, non-abstract <see cref="IAcceleratorPlugin"/> with a
    /// parameterless constructor. A missing or blank directory yields an empty list.
    /// </summary>
    public static IReadOnlyList<IAcceleratorPlugin> LoadFrom(string? directory, out List<string> warnings)
    {
        warnings = [];
        var plugins = new List<IAcceleratorPlugin>();
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return plugins;

        PluginAssemblyResolver.RegisterDirectory(directory);

        foreach (var dllPath in Directory.EnumerateFiles(directory, "*.dll", SearchOption.AllDirectories))
        {
            Assembly asm;
            try
            {
                // A plugin's own folder is where its sibling dependencies live.
                PluginAssemblyResolver.RegisterDirectory(Path.GetDirectoryName(dllPath));
                asm = Assembly.LoadFrom(dllPath);
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to load '{Path.GetFileName(dllPath)}': {ex.Message}");
                continue;
            }
            PluginAssemblyResolver.RegisterEmbeddedResources(asm);
            Scan(asm, plugins, warnings);
        }

        return plugins;
    }

    /// <summary>
    /// Adds the accelerator plugins declared in <paramref name="assembly"/> to
    /// <paramref name="into"/>. Types without a parameterless constructor, types
    /// whose constructor throws, and names already present (case-insensitive) are
    /// skipped with a warning.
    /// </summary>
    public static void Scan(Assembly assembly, List<IAcceleratorPlugin> into, List<string> warnings)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        ArgumentNullException.ThrowIfNull(into);
        ArgumentNullException.ThrowIfNull(warnings);

        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.OfType<Type>().ToArray(); }

        foreach (var type in types)
        {
            if (!type.IsClass || type.IsAbstract || !typeof(IAcceleratorPlugin).IsAssignableFrom(type))
                continue;

            if (type.GetConstructor(Type.EmptyTypes) is null)
            {
                warnings.Add($"Accelerator plugin {type.FullName} has no parameterless constructor, skipped.");
                continue;
            }

            IAcceleratorPlugin plugin;
            try { plugin = (IAcceleratorPlugin)Activator.CreateInstance(type)!; }
            catch (Exception ex)
            {
                warnings.Add($"Accelerator plugin {type.FullName} failed to construct: {ex.GetBaseException().Message}");
                continue;
            }

            if (into.Any(p => string.Equals(p.Name, plugin.Name, StringComparison.OrdinalIgnoreCase)))
            {
                warnings.Add($"Accelerator '{plugin.Name}' ({type.FullName}) is a duplicate of an already loaded plugin, skipped.");
                continue;
            }

            into.Add(plugin);
        }
    }

    /// <summary>All capabilities of type <typeparamref name="T"/> across <paramref name="plugins"/>, in plugin order.</summary>
    public static IEnumerable<T> Capabilities<T>(this IEnumerable<IAcceleratorPlugin> plugins)
        => plugins.SelectMany(p => p.Capabilities).OfType<T>();
}
