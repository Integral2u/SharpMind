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
    /// instantiates each non-abstract <see cref="IAcceleratorPlugin"/> implementation
    /// with a parameterless constructor. A missing or blank directory yields an empty list.
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
            catch (BadImageFormatException)
            {
                // Not a managed assembly — a native dependency sitting beside the plugin
                // (cudart64_12.dll, cublas64_12.dll, ...). The scan is recursive precisely
                // so a plugin can ship these in its own folder; this is expected, not a
                // load failure worth warning about on every run.
                continue;
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
    internal static void Scan(Assembly assembly, List<IAcceleratorPlugin> into, List<string> warnings)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        ArgumentNullException.ThrowIfNull(into);
        ArgumentNullException.ThrowIfNull(warnings);

        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = HandleTypeLoadFailure(ex, assembly, warnings); }

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

            // The Name getter is plugin code and may touch driver/VRAM state; a throw
            // here must not abort the scan of the rest of the assembly.
            string name;
            try { name = plugin.Name; }
            catch (Exception ex)
            {
                warnings.Add($"Accelerator plugin {type.FullName} threw from its Name property, skipped: {ex.GetBaseException().Message}");
                continue;
            }

            if (into.Any(p => string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase)))
            {
                warnings.Add($"Accelerator '{name}' ({type.FullName}) is a duplicate of an already loaded plugin, skipped.");
                continue;
            }

            into.Add(plugin);
        }
    }

    /// <summary>
    /// Turns a partial <see cref="Assembly.GetTypes"/> failure into a warning naming the
    /// distinct loader failures, and returns the types that did resolve. Split out from
    /// <see cref="Scan"/> so the warning text is unit-testable without needing a genuinely
    /// broken assembly on disk.
    /// </summary>
    internal static Type[] HandleTypeLoadFailure(ReflectionTypeLoadException ex, Assembly assembly, List<string> warnings)
    {
        string[] messages = [.. ex.LoaderExceptions.OfType<Exception>().Select(le => le.Message).Distinct()];
        warnings.Add($"Some types in '{assembly.GetName().Name}' failed to load: {string.Join("; ", messages)}");
        return [.. ex.Types.OfType<Type>()];
    }
}
