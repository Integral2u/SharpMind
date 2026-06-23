using System.Reflection;

namespace SharpMind.CUI.App;

/// <summary>
/// Loads tool-provider classes out of external DLLs so a project can ship its
/// own tools (file readers, web calls, project-specific actions) without
/// recompiling SharpMind itself.
///
/// Contract for a tool assembly: any public, non-abstract class with a
/// parameterless constructor that has at least one public method carrying
/// <see cref="ToolDescAttribute"/> is treated as a tool provider and handed
/// to <c>AgentBuilder.WithTools</c>. This mirrors exactly what
/// <c>EchoTool</c> in the core package looks like — nothing DLL-specific is
/// required of the tool author beyond "be a normal class with [ToolDesc]
/// methods".
/// </summary>
public static class ToolAssemblyLoader
{
    /// <summary>
    /// Loads every qualifying tool class from the given assembly paths.
    /// Failures in one assembly (bad path, load error, no qualifying types)
    /// are collected as warnings rather than aborting the whole load — one
    /// broken tool DLL shouldn't prevent the rest of the session from
    /// starting.
    /// </summary>
    public static (List<object> Instances, List<string> Warnings) Load(IEnumerable<string> assemblyPaths)
    {
        var instances = new List<object>();
        var warnings = new List<string>();

        foreach (var path in assemblyPaths)
        {
            if (string.IsNullOrWhiteSpace(path)) continue;

            if (!File.Exists(path))
            {
                warnings.Add($"Tool assembly not found: {path}");
                continue;
            }

            Assembly asm;
            try
            {
                asm = Assembly.LoadFrom(path);
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to load '{path}': {ex.Message}");
                continue;
            }

            Type[] types;
            try
            {
                types = asm.GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                // Some types in the assembly failed to load (missing deps etc).
                // Still use whichever types DID load rather than discarding the whole DLL.
                types = ex.Types.Where(t => t is not null).Select(t => t!).ToArray();
                warnings.Add($"'{Path.GetFileName(path)}' loaded with {ex.LoaderExceptions.Length} type-load warning(s).");
            }

            int foundInThisAssembly = 0;
            foreach (var type in types)
            {
                if (!type.IsClass || type.IsAbstract) continue;
                if (type.GetConstructor(Type.EmptyTypes) is null) continue;

                bool hasToolMethod = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                    .Any(m => m.GetCustomAttributes(typeof(ToolDescAttribute), true).Length != 0);
                if (!hasToolMethod) continue;

                try
                {
                    var instance = Activator.CreateInstance(type);
                    if (instance is not null)
                    {
                        instances.Add(instance);
                        foundInThisAssembly++;
                    }
                }
                catch (Exception ex)
                {
                    warnings.Add($"Could not instantiate '{type.FullName}' from '{Path.GetFileName(path)}': {ex.Message}");
                }
            }

            if (foundInThisAssembly == 0)
                warnings.Add($"'{Path.GetFileName(path)}' loaded but contained no [ToolDesc]-tagged classes.");
        }

        return (instances, warnings);
    }
}
