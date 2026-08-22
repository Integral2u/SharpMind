using System.Reflection;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Chat;

namespace SharpMind.CUI.App;

public sealed class PluginLoadResult
{
    public List<IContextCompactor> Compactors { get; } = [];
    public List<IPromptPreProcessor> PreProcessors { get; } = [];
    public List<IPromptPostProcessor> PostProcessors { get; } = [];
    public List<PluginGeneratorInfo> Generators { get; } = [];
    public List<object> Tools { get; } = [];
    public List<string> Warnings { get; } = [];
}

public sealed class PluginGeneratorInfo
{
    public required string Name { get; init; }
    public required Type BuilderType { get; init; }
    public required Type CacheBuilderType { get; init; }
}

public static class PluginLoader
{
    /// <summary>
    /// Scans every *.dll in <paramref name="directory"/> for plugin components.
    /// Failures in one assembly (bad path, load error, no qualifying types) are
    /// collected as warnings rather than aborting the whole load.
    /// </summary>
    public static PluginLoadResult LoadFrom(string directory)
    {
        var result = new PluginLoadResult();
        var state = new LoadState(result);

        if (!Directory.Exists(directory))
            return result;

        // Let plugins resolve sibling dependency DLLs from this folder and serve
        // dependencies from their own embedded resources (self-contained DLLs).
        PluginAssemblyResolver.RegisterDirectory(directory);

        foreach (var dllPath in Directory.GetFiles(directory, "*.dll"))
        {
            Assembly asm;
            try
            {
                asm = Assembly.LoadFrom(dllPath);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Failed to load '{Path.GetFileName(dllPath)}': {ex.Message}");
                continue;
            }
            PluginAssemblyResolver.RegisterEmbeddedResources(asm);
            state.Scan(asm, Path.GetFileName(dllPath));
        }

        return result;
    }

    /// <summary>
    /// Materializes plugin components out of raw assembly bytes — used for the plugin
    /// section embedded inside an SMM model container. Same discovery contract as
    /// <see cref="LoadFrom"/>.
    /// </summary>
    public static PluginLoadResult LoadFromBytes(IEnumerable<(string Name, byte[] Bytes)> assemblies)
    {
        var result = new PluginLoadResult();
        var state = new LoadState(result);

        foreach (var (name, bytes) in assemblies)
        {
            if (bytes is null || bytes.Length == 0) continue;

            Assembly asm;
            try
            {
                asm = Assembly.Load(bytes);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Failed to load embedded plugin '{name}': {ex.Message}");
                continue;
            }
            // SMM-embedded plugins carry their dependencies as embedded *.dll
            // resources inside the same bytes; serve those on demand. Note this
            // path never probes the file system or network — an embedded plugin
            // must be self-contained, and any file/network access its code makes
            // at run time goes through the session permission gate like any
            // other plugin tool.
            PluginAssemblyResolver.RegisterEmbeddedResources(asm);
            state.Scan(asm, name);
        }

        return result;
    }

    /// <summary>
    /// Per-call scanning state: shared name-deduplication across every assembly
    /// scanned by one LoadFrom/LoadFromBytes call, plus the result being filled.
    /// </summary>
    private sealed class LoadState(PluginLoadResult result)
    {
        private readonly HashSet<string> _compactorNames = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _preProcessorNames = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _postProcessorNames = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _generatorNames = new(StringComparer.OrdinalIgnoreCase);

        public void Scan(Assembly asm, string sourceName)
        {
            Type[] types;
            try
            {
                types = asm.GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                // Some types in the assembly failed to load (missing deps etc).
                // Still use whichever types DID load rather than discarding the whole DLL.
                types = [.. ex.Types.Where(t => t is not null).Select(t => t!)];
                result.Warnings.Add($"'{sourceName}' loaded with {ex.LoaderExceptions.Length} type-load warning(s).");
            }

            foreach (var type in types)
            {
                if (!type.IsClass || type.IsAbstract) continue;
                if (type.GetConstructor(Type.EmptyTypes) is null) continue;

                if (typeof(IContextCompactor).IsAssignableFrom(type))
                    TryAddCompactor(type, _compactorNames);

                if (typeof(IPromptPreProcessor).IsAssignableFrom(type))
                    TryAddPreProcessor(type, _preProcessorNames);

                if (typeof(IPromptPostProcessor).IsAssignableFrom(type))
                    TryAddPostProcessor(type, _postProcessorNames);

                TryAddGenerator(type, _generatorNames);

                // [ToolDesc]-tagged methods → tool provider
                TryAddTool(type);
            }
        }

        private void TryAddCompactor(Type type, HashSet<string> names)
        {
            try
            {
                var instance = (IContextCompactor)Activator.CreateInstance(type)!;
                if (!names.Add(instance.Name))
                {
                    result.Warnings.Add($"Compactor '{instance.Name}' from '{type.Assembly.GetName().Name}' skipped (name already registered).");
                    return;
                }
                result.Compactors.Add(instance);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Could not instantiate compactor '{type.FullName}': {ex.Message}");
            }
        }

        private void TryAddPreProcessor(Type type, HashSet<string> names)
        {
            try
            {
                var instance = (IPromptPreProcessor)Activator.CreateInstance(type)!;
                if (!names.Add(instance.Name))
                {
                    result.Warnings.Add($"Pre-processor '{instance.Name}' from '{type.Assembly.GetName().Name}' skipped (name already registered).");
                    return;
                }
                result.PreProcessors.Add(instance);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Could not instantiate pre-processor '{type.FullName}': {ex.Message}");
            }
        }

        private void TryAddPostProcessor(Type type, HashSet<string> names)
        {
            try
            {
                var instance = (IPromptPostProcessor)Activator.CreateInstance(type)!;
                if (!names.Add(instance.Name))
                {
                    result.Warnings.Add($"Post-processor '{instance.Name}' from '{type.Assembly.GetName().Name}' skipped (name already registered).");
                    return;
                }
                result.PostProcessors.Add(instance);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Could not instantiate post-processor '{type.FullName}': {ex.Message}");
            }
        }

        private void TryAddTool(Type type)
        {
            bool hasToolMethod = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                .Any(m => m.GetCustomAttributes(typeof(ToolDescAttribute), inherit: false).Length != 0);
            if (!hasToolMethod) return;

            try
            {
                var instance = Activator.CreateInstance(type);
                if (instance is not null)
                    result.Tools.Add(instance);
            }
            catch (Exception ex)
            {
                result.Warnings.Add($"Could not instantiate tool '{type.FullName}': {ex.Message}");
            }
        }

        private void TryAddGenerator(Type type, HashSet<string> names)
        {
            foreach (var iface in type.GetInterfaces())
            {
                if (!iface.IsGenericType) continue;
                if (iface.GetGenericTypeDefinition() != typeof(IGeneratorBuilder<>)) continue;

                var genericArgs = iface.GetGenericArguments();
                if (genericArgs.Length != 1) continue;

                string name = type.Name;
                if (!names.Add(name))
                {
                    result.Warnings.Add($"Generator '{name}' from '{type.Assembly.GetName().Name}' skipped (name already registered).");
                    return;
                }

                result.Generators.Add(new PluginGeneratorInfo
                {
                    Name = name,
                    BuilderType = type,
                    CacheBuilderType = genericArgs[0]
                });
                return;
            }
        }
    }
}
