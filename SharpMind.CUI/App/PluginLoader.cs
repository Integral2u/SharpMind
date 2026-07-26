using System.Reflection;
using SharpMind;
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
    public static PluginLoadResult LoadFrom(string directory)
    {
        var result = new PluginLoadResult();
        var compactorNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var preProcessorNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var postProcessorNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var generatorNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (!Directory.Exists(directory))
            return result;

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

            Type[] types;
            try
            {
                types = asm.GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                types = ex.Types.Where(t => t is not null).Select(t => t!).ToArray();
                result.Warnings.Add($"'{Path.GetFileName(dllPath)}' loaded with {ex.LoaderExceptions.Length} type-load warning(s).");
            }

            foreach (var type in types)
            {
                if (!type.IsClass || type.IsAbstract) continue;
                if (type.GetConstructor(Type.EmptyTypes) is null) continue;

                if (typeof(IContextCompactor).IsAssignableFrom(type))
                    TryAddCompactor(type, result, compactorNames);

                if (typeof(IPromptPreProcessor).IsAssignableFrom(type))
                    TryAddPreProcessor(type, result, preProcessorNames);

                if (typeof(IPromptPostProcessor).IsAssignableFrom(type))
                    TryAddPostProcessor(type, result, postProcessorNames);

                TryAddGenerator(type, result, generatorNames);

                // [ToolDesc]-tagged methods → tool provider
                TryAddTool(type, result);
            }
        }

        return result;
    }

    private static void TryAddCompactor(Type type, PluginLoadResult result, HashSet<string> names)
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

    private static void TryAddPreProcessor(Type type, PluginLoadResult result, HashSet<string> names)
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

    private static void TryAddPostProcessor(Type type, PluginLoadResult result, HashSet<string> names)
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

    private static void TryAddTool(Type type, PluginLoadResult result)
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

    private static void TryAddGenerator(Type type, PluginLoadResult result, HashSet<string> names)
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
