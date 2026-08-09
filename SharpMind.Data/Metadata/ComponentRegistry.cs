using System.Globalization;
using System.Reflection;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;

namespace SharpMind.Data.Metadata;

/// <summary>Which part of the training pipeline a <see cref="ComponentDescriptor"/> plugs into.</summary>
public enum ComponentKind
{
    /// <summary>A raw corpus origin implementing <see cref="IDataSource"/>.</summary>
    Source,

    /// <summary>A document transform implementing <see cref="ICleaningStage"/>.</summary>
    Stage,
}

/// <summary>One constructor parameter of a component, plus its wizard metadata.</summary>
public sealed class ComponentParameter
{
    public required ParameterInfo Parameter { get; init; }
    public required string Name { get; init; }
    public required Type Type { get; init; }

    /// <summary>Present when this parameter should be filled by a file picker.</summary>
    public FileChooserAttribute? FileChooser { get; init; }

    /// <summary>Present when this parameter should be filled by a folder picker.</summary>
    public FolderChooserAttribute? FolderChooser { get; init; }

    public MinMaxDefaultAttribute? MinMax { get; init; }
    public DefaultValueAttribute? DefaultValue { get; init; }
    public TooltipAttribute? Tooltip { get; init; }

    /// <summary>Fixed choice set (radio group) when present.</summary>
    public string[]? Choices { get; init; }

    /// <summary>Value currently supplied by the wizard, if any.</summary>
    public string? CurrentValue { get; set; }

    public bool IsRequired => !Parameter.IsOptional;
}

/// <summary>A discoverable data component (source or stage) plus its constructor surface.</summary>
public sealed class ComponentDescriptor
{
    public required Type Type { get; init; }
    public required ComponentKind Kind { get; init; }
    public required string Name { get; init; }
    public required string Description { get; init; }
    public required IReadOnlyList<ComponentParameter> Parameters { get; init; }

    public override string ToString() => Name;
}

/// <summary>
/// Discovers and rebuilds data components — corpora and pipeline stages — from
/// constructor metadata. Scans the built-in <see cref="SharpMind.Data"/>
/// assembly plus any modules found in a user plugins folder (mirroring how the
/// CUI discovers tool assemblies). Bad plugin DLLs are collected as warnings
/// rather than blocking the scan.
/// </summary>
public static class ComponentRegistry
{
    /// <summary>The assembly holding the built-in components (this assembly).</summary>
    public static Assembly BuiltInAssembly => typeof(ComponentRegistry).Assembly;

    /// <summary>
    /// Finds the first descriptor in <paramref name="registry"/> whose type matches
    /// <paramref name="typeName"/> (assembly-qualified name, full name, or simple name).
    /// Returns null when nothing matches.
    /// </summary>
    public static ComponentDescriptor? Find(string? typeName, IEnumerable<ComponentDescriptor> registry)
    {
        if (string.IsNullOrWhiteSpace(typeName)) return null;
        foreach (var d in registry)
        {
            if (d.Type.AssemblyQualifiedName == typeName
                || d.Type.FullName == typeName
                || d.Type.Name == typeName)
                return d;
        }
        return null;
    }

    /// <summary>
    /// Scans <paramref name="assembly"/> for classes carrying
    /// <see cref="ComponentKindAttribute"/>, classified as a source
    /// (<see cref="IDataSource"/>) or a stage (<see cref="ICleaningStage"/>).
    /// Classes implementing neither are skipped. Describes the widest public
    /// constructor.
    /// </summary>
    public static List<ComponentDescriptor> Scan(Assembly assembly)
    {
        var result = new List<ComponentDescriptor>();
        if (assembly is null) return result;

        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex)
        {
            types = ex.Types.Where(t => t is not null).Select(t => t!).ToArray();
        }

        foreach (var type in types)
        {
            if (!type.IsClass || type.IsAbstract) continue;
            var attr = type.GetCustomAttribute<ComponentKindAttribute>(inherit: false);
            if (attr is null) continue;

            bool isSource = typeof(IDataSource).IsAssignableFrom(type);
            bool isStage  = typeof(ICleaningStage).IsAssignableFrom(type);
            ComponentKind kind;
            if (isSource && !isStage) kind = ComponentKind.Source;
            else if (isStage && !isSource) kind = ComponentKind.Stage;
            else continue;

            var ctor = SelectConstructor(type);
            if (ctor is null) continue;

            var parameters = ctor.GetParameters().Select(p => DescribeParameter(p)).ToList();
            result.Add(new ComponentDescriptor
            {
                Type = type,
                Kind = kind,
                Name = attr.Name,
                Description = attr.Description,
                Parameters = parameters,
            });
        }

        return result;
    }

    /// <summary>
    /// Scans <paramref name="directory"/>'s <c>*.dll</c> files and combines the
    /// result with the built-in components from <see cref="SharpMind.Data"/>.
    /// Failures in one module are collected as <paramref name="warnings"/>.
    /// </summary>
    public static List<ComponentDescriptor> ScanFolder(string directory, out List<string> warnings)
    {
        warnings = new List<string>();
        var result = Scan(typeof(ComponentRegistry).Assembly);

        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return result;

        foreach (var dllPath in Directory.GetFiles(directory, "*.dll"))
        {
            try
            {
                var asm = Assembly.LoadFrom(dllPath);
                result.AddRange(Scan(asm));
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to load plugin '{Path.GetFileName(dllPath)}': {ex.Message}");
            }
        }

        return result;
    }

    /// <summary>
    /// Constructs a component described by <paramref name="descriptor"/> from
    /// <paramref name="values"/> (parameter name → string). Missing optional
    /// parameters fall back to their declared defaults; a missing required
    /// value (or a conversion failure) throws.
    /// </summary>
    public static T Build<T>(ComponentDescriptor descriptor, IReadOnlyDictionary<string, string> values)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentNullException.ThrowIfNull(values);

        var ctor = SelectConstructor(descriptor.Type)
            ?? throw new InvalidOperationException($"No public constructor on {descriptor.Type.Name}.");

        var args = new List<object?>();
        foreach (var p in ctor.GetParameters())
        {
            if (values.TryGetValue(p.Name!, out var raw) && !string.IsNullOrWhiteSpace(raw))
            {
                args.Add(ConvertValue(raw, p.ParameterType, p.Name!));
            }
            else if (p.IsOptional)
            {
                args.Add(p.DefaultValue);
            }
            else
            {
                var dv = p.GetCustomAttribute<DefaultValueAttribute>();
                if (dv is not null)
                {
                    args.Add(ConvertValue(dv.Value, p.ParameterType, p.Name!));
                }
                else
                {
                    throw new InvalidOperationException($"Missing required value for '{p.Name}'.");
                }
            }
        }

        return (T)Activator.CreateInstance(
            descriptor.Type,
            BindingFlags.Instance | BindingFlags.Public,
            binder: null,
            args: args.ToArray(),
            culture: CultureInfo.InvariantCulture)!;
    }

    /// <summary>Widest public constructor whose parameters are all wizard-friendly.</summary>
    private static ConstructorInfo? SelectConstructor(Type type)
    {
        return type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().All(p => IsSupportedType(p.ParameterType)))
            .OrderByDescending(c => c.GetParameters().Length)
            .FirstOrDefault();
    }

    private static bool IsSupportedType(Type t)
    {
        t = Nullable.GetUnderlyingType(t) ?? t;
        return t == typeof(string)
            || t == typeof(bool)
            || t == typeof(char)
            || t == typeof(int)
            || t == typeof(long)
            || t == typeof(float)
            || t == typeof(double)
            || t == typeof(decimal)
            || t.IsEnum;
    }

    private static ComponentParameter DescribeParameter(ParameterInfo p) => new()
    {
        Name         = p.Name!,
        Type         = p.ParameterType,
        Parameter    = p,
        FileChooser  = p.GetCustomAttribute<FileChooserAttribute>(),
        FolderChooser = p.GetCustomAttribute<FolderChooserAttribute>(),
        MinMax       = p.GetCustomAttribute<MinMaxDefaultAttribute>(),
        DefaultValue = p.GetCustomAttribute<DefaultValueAttribute>(),
        Tooltip      = p.GetCustomAttribute<TooltipAttribute>(),
        Choices      = p.GetCustomAttribute<ChoicesAttribute>()?.Choices,
    };

    private static object ConvertValue(string raw, Type targetType, string name)
    {
        try
        {
            if (targetType == typeof(string)) return raw;
            if (targetType.IsEnum)            return Enum.Parse(targetType, raw, ignoreCase: true);
            if (targetType == typeof(char))   return char.Parse(raw);
            if (targetType == typeof(bool))   return bool.Parse(raw);
            if (targetType == typeof(int))    return int.Parse(raw, CultureInfo.InvariantCulture);
            if (targetType == typeof(long))   return long.Parse(raw, CultureInfo.InvariantCulture);
            if (targetType == typeof(float))  return float.Parse(raw, CultureInfo.InvariantCulture);
            if (targetType == typeof(double)) return double.Parse(raw, CultureInfo.InvariantCulture);
            if (targetType == typeof(decimal)) return decimal.Parse(raw, CultureInfo.InvariantCulture);
            return Convert.ChangeType(raw, targetType, CultureInfo.InvariantCulture);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Could not convert '{raw}' for '{name}' to {targetType.Name}.", ex);
        }
    }
}