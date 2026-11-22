namespace SharpMind.Inference;

/// <summary>
/// Host hook that configures a generator builder before it creates a generator.
/// The builder is passed as <see cref="object"/> because sessions close their
/// builder type at runtime (for example Medusa calibration and head-file paths).
/// </summary>
public interface IGeneratorBuilderConfigurator
{
    void Configure(object builder);
}

/// <summary>Adapts a plain delegate to <see cref="IGeneratorBuilderConfigurator"/>.</summary>
public sealed class DelegateBuilderConfigurator : IGeneratorBuilderConfigurator
{
    private readonly Action<object> _configure;

    public DelegateBuilderConfigurator(Action<object> configure)
    {
        _configure = configure ?? throw new ArgumentNullException(nameof(configure));
    }

    public void Configure(object builder) => _configure(builder);
}
