using SharpMind.Tests.Core;

[assembly: CollectionBehavior(DisableTestParallelization = false)]

namespace SharpMind.Tests.Core;

[CollectionDefinition("Non-Parallel", DisableParallelization = true)]
public sealed class NonParallelCollectionDefinition { }
