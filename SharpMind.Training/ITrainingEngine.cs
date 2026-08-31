using SharpMind.Data.Batching;

namespace SharpMind.Training;

/// <summary>
/// One training step's compute: forward + backward for a batch. Implementations
/// accumulate into <c>Parameter.Grad</c> and return the batch's mean loss; the
/// caller (<see cref="TrainLoop"/>) owns gradient zeroing, clipping and the
/// optimizer step. An accelerator plugin supplies its own implementation through
/// <see cref="ITrainingEngineFactory"/> so the whole step can stay resident on
/// the device — per-kernel substitution cannot do that.
/// </summary>
public interface ITrainingEngine : IDisposable
{
    /// <summary>
    /// What the engine actually runs on, for the UI: e.g. <c>"CPU"</c>, or a device string like
    /// <c>"[Cuda] GeForce GTX 1060, 6144 MB, cuBLAS 12.8"</c> (ILGPU <c>GpuDevice.Description</c>).
    /// Shown on the training-progress screen so OpenCL vs. ILGPU-CUDA vs. cuBLAS vs. CPU is always
    /// visible.
    /// </summary>
    string Description { get; }

    /// <summary>Runs forward and backward for <paramref name="batch"/>; returns the mean loss.</summary>
    float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken = default);
}
