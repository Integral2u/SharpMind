using SharpMind.Core.Training;
using SharpMind.Training;

namespace SharpMind.GPU;

/// <summary>
/// Training capability of <see cref="CudaAcceleratorPlugin"/>.
///
/// Two conversions happen here and nowhere else. First, the device is acquired lazily:
/// <c>IAcceleratorPlugin</c>'s lifetime contract says a plugin is constructed on every
/// plugins-folder scan — the training wizard scans each time it opens — and is never disposed,
/// so a CUDA context must not be taken in a plugin constructor. It is taken here, where the
/// returned engine owns it.
///
/// Second, <see cref="GpuBackpropEngine"/> reports an unsupported model by throwing
/// <see cref="NotSupportedException"/>, while this interface reports it by returning null with a
/// reason. Letting that exception escape would reach the user as a bare stack trace instead of
/// "this accelerator cannot train that model, because …".
/// </summary>
public sealed class GpuTrainingEngineFactory : ITrainingEngineFactory
{
    public ITrainingEngine? TryCreate(TrainingEngineContext context, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(context);

        GpuDevice? device = GpuDevice.TryCreate(out reason);
        if (device is null) return null;

        try
        {
            var engine = new GpuBackpropEngine(
                device,
                context.Model,
                context.Parameters,
                context.Config,
                context.BatchSize,
                context.SeqLen,
                context.IgnoreId,
                context.LabelSmoothing);

            reason = null;
            device = null;          // ownership handed to the engine
            return engine;
        }
        catch (NotSupportedException ex)
        {
            // The engine names the exact shape it refuses (LayerNorm, MoE, dense FFN,
            // quantisation-aware training, …). That message is the whole value here.
            reason = ex.Message;
            return null;
        }
        finally
        {
            device?.Dispose();      // only runs when we did not hand it over
        }
    }
}
