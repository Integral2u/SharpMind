namespace SharpMind.Training.Schedulers;

/// <summary>
/// Computes learning rate for a given step. The scheduler is stateless —
/// it computes the rate from the step number rather than tracking it internally.
/// Call <see cref="GetLr"/> then assign the result to the optimizer's
/// <c>LearningRate</c> property before each <c>Update()</c>.
/// </summary>
public interface IScheduler
{
    float GetLr(int step);
}
