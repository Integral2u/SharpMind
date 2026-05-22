using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Training.Distillation;

/// <summary>
/// Distillation trainer — trains student while frozen teacher provides soft targets.
/// </summary>
public class DistillationTrainer(
    Transformer teacher,
    Transformer student,
    float temperature = 4f,
    float alpha = 0.5f)
{
    private readonly Transformer _teacher = teacher;
    private readonly Transformer _student = student;
    private readonly float _temperature = temperature;
    private readonly float _alpha = alpha;

    /// <summary>
    /// One distillation training step.
    /// </summary>
    public float TrainStep(Tensor<int> inputIds, Tensor<int> targetIds)
    {
        // Get teacher predictions (frozen)
        using var teacherLogits = _teacher.Forward(inputIds);

        // Get student predictions (trainable)
        using var studentLogits = _student.Forward(inputIds);

        // Compute distillation loss
        float loss = DistillationKernels.ComputeLoss(
            studentLogits,
            teacherLogits,
            targetIds,
            _temperature,
            _alpha);

        return loss;
    }

    /// <summary>
    /// Train with feature-level distillation.
    /// </summary>
    public float TrainStepWithFeatures(
        Tensor<int> inputIds,
        Tensor<int> targetIds,
        Tensor<float> studentFeatures,
        Tensor<float> teacherFeatures,
        Tensor<float> featureAdapter)
    {
        using var teacherLogits = _teacher.Forward(inputIds);
        using var studentLogits = _student.Forward(inputIds);

        float logitsLoss = DistillationKernels.ComputeLoss(
            studentLogits, teacherLogits, targetIds, _temperature, _alpha);

        float featureLoss = DistillationKernels.FeatureDistillationLoss(
            studentFeatures, teacherFeatures, featureAdapter);

        return logitsLoss + 0.5f * featureLoss;
    }
}