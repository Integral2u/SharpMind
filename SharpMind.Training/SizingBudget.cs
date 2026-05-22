namespace SharpMind.Training;

public record SizingBudget(
    int MaxTotalParameters = 10_000_000,
    int SampleSize = 1000,
    int StepsPerConfig = 50);
