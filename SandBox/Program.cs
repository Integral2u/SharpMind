//This class should only be used to call external functions.

//Add functions here
await Task.Run(SharpMind.Samples.Tests.PseudoLanguage.Run);
await Task.Run(SharpMind.Samples.Tests.TrainingForwardPass.Run);

Console.In.ReadLine();