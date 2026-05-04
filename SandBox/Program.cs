//This class should only be used to call external functions.

//Add functions here
//await Task.Run(SharpMind.Samples.Tests.PseudoLanguage.Run);
//await Task.Run(SharpMind.Samples.Tests.TrainingForwardPass.Run);
//await Task.Run(SharpMind.Samples.Tests.FullTraining.Run);
await Task.Run(()=>SharpMind.Samples.Tests.RealDataTraining.RunFusechat(32));
//await Task.Run(()=>SharpMind.Samples.Tests.RealDataTraining.RunParquet(128));

Console.In.ReadLine();