using SharpMind.Tokenization;
using System;
using System.IO;
using System.Linq;

//SandBox.LlamaTest.WeightValidation();
//await SandBox.LlamaTest.CompareLogits();
await SharpMind.Samples.Examples.InteractiveChat.RunAsync();

return;
var tokenizer = Tokenizer.FromFile(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-fp16.json");
Console.WriteLine($"Token 198: '{tokenizer.IdToToken(198)}'");

for(int i=0; i<1000; i++) {
    var tok = tokenizer.IdToToken(i);
    if (tok.Contains("\n")) Console.WriteLine($"Token {i}: '{tok}'");
}
