using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;
using SharpMind.GPU;

namespace SharpMind.Samples.Tests;

public static class ProductionTraining
{
    public static async Task Run()
    {
        // 1. Tokenizer Setup
        string tokenizerPath = "tokenizer.json";
        Tokenizer tokenizer;
        
        if (File.Exists(tokenizerPath))
        {
            tokenizer = TokenizationPipeline.Load(tokenizerPath);
        }
        else
        {
            IDataSource rawSource = new FusechatSource(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\fusechat_v1\*.json");
            tokenizer = await TokenizationPipeline.TrainAndSaveAsync(rawSource, tokenizerPath);
        }

        // 2. Data Pipeline (Using new PipelineNode.From)
        IDataSource source = new FusechatSource(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\fusechat_v1\*.json");
        var pipeline = PipelineNode.From(source);
        
        var loader = new DataLoader(
            pipeline,
            tokenise: text => tokenizer.Encode(text),
            batcher: new PackingBatcher(batchSize: 8, maxSeqLen: 256, eosTokenId: tokenizer.EosId, padTokenId: tokenizer.PadId)
        );

        // 3. Model & Hardware Setup (GPU Extension)
        var modelConfig = ModelConfig.Tiny with { VocabSize = tokenizer.VocabSize };
        var sharpConfig = SharpMindConfig.Gpt;
        
        // We use a custom mapping for GPU
        var mapping = new MappingBuilder(HardwareTier.Scalar)
            .ApplyPreset(sharpConfig)
            .WithGpu()
            .Build();

        // To support this, we'd need a ModelFactory.Create with custom mapping.
        // For now, I'll keep the factory as is and assume we laer add a mapping override.
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        
        // 4. Trainer Setup
        var optimizer = new AdamW(model.Parameters(), lr: 3e-4f);
        var scheduler = new CosineScheduler(maxLr: 3e-4f, totalSteps: 1000);
        var lossFn = new CrossEntropyLoss();
        
        var trainer = new Trainer(model, loader, optimizer, scheduler, lossFn);
        var evaluator = new Evaluator(model, lossFn);

        // 5. Training Loop
        Console.WriteLine("Starting Production Training on GPU...");
        await trainer.TrainAsync(totalSteps: 1000);
        
        Console.WriteLine("Training complete.");
    }
}
