using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.JSInterop;
using SharpMind.Inference;
using SharpMind.Live;
using SharpMind.Live.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

var host = builder.Build();
try
{
    SharpMindEngine.SetRuntime(host.Services.GetRequiredService<IJSRuntime>());
    // Single-threaded WASM: repaint the page between decode steps instead of
    // running the whole generation as one blocking synchronous block.
    StandardGenerator.YieldBetweenTokens = true;
}
catch
{
    // Streaming logging is unavailable, but LoadModel still returns its full log on completion.
}
await host.RunAsync();
