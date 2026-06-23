using SharpMind.CUI.App;

// Single entry point, no DI container, no startup ceremony: build the app,
// run it, dispose it. This is meant to feel as close as possible to typing
// CUI.EXE at a DOS prompt — instant, with nothing to configure first.
var app = new App();
try
{
    await app.RunAsync();
}
finally
{
    await app.DisposeAsync();
}
