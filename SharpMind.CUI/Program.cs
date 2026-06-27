using SharpMind.CUI;
using Terminal.Gui;
// Terminal.Gui owns the entire input/render loop once Application.Run is
// called - there's no manual frame loop to write here, unlike the previous
// hand-rolled console UI. Init/Run/Shutdown is the whole lifecycle.

Application.Init();
try
{
    var top = Application.Top;
    top.Add(new MainWindow());
    Application.Run();
}
finally
{
    Application.Shutdown();
}
