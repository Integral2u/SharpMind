using SharpMind.CUI.Screen;

namespace SharpMind.CUI.App;

/// <summary>
/// The settings screen: every <see cref="SessionOptions"/> field laid out as
/// a simple up/down field list, Tab/Shift-Tab between fields, Left/Right (or
/// Enter on enum fields) to cycle values, typing for text fields. Deliberately
/// plain — this is a form, not a dashboard.
/// </summary>
public sealed class OptionsScreen(SessionOptions options, string? modelPath)
{
    private enum Field
    {
        ModelPath, ProjectPath, Generator, Cache, HardwareTier, UseGpu,
        Temperature, TopK, TopP, MaxNewTokens, RepetitionPenalty,
        AgentName, AgentsEnabled, MaxAgentDepth,
        SkillFolders, ToolAssemblies, ToolsFolder,
        Launch
    }

    private static readonly Field[] FieldOrder = Enum.GetValues<Field>();
    private int _fieldIndex;
    private string? _textEditBuffer;   // non-null while editing a free-text field

    public bool LaunchRequested { get; private set; }
    public bool Cancelled { get; private set; }

    /// <summary>Called by the app after handling a launch attempt (success or failure) so the next keypress doesn't immediately relaunch.</summary>
    public void AcknowledgeLaunchRequest() => LaunchRequested = false;

    public void HandleKey(ConsoleKeyInfo key)
    {
        if (_textEditBuffer is not null)
        {
            HandleTextEditKey(key);
            return;
        }

        switch (key.Key)
        {
            case ConsoleKey.Escape: Cancelled = true; return;
            case ConsoleKey.Tab when key.Modifiers.HasFlag(ConsoleModifiers.Shift):
                _fieldIndex = (_fieldIndex - 1 + FieldOrder.Length) % FieldOrder.Length; return;
            case ConsoleKey.Tab:
            case ConsoleKey.DownArrow:
                _fieldIndex = (_fieldIndex + 1) % FieldOrder.Length; return;
            case ConsoleKey.UpArrow:
                _fieldIndex = (_fieldIndex - 1 + FieldOrder.Length) % FieldOrder.Length; return;
            case ConsoleKey.LeftArrow: CycleEnum(-1); return;
            case ConsoleKey.RightArrow: CycleEnum(1); return;
            case ConsoleKey.Enter: ActivateField(); return;
        }
    }

    private Field Current => FieldOrder[_fieldIndex];

    private void CycleEnum(int dir)
    {
        switch (Current)
        {
            case Field.Generator:
                options.Generator = Cycle(options.Generator, dir);
                break;
            case Field.Cache:
                options.Cache = Cycle(options.Cache, dir);
                break;
            case Field.HardwareTier:
                options.HardwareTier = Cycle(options.HardwareTier, dir);
                break;
            case Field.UseGpu:
                options.UseGpu = !options.UseGpu;
                break;
            case Field.Temperature:
                options.Sampling = options.Sampling with { Temperature = Clamp(options.Sampling.Temperature + dir * 0.05f, 0f, 2f) };
                break;
            case Field.TopK:
                options.Sampling = options.Sampling with { TopK = Math.Max(0, options.Sampling.TopK + dir * 5) };
                break;
            case Field.TopP:
                options.Sampling = options.Sampling with { TopP = Clamp(options.Sampling.TopP + dir * 0.05f, 0f, 1f) };
                break;
            case Field.MaxNewTokens:
                options.Generation = options.Generation with { MaxNewTokens = Math.Max(16, options.Generation.MaxNewTokens + dir * 32) };
                break;
            case Field.RepetitionPenalty:
                options.Generation = options.Generation with { RepetitionPenalty = Clamp(options.Generation.RepetitionPenalty + dir * 0.05f, 1f, 2f) };
                break;
            case Field.AgentsEnabled:
                options.AgentsEnabled = !options.AgentsEnabled;
                break;
            case Field.MaxAgentDepth:
                options.MaxAgentDepth = Math.Max(0, options.MaxAgentDepth + dir);
                break;
        }
    }

    private static float Clamp(float v, float lo, float hi) => Math.Max(lo, Math.Min(hi, v));
    private static T Cycle<T>(T current, int dir) where T : struct, Enum
    {
        var values = Enum.GetValues<T>();
        int idx = Array.IndexOf(values, current);
        idx = (idx + dir + values.Length) % values.Length;
        return values[idx];
    }

    private void ActivateField()
    {
        switch (Current)
        {
            case Field.ProjectPath: _textEditBuffer = options.ProjectPath ?? ""; break;
            case Field.AgentName: _textEditBuffer = options.AgentName; break;
            case Field.AgentsEnabled: options.AgentsEnabled = !options.AgentsEnabled; break;
            case Field.UseGpu: options.UseGpu = !options.UseGpu; break;
            case Field.SkillFolders: _textEditBuffer = string.Join(";", options.SkillFolders); break;
            case Field.ToolAssemblies: _textEditBuffer = string.Join(";", options.ToolAssemblyPaths); break;
            case Field.ToolsFolder: _textEditBuffer = options.ToolsFolder ?? ""; break;
            case Field.Launch: LaunchRequested = true; break;
        }
    }

    private void HandleTextEditKey(ConsoleKeyInfo key)
    {
        switch (key.Key)
        {
            case ConsoleKey.Enter:
                CommitTextEdit();
                _textEditBuffer = null;
                return;
            case ConsoleKey.Escape:
                _textEditBuffer = null; // discard edit
                return;
            case ConsoleKey.Backspace:
                if (_textEditBuffer!.Length > 0) _textEditBuffer = _textEditBuffer[..^1];
                return;
            default:
                if (!char.IsControl(key.KeyChar)) _textEditBuffer += key.KeyChar;
                return;
        }
    }

    private void CommitTextEdit()
    {
        var value = _textEditBuffer ?? "";
        switch (Current)
        {
            case Field.ProjectPath: options.ProjectPath = value; break;
            case Field.AgentName: options.AgentName = value; break;
            case Field.SkillFolders:
                options.SkillFolders = value.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
                break;
            case Field.ToolAssemblies:
                options.ToolAssemblyPaths = value.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).ToList();
                break;
            case Field.ToolsFolder:
                options.ToolsFolder = string.IsNullOrWhiteSpace(value) ? null : value;
                break;
        }
    }

    /// <summary>
    /// Re-checked every Draw call (not cached) so dropping a DLL into the
    /// folder while this screen is open is reflected immediately, the same
    /// way the actual launch-time scan in SessionLauncher will see it.
    /// </summary>
    private string ToolsFolderSummary()
    {
        if (string.IsNullOrWhiteSpace(options.ToolsFolder)) return "(none)";
        if (!Directory.Exists(options.ToolsFolder)) return $"{options.ToolsFolder} (folder not found)";
        int count = Directory.GetFiles(options.ToolsFolder, "*.dll").Length;
        return $"{options.ToolsFolder} ({count} dll{(count == 1 ? "" : "s")} found)";
    }

    public void Draw(ScreenBuffer buf, int x, int y, int w, int h, Theme theme)
    {
        ConsoleColor bg = theme.Background;
        ConsoleColor fg = theme.Text;
        ConsoleColor accent = theme.Accent;
        ConsoleColor label = theme.DimText;

        buf.FillRect(x, y, w, h, ' ', fg, bg);
        buf.DrawBox(x, y, w, h, accent, bg);
        buf.WriteCentered(x, y, w, " Session Options ", theme.Text, bg);

        int row = y + 2;
        int labelW = 20;

        void DrawField(Field f, string labelText, string value)
        {
            bool selected = f == Current;
            string text = f == Current && _textEditBuffer is not null ? _textEditBuffer + "_" : value;
            buf.Write(x + 2, row, labelText.PadRight(labelW), selected ? theme.SelectionFg : label, selected ? theme.SelectionBg : bg);
            buf.Write(x + 2 + labelW, row, text.Length > w - labelW - 4 ? text[..(w - labelW - 4)] : text,
                selected ? theme.SelectionFg : fg, selected ? theme.SelectionBg : bg);
            row++;
        }

        DrawField(Field.ModelPath, "Model", modelPath is not null ? Path.GetFileName(modelPath) : "(none — UIDebug mode)");
        DrawField(Field.ProjectPath, "Project path", options.ProjectPath ?? "(none)");
        row++;
        DrawField(Field.Generator, "Generator strategy", $"< {options.Generator} >");
        DrawField(Field.Cache, "KV cache strategy", $"< {options.Cache} >");
        DrawField(Field.HardwareTier, "Hardware tier", $"< {options.HardwareTier} >");
        DrawField(Field.UseGpu, "Use GPU", $"< {(options.UseGpu ? "Yes" : "No")} >");
        row++;
        DrawField(Field.Temperature, "Temperature", $"< {options.Sampling.Temperature:F2} >");
        DrawField(Field.TopK, "Top-K", $"< {options.Sampling.TopK} >");
        DrawField(Field.TopP, "Top-P", $"< {options.Sampling.TopP:F2} >");
        DrawField(Field.MaxNewTokens, "Max new tokens", $"< {options.Generation.MaxNewTokens} >");
        DrawField(Field.RepetitionPenalty, "Repetition penalty", $"< {options.Generation.RepetitionPenalty:F2} >");
        row++;
        DrawField(Field.AgentName, "Agent name", options.AgentName);
        DrawField(Field.AgentsEnabled, "Sub-agents enabled", $"< {(options.AgentsEnabled ? "Yes" : "No")} >");
        DrawField(Field.MaxAgentDepth, "Max agent depth", $"< {options.MaxAgentDepth} >");
        row++;
        DrawField(Field.SkillFolders, "Skill folders (;sep)", options.SkillFolders.Count > 0 ? string.Join(";", options.SkillFolders) : "(none)");
        DrawField(Field.ToolAssemblies, "Tool DLLs (;-sep)", options.ToolAssemblyPaths.Count > 0 ? string.Join(";", options.ToolAssemblyPaths) : "(none)");
        DrawField(Field.ToolsFolder, "Tools folder", ToolsFolderSummary());
        row++;

        bool launchSelected = Current == Field.Launch;
        buf.Write(x + 2, row, "[ Launch Session ]", launchSelected ? theme.SelectionFg : theme.Success, launchSelected ? theme.SelectionBg : bg);

        buf.Write(x + 2, y + h - 2, "Tab/Shift-Tab move   Left/Right cycle   Enter edit/activate   Esc back",
            theme.DimText, bg);
    }
}
