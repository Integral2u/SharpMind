using NStack;
using SharpMind.Core.Plugins;
using Terminal.Gui;

namespace SharpMind.CUI.App;

/// <summary>
/// Modal consent dialog shown when a chosen accelerator exists but can't run on this machine
/// (e.g. the ILGPU plugin was chosen but no CUDA/OpenCL device is present). It lists the CPU
/// (guaranteed to work) plus every discovered plugin offering the relevant engine capability, so
/// the user explicitly chooses the fallback — never a silent CPU fallback, and never a hard fail
/// when the only problem is that a device is simply absent.
///
/// Returns the chosen accelerator name, or null when the user cancels (which the caller treats as
/// aborting the launch/run). A chosen CPU maps to the reserved <see cref="InferenceEngineResolver.CpuName"/>.
/// </summary>
public static class AcceleratorPicker
{
    private sealed record Option(string Name, string Label);

    public static string? Show(
        string requestedName,
        string reason,
        IReadOnlyList<IAcceleratorPlugin> plugins,
        Type capabilityType)
    {
        var options = new List<Option> { new(AcceleratorSelector.CpuName, "CPU — built-in, no accelerator") };
        foreach (var p in plugins)
        {
            if (!Offers(p, capabilityType)) continue;
            options.Add(new(p.Name, $"{p.Name} — {OneLine(p.Description)}"));
        }

        var ordered = options.ToArray();
        int defaultIndex = 0; // CPU is the guaranteed-workable default

        var dialog = new Dialog((ustring)$"Accelerator '{requestedName}' can't run here", 74, Math.Min(17, ordered.Length + 5));
        var prompt = new Label((ustring)($"Reason: {OneLine(reason)}"))
        {
            X = 1, Y = 0, Width = Dim.Fill(2), Height = 2
        };
        var list = new ListView(ordered.Select(o => (ustring)o.Label).ToArray())
        {
            X = 1, Y = 3, Width = Dim.Fill(2), Height = Math.Max(3, dialog.Frame.Height - 8),
            AllowsMarking = false
        };
        list.SelectedItem = defaultIndex;
        var hint = new Label("Up/Down: choose   Enter: use this engine   Esc: cancel")
        {
            X = 1, Y = Pos.AnchorEnd(3), Width = Dim.Fill(2)
        };
        var ok = new Button(IsCpu(defaultIndex, ordered) ? "Use CPU" : "Use engine") { X = 1, Y = Pos.AnchorEnd(1), IsDefault = true };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = Pos.AnchorEnd(1) };

        string? result = null;

        void Complete()
        {
            int idx = list.SelectedItem;
            if (idx < 0 || idx >= ordered.Length) return;
            result = ordered[idx].Name;
            Application.RequestStop();
        }

        void RefreshButton()
        {
            int idx = list.SelectedItem;
            ok.Text = IsCpu(idx, ordered) ? "Use CPU" : "Use engine";
        }

        list.SelectedItemChanged += (_) => RefreshButton();
        list.OpenSelectedItem += (_) => Complete();
        ok.Clicked += Complete;
        cancel.Clicked += () => Application.RequestStop();

        dialog.Add(prompt, list, hint, ok, cancel);
        list.SetFocus();
        Application.Run(dialog);
        return result;
    }

    private static bool IsCpu(int index, Option[] ordered) =>
        index >= 0 && index < ordered.Length && ordered[index].Name.Equals(AcceleratorSelector.CpuName, StringComparison.OrdinalIgnoreCase);

    private static bool Offers(IAcceleratorPlugin plugin, Type capabilityType)
    {
        try { return plugin.Capabilities.Any(c => capabilityType.IsInstanceOfType(c)); }
        catch { return false; }
    }

    private static string OneLine(string description)
    {
        string s = description?.Replace('\n', ' ')?.Trim() ?? "";
        return s.Length <= 48 ? s : s[..45] + "…";
    }
}
