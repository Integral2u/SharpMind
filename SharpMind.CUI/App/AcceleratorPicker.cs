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
/// When <paramref name="cpuFallbackFirst"/> is set the first (default) row is a consent sentinel,
/// "Allow CPU fallback", for the accelerator's per-tensor host fallback — the chosen accelerator
/// itself is kept, so choosing it returns the requested name unchanged.
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
        Type capabilityType,
        bool cpuFallbackFirst = false)
    {
        string requested = requestedName?.Trim() ?? "";
        var options = new List<Option>();
        if (cpuFallbackFirst)
        {
            // The accelerator itself stays selected; consenting just lets it route the
            // kernels it lacks to the host. Return the requested name so the caller can't
            // tell consent apart from "keep this accelerator".
            options.Add(new(requested, "Allow CPU fallback — keep the accelerator, run the fallen-back tensors on the CPU"));
        }
        options.Add(new(AcceleratorSelector.CpuName, "CPU — built-in, no accelerator"));
        foreach (var p in plugins)
        {
            if (!Offers(p, capabilityType)) continue;
            // In the consent dialog the requested accelerator is already the sentinel's subject;
            // re-listing it as a separate row would be a duplicate of the "keep it" choice.
            if (cpuFallbackFirst && string.Equals(p.Name, requested, StringComparison.OrdinalIgnoreCase)) continue;
            options.Add(new(p.Name, $"{p.Name} — {OneLine(p.Description)}"));
        }

        var ordered = options.ToArray();
        int defaultIndex = 0; // the consent sentinel (or the CPU, when no sentinel) is the default
        // Row 0 is the consent sentinel exactly when the caller asked for it — its Name is the
        // requested accelerator (which no plugin row re-lists) and it is not the CPU.
        int? sentinelIndex = cpuFallbackFirst ? 0 : null;

        // Layout is laid out in absolute rows (not AnchorEnd) so no two controls ever overlap:
        // the reason text has its own height (it wraps), the option list sits below it, then the
        // hint and the buttons. Wrapping the reason separately stops the long NotSupportedException
        // messages from running over — and under — the button row.
        const int width = 78;                                           // dialog width
        int innerWidth = width - 5;                                     // usable text width inside the border
        string reasonText = $"Reason: {(reason?.Replace('\n', ' ').Trim() ?? "unknown")}";
        int reasonLines = WrappedLineCount(reasonText, innerWidth);     // how tall the reason box must be
        int listRows = Math.Min(8, ordered.Length);                     // the option list (scrollable, never taller than 8)
        int row = 0;

        int dialogHeight = Math.Min(24, 2 + reasonLines + 1 + listRows + 2);
        // If the reason is very long, cap it and let the list stay visible; the reason box scrolls.
        reasonLines = Math.Min(reasonLines, Math.Max(3, dialogHeight - (listRows + 5)));

        string title = cpuFallbackFirst
            ? $"Accelerator '{(string.IsNullOrEmpty(requested) ? "selected" : requested)}' needs CPU fallback"
            : $"Accelerator '{(string.IsNullOrEmpty(requested) ? "unknown" : requested)}' can't run here";
        var dialog = new Dialog((ustring)title, width, dialogHeight);
        var prompt = new TextView { ReadOnly = true, WordWrap = true, TabStop = false, X = 1, Y = 0, Width = Dim.Fill(2), Height = reasonLines };
        prompt.Text = (ustring)reasonText;
        row = reasonLines + 1;
        var list = new ListView(ordered.Select(o => (ustring)o.Label).ToArray())
        {
            X = 1, Y = row, Width = Dim.Fill(2), Height = listRows,
            AllowsMarking = false
        };
        list.SelectedItem = defaultIndex;
        row += listRows;
        var hint = new Label("Up/Down: choose   Enter: use this engine   Esc: cancel")
        {
            X = 1, Y = row, Width = Dim.Fill(2)
        };
        var ok = new Button(IsCpu(defaultIndex, ordered) ? "Use CPU" : defaultIndex == sentinelIndex ? "Allow CPU fallback" : "Use engine") { X = 1, Y = row + 1, IsDefault = true };
        var cancel = new Button("Cancel") { X = Pos.Right(ok) + 2, Y = row + 1 };

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
            ok.Text = IsCpu(idx, ordered) ? "Use CPU" : idx == sentinelIndex ? "Allow CPU fallback" : "Use engine";
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

    /// <summary>Number of lines a string occupies when word-wrapped at <paramref name="width"/> columns
    /// (greedy, splitting on spaces; falls back to hard-breaking a too-long word). Drives the height of
    /// the reason box in <see cref="Show"/> so long "does not support ..." messages are fully visible
    /// and never overlap the controls below.</summary>
    private static int WrappedLineCount(string text, int width)
    {
        if (width < 2) return 1;
        int lines = 0, col = 0;
        foreach (var rawWord in text.Split(' '))
        {
            string word = string.IsNullOrEmpty(rawWord) ? "" : rawWord;
            int w = Math.Min(word.Length, width);
            if (lines == 0 || col == 0)
            {
                lines++; col = w;
            }
            else if (col + 1 + w <= width)
            {
                col += 1 + w;
            }
            else
            {
                lines++; col = w;
            }
        }
        return lines > 0 ? lines : 1;
    }
}
