using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpMind.Core;

namespace SharpMind.CUI.App;

/// <summary>
/// Tool methods registered on every session by default, regardless of what
/// other tools the user has configured — see <see cref="SessionLauncher"/>.
/// These exist so a model running inside the CUI can actually act through
/// the UI it's embedded in (present a choice instead of just describing one
/// in text) and answer practical questions about the machine it's running
/// on, rather than only ever talking about, never touching, the surface
/// it's actually wired into.
///
/// Deliberately instance methods bound to one <see cref="CuiToolContext"/>
/// per session — that's the shared mailbox <see cref="App"/> polls every
/// frame to know a dialog needs to be shown, and the only piece of state
/// these tools actually need.
/// </summary>
public sealed class CuiTools(CuiToolContext context)
{
    [ToolDesc("Shows the person a list of options to choose from, optionally letting them type their own answer instead. Use this whenever you want the person to pick between a small number of concrete choices, rather than asking them to type a free-form reply that you'd then have to interpret. Returns the option text the person selected, or their typed answer.")]
    public async Task<string> UIShowOptionSelection(
        [ToolDesc("A short question or instruction shown above the option list, e.g. 'Which file format would you like?'")] string prompt,
        [ToolDesc("The list of choices to present, each shown as a selectable option. Provide at least one.")] List<string> options,
        [ToolDesc("Whether to also let the person type their own answer instead of picking a listed option. Defaults to false.")] bool allowFreeText = false)
    {
        if (options is null || options.Count == 0)
            throw new ArgumentException("UIShowOptionSelection requires at least one option.");

        return await context.RequestChoiceAsync(prompt, options, allowFreeText);
    }

    [ToolDesc("Reports how much memory is currently available on the machine this session is running on, in megabytes. Use this before suggesting memory-heavy operations (e.g. loading another large model) to check whether there's realistically room for it.")]
    public string UIGetFreeMemory()
    {
        // GCMemoryInfo reflects the .NET GC's own view of available memory, which
        // tracks total physical memory reasonably closely on most systems without
        // needing any OS-specific P/Invoke call — it's the only memory figure the
        // BCL exposes directly, and it's good enough for "is there roughly enough
        // room" decisions, which is the only thing this is meant to inform.
        var info = GC.GetGCMemoryInfo();
        long totalMb = info.TotalAvailableMemoryBytes / (1024 * 1024);
        long usedByProcessMb = Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
        return $"Total available memory: ~{totalMb} MB. This process is currently using ~{usedByProcessMb} MB of it.";
    }

    [ToolDesc("Reports basic specs of the machine this session is running on: operating system, processor architecture, and logical core count. Use this to tailor advice about performance, threading, or hardware-dependent options to the actual machine in use.")]
    public string UIGetSystemSpecs()
    {
        string os = RuntimeInformation.OSDescription;
        string arch = RuntimeInformation.OSArchitecture.ToString();
        int cores = Environment.ProcessorCount;
        bool is64Bit = Environment.Is64BitOperatingSystem;
        return $"OS: {os}. Architecture: {arch} ({(is64Bit ? "64-bit" : "32-bit")}). Logical cores: {cores}.";
    }
}
