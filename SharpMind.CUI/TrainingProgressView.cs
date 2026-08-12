using System.Diagnostics;
using NStack;
using SharpMind.CUI.App;
using SharpMind.Training;
using Terminal.Gui;

namespace SharpMind.CUI;

/// <summary>
/// The training progress screen: a progress bar, a scrolling status log, and a
/// Cancel button that interrupts the background running task. On completion
/// (interrupted or finished) it flips to a results panel offering actions:
/// "Browse to test" the exported model, "Clean checkpoints" to delete the
/// retained checkpoint directories, or "Back". All progress is marshalled onto
/// the UI thread via <c>Application.MainLoop.Invoke</c>.
/// </summary>
public sealed class TrainingProgressView : View
{
    private readonly AppSettings _settings;
    private readonly TrainJobSettings _job;
    private readonly Action<string> _onBrowse;
    private readonly Action _onBack;

    private readonly CancellationTokenSource _cts = new();
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly List<string> _log = [];
    private readonly Label _statusLabel;
    private readonly Label _progressLabel;
    private readonly Label _memoryLabel;
    private readonly TextView _logView;
    private readonly Button _cancelButton;
    private readonly Button _browseButton;
    private readonly Button _cleanButton;
    private readonly Button _backButton;
    private TrainRunResult? _result;
    private string _etaText = "";

    public TrainingProgressView(
        AppSettings settings,
        TrainJobSettings job,
        Action<string> browseModel,
        Action onBack)
    {
        _settings = settings;
        _job = job;
        _onBrowse = browseModel;
        _onBack = onBack;

        var frame = new FrameView("Training") { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(1) };

        _statusLabel = new Label("Preparing…") { X = 1, Y = 0, Width = Dim.Fill(2) };
        _progressLabel = new Label("[          ]  0.00%  step 0/0") { X = 1, Y = 1, Width = Dim.Fill(2) };
        _memoryLabel = new Label("") { X = 1, Y = 2, Width = Dim.Fill(2) };
        _logView = new TextView { X = 1, Y = 4, Width = Dim.Fill(2), Height = Dim.Fill(5), ReadOnly = true };
        _cancelButton = new Button("Cancel") { X = 1, Y = Pos.AnchorEnd(1) };
        _cancelButton.Clicked += () => _cts.Cancel();
        _browseButton = new Button("Browse to test…") { X = 1, Y = Pos.AnchorEnd(1), Visible = false };
        _browseButton.Clicked += () => { if (_result?.ExportPath is { } p) _onBrowse(p); };
        _cleanButton = new Button("Clean checkpoints") { X = Pos.Right(_browseButton) + 2, Y = Pos.AnchorEnd(1), Visible = false };
        _cleanButton.Clicked += CleanCheckpoints;
        _backButton = new Button("Back") { X = Pos.Right(_cleanButton) + 2, Y = Pos.AnchorEnd(1), Visible = false };
        _backButton.Clicked += () => _onBack();

        frame.Add(_statusLabel, _progressLabel, _memoryLabel, _logView, _cancelButton, _browseButton, _cleanButton, _backButton);
        Add(frame);

        // Kick off training on a background task; callbacks marshal to UI.
        var status = new Progress<string>(s => Application.MainLoop.Invoke(() => Log(s)));
        var progress = new Progress<float>(p => Application.MainLoop.Invoke(() => SetProgress(p)));
        _ = Task.Run(() => TrainRunner.RunAsync(
                job,
                _settings.PluginsFolder ?? "",
                status,
                progress,
                onStep: r => Application.MainLoop.Invoke(() => OnStep(r)),
                cancellationToken: _cts.Token))
            .ContinueWith(t => Application.MainLoop.Invoke(() => OnComplete(t.Result)),
                TaskScheduler.Default);
    }

    private void OnStep(TrainStepResult r)
    {
        double secPerStep = _stopwatch.Elapsed.TotalSeconds / Math.Max(r.Step, 1);
        int remaining = Math.Max(_job.TotalSteps - r.Step, 0);
        _etaText = remaining > 0
            ? $"~{FormatDuration(TimeSpan.FromSeconds(secPerStep * remaining))} left"
            : "done";
        Log($"step {r.Step,5}/{_job.TotalSteps}: loss = {r.Loss:F4}  gradNorm = {r.GradNorm:F3}  {r.StepTime.TotalSeconds:F1}s");
    }

    private void Log(string line)
    {
        _log.Add(line);
        if (_log.Count > 200) _log.RemoveRange(0, _log.Count - 200);
        _logView.Text = string.Join("\n", _log);
        _logView.MoveEnd();
        _statusLabel.Text = line;
        SetNeedsDisplay();
    }

    private void SetProgress(float p)
    {
        int ticks = (int)(p * 10);
        string eta = string.IsNullOrEmpty(_etaText) ? "" : $"  {_etaText}";
        _progressLabel.Text = $"[{new string('#', ticks)}{new string(' ', 10 - ticks)}]  {p * 100:F2}%  step {CurrentStep(p)}/{_job.TotalSteps}{eta}";
        UpdateMemory();
        SetNeedsDisplay();
    }

    /// <summary>
    /// The overall 0..1 figure is shifted by TrainRunner (0.12 prep + 0.78 ×
    /// loop progress). Reverse the mapping so the visible step tracks the run.
    /// </summary>
    private int CurrentStep(float overall)
    {
        int step = (int)Math.Round((overall - 0.12f) / 0.78f * _job.TotalSteps);
        return Math.Clamp(step, 0, _job.TotalSteps);
    }

    private void UpdateMemory()
    {
        try
        {
            var process = Process.GetCurrentProcess();
            long used = process.WorkingSet64;
            long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long free = Math.Max(0, total - used);
            _memoryLabel.Text = $"mem  used {FormatBytes(used)}  free {FormatBytes(free)}  ({FormatBytes(total)} total)";
        }
        catch
        {
            _memoryLabel.Text = "";
        }
        SetNeedsDisplay();
    }

    private static string FormatBytes(long bytes)
        => bytes switch
        {
            >= 1L << 30 => $"{bytes / (double)(1L << 30):F1} GB",
            >= 1L << 20 => $"{bytes / (double)(1L << 20):F0} MB",
            _           => $"{bytes} B",
        };

    private static string FormatDuration(TimeSpan t)
    {
        if (t.TotalHours >= 1)
            return $"{(int)t.TotalHours}h {t.Minutes}m {t.Seconds}s";
        if (t.TotalMinutes >= 1)
            return $"{(int)t.TotalMinutes}m {t.Seconds}s";
        return $"{t.Seconds}s";
    }

    private void OnComplete(TrainRunResult result)
    {
        _stopwatch.Stop();
        TimeSpan total = _stopwatch.Elapsed;
        _result = result;
        _statusLabel.Text = result.Success
            ? $"Training done — completed {result.FinalStep} steps in {FormatDuration(total)}. Exported: {result.ExportPath}"
            : result.Error == "cancelled"
                ? $"Training interrupted after {FormatDuration(total)}."
                : $"Training failed after {FormatDuration(total)}: {result.Error}";
        _cancelButton.Visible = false;
        _browseButton.Visible = result.Success;
        _cleanButton.Visible = _job.CheckpointDir is not null && Directory.Exists(_job.CheckpointDir);
        _backButton.Visible = true;
        SetNeedsDisplay();
    }

    private void CleanCheckpoints()
    {
        var dir = _job.CheckpointDir;
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)) return;
        if (MessageBox.Query("Clean checkpoints", $"Delete all checkpoints under:\n{dir}?", "Yes", "No") != 0) return;
        try
        {
            foreach (var sub in Directory.GetDirectories(dir))
                Directory.Delete(sub, recursive: true);
            _cleanButton.Visible = false;
            _statusLabel.Text = "Checkpoints cleaned.";
            SetNeedsDisplay();
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Clean checkpoints", ex.Message, "OK");
        }
    }
}