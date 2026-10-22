using System.Diagnostics;
using SharpMind.Core;

namespace SharpMind.Extensions.Tools;

/// <summary>
/// Executes common git commands and returns their output. Read-only by
/// default — the model can inspect history, status, and diffs without
/// needing shell access.
/// </summary>
public class GitTool
{
    [ToolDesc("Runs a git command in the specified repository directory and returns its output. Supports common read-only commands: status, log, diff, show, blame, branch, remote, tag.")]
    public static string Git(
        [ToolDesc("The git sub-command and arguments (e.g. 'status', 'log --oneline -10', 'diff HEAD~3', 'show abc123').")] string command,
        [ToolDesc("The repository directory. Defaults to the current working directory.")] string repository = ".")
    {
        try
        {
            string repoDir = Path.GetFullPath(repository);
            if (!Directory.Exists(repoDir))
                return $"Repository directory not found: {repository}";

            string[] tokens = command.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (tokens.Length == 0)
                return "No git command supplied.";

            // Global options can bypass the sub-command guard: -c/--config-env
            // can define an alias that runs a shell command, and --exec-path
            // can point git at arbitrary helpers. --output writes files.
            foreach (string token in tokens)
            {
                string lower = token.ToLowerInvariant();
                if (lower == "-c" || (lower.StartsWith("-c") && !lower.StartsWith("--"))
                    || lower == "--config-env" || lower.StartsWith("--config-env=")
                    || lower == "--exec-path" || lower.StartsWith("--exec-path=")
                    || lower == "--git-dir" || lower.StartsWith("--git-dir=")
                    || lower == "--work-tree" || lower.StartsWith("--work-tree=")
                    || lower == "--namespace" || lower.StartsWith("--namespace=")
                    || lower == "-c" || lower == "-C"
                    || lower == "--bare" || lower == "--no-index"
                    || lower == "--ext-diff" || lower == "--textconv"
                    || lower == "-o" || lower.StartsWith("--output"))
                {
                    return $"Git option '{token}' can bypass the read-only guard and is blocked.";
                }
            }

            string? subCommand = tokens.FirstOrDefault(t => !t.StartsWith('-'));
            if (subCommand is null)
                return "No git sub-command supplied.";
            string firstWord = subCommand.ToLowerInvariant();
            string[] blocked = ["push", "pull", "commit", "merge", "rebase", "reset", "checkout", "switch",
                                "stash", "cherry-pick", "revert", "clean", "rm", "mv", "tag", "branch",
                                "config", "gc", "prune", "update-index", "update-ref", "symbolic-ref",
                                "notes", "worktree", "submodule", "filter-branch"];

            if (Array.Exists(blocked, b => firstWord == b))
                return $"Command '{firstWord}' modifies the repository and is blocked. Use only read-only commands (status, log, diff, show, blame, remote).";

            var psi = new ProcessStartInfo("git", command)
            {
                WorkingDirectory = repoDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            using var process = Process.Start(psi);
            if (process is null)
                return "Failed to start git process.";

            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0 && string.IsNullOrWhiteSpace(stdout))
                return $"git {command} failed (exit {process.ExitCode}):\n{stderr}";

            // Truncate very long output to avoid flooding the context.
            const int maxLen = 4096;
            if (stdout.Length > maxLen)
                stdout = stdout[..maxLen] + $"\n... (truncated, {stdout.Length} chars total)";

            return string.IsNullOrWhiteSpace(stdout)
                ? $"git {command} completed with no output.\n{stderr}"
                : stdout.TrimEnd();
        }
        catch (Exception ex)
        {
            return $"Error running git: {ex.Message}";
        }
    }
}
