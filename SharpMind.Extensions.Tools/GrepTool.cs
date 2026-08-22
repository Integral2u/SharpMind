using System.Text.RegularExpressions;
using SharpMind.Core;

namespace SharpMind.Extensions.Tools;

/// <summary>
/// Searches file contents by pattern (regex or literal) across a directory tree.
/// Returns matching lines with file paths and line numbers — a built-in
/// alternative to external grep/ripgrep for models that need to search code.
/// </summary>
public class GrepTool
{
    [ToolDesc("Searches file contents for a pattern and returns matching lines with file paths and line numbers. Supports regex syntax.")]
    public static string Grep(
        [ToolDesc("The search pattern (regex or literal text).")] string pattern,
        [ToolDesc("The directory to search in. Defaults to the current working directory.")] string directory = ".",
        [ToolDesc("File extension filter (e.g. '.cs', '.txt'). Leave empty to search all files.")] string? extension = null,
        [ToolDesc("Maximum number of results to return. Defaults to 50.")] int maxResults = 50)
    {
        try
        {
            string searchDir = Path.GetFullPath(directory);
            if (!Directory.Exists(searchDir))
                return $"Directory not found: {directory}";

            string searchPattern = extension is { Length: > 0 }
                ? $"*{extension}"
                : "*.*";

            var regex = new Regex(pattern, RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var results = new List<string>();
            int matchCount = 0;

            foreach (var file in Directory.EnumerateFiles(searchDir, searchPattern, SearchOption.AllDirectories))
            {
                // Skip binary-looking files and common non-text directories.
                string fileName = Path.GetFileName(file);
                if (fileName is ".git" or "node_modules" or "bin" or "obj" or ".vs" or "packages")
                    continue;

                try
                {
                    int lineNum = 0;
                    foreach (var line in File.ReadLines(file))
                    {
                        lineNum++;
                        if (regex.IsMatch(line))
                        {
                            matchCount++;
                            if (results.Count < maxResults)
                            {
                                string relativePath = Path.GetRelativePath(searchDir, file);
                                results.Add($"{relativePath}:{lineNum}: {line.TrimEnd()}");
                            }
                        }
                    }
                }
                catch
                {
                    // Skip files we can't read (permissions, encoding, etc).
                }

                if (matchCount >= maxResults * 2)
                    break;
            }

            if (results.Count == 0)
                return $"No matches found for '{pattern}' in {searchDir}.";

            string header = matchCount > maxResults
                ? $"Showing {results.Count} of {matchCount} matches (limit: {maxResults}):\n"
                : $"{matchCount} match(es):\n";

            return header + string.Join("\n", results);
        }
        catch (Exception ex)
        {
            return $"Error searching: {ex.Message}";
        }
    }
}
