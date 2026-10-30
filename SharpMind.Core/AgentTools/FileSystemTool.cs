using System.Text;

namespace SharpMind.Core.AgentTools;

/// <summary>
/// Project-scoped file tools. Every path is resolved inside the project root, following
/// symlinks, so a link cannot reach outside it. Reads are line-numbered and capped; edits
/// replace exactly one occurrence.
/// </summary>
public class FileSystemTool(string projectRoot) : IFileToolService
{
    private readonly string _projectRoot = ResolveRoot(projectRoot);

    /// <summary>Lines returned by ReadFile when no limit is given.</summary>
    public const int DefaultReadLines = 200;

    /// <summary>Hits returned by Search at most.</summary>
    public const int MaxSearchHits = 50;

    /// <summary>
    /// Path comparison follows the OS: Windows and macOS file systems are case-insensitive by
    /// default, Linux is case-sensitive.
    /// </summary>
    internal static StringComparison PathComparison =>
        OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;

    private static string ResolveRoot(string root)
    {
        string full = Path.GetFullPath(root);
        return ResolveLinks(full);
    }

    /// <summary>The path with every symlink in it resolved, so containment is checked on real locations.</summary>
    internal static string ResolveLinks(string path)
    {
        try
        {
            // A symlink to a directory and one to a file are different FileSystemInfo types, so
            // check the link itself for either. Only an existing entry can be a link.
            FileSystemInfo entry = Directory.Exists(path) ? new DirectoryInfo(path) : new FileInfo(path);
            if (entry.LinkTarget is not null)
            {
                var target = entry.ResolveLinkTarget(returnFinalTarget: true);
                if (target is not null) return target.FullName;
            }
        }
        catch (IOException) { }
        return path;
    }

    /// <summary>
    /// Resolves <paramref name="relativePath"/> under the project root. The nearest existing
    /// ancestor is resolved through symlinks, so a link anywhere on the path is judged by its
    /// real target. Throws if the result is outside the root.
    /// </summary>
    internal string ResolvePath(string relativePath)
    {
        string lexical = Path.GetFullPath(Path.Combine(_projectRoot, relativePath));
        string real = ResolveExisting(lexical);

        if (!IsInside(real))
            throw new UnauthorizedAccessException($"Access denied: Path {relativePath} is outside the project root.");
        return lexical;
    }

    /// <summary>
    /// Resolves every symlink along the path, one component at a time from the root down.
    /// Each component is checked as it is appended, so a link at any depth, including one
    /// that is the parent of a not-yet-existing file, is followed to its real target.
    /// </summary>
    private static string ResolveExisting(string path)
    {
        string root = Path.GetPathRoot(path) ?? "";
        string resolved = root;
        string rest = path[root.Length..];
        foreach (string part in rest.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar,
                     StringSplitOptions.RemoveEmptyEntries))
        {
            resolved = ResolveLinks(Path.Combine(resolved, part));
        }
        return resolved;
    }

    private bool IsInside(string fullPath)
    {
        if (string.Equals(fullPath, _projectRoot, PathComparison)) return true;
        string prefix = _projectRoot.EndsWith(Path.DirectorySeparatorChar) ? _projectRoot : _projectRoot + Path.DirectorySeparatorChar;
        return fullPath.StartsWith(prefix, PathComparison);
    }

    [ToolDesc("Lists the files and folders in a directory of the project. Returns one entry per line, folders marked [DIR]. Use '.' for the project root.")]
    public string ListFiles([ToolDesc("Directory path relative to the project root, e.g. '.' or 'src'.")] string relativePath = ".")
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!Directory.Exists(path)) return $"Directory not found: {relativePath}";

            var entries = Directory.GetFileSystemEntries(path).OrderBy(e => e, StringComparer.OrdinalIgnoreCase).ToArray();
            if (entries.Length == 0) return "Directory is empty.";

            var sb = new StringBuilder();
            foreach (var entry in entries)
            {
                string name = Path.GetFileName(entry);
                bool isDir = Directory.Exists(entry);
                sb.AppendLine($"{(isDir ? "[DIR]" : "[FILE]")} {name}");
            }
            return sb.ToString();
        }
        catch (Exception ex) { return $"Error listing files: {ex.Message}"; }
    }

    [ToolDesc("Reads a text file of the project, returning numbered lines. Returns the first 200 lines by default; use offset (first line, 1-based) and limit (number of lines) to read further. Tells you when more lines remain.")]
    public string ReadFile(
        [ToolDesc("File path relative to the project root.")] string relativePath,
        [ToolDesc("First line to return, 1-based. Default 1.")] int offset = 1,
        [ToolDesc("Number of lines to return. Default 200.")] int limit = DefaultReadLines)
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!File.Exists(path)) return $"File not found: {relativePath}";
            if (offset < 1) offset = 1;
            if (limit < 1) limit = DefaultReadLines;

            var lines = File.ReadAllLines(path);
            if (lines.Length == 0) return "(empty file)";
            if (offset > lines.Length) return $"The file has {lines.Length} lines; offset {offset} is past the end.";

            int last = Math.Min(lines.Length, offset - 1 + limit);
            var sb = new StringBuilder();
            for (int i = offset - 1; i < last; i++)
                sb.Append(i + 1).Append(" | ").AppendLine(lines[i]);
            if (last < lines.Length)
                sb.Append($"(showing lines {offset}-{last} of {lines.Length}; call again with offset={last + 1} for more)");
            return sb.ToString().TrimEnd();
        }
        catch (Exception ex) { return $"Error reading file: {ex.Message}"; }
    }

    [ToolDesc("Searches the text of project files for a literal string, case-insensitively. Returns up to 50 matches as path:line: text. Use it to find where something is before reading or editing.")]
    public string Search(
        [ToolDesc("Text to find. Matched literally, ignoring case.")] string query,
        [ToolDesc("Folder to search, relative to the project root. Default '.'.")] string relativePath = ".")
    {
        try
        {
            if (string.IsNullOrEmpty(query)) return "Error: the search text is empty.";
            string root = ResolvePath(relativePath);
            if (!Directory.Exists(root)) return $"Directory not found: {relativePath}";

            var hits = new List<string>();
            foreach (string file in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
            {
                if (hits.Count >= MaxSearchHits) break;
                if (IsIgnoredPath(file)) continue;
                string[] lines;
                try { lines = File.ReadAllLines(file); }
                catch (IOException) { continue; }
                catch (UnauthorizedAccessException) { continue; }

                for (int i = 0; i < lines.Length && hits.Count < MaxSearchHits; i++)
                {
                    if (lines[i].Contains(query, StringComparison.OrdinalIgnoreCase))
                        hits.Add($"{Path.GetRelativePath(_projectRoot, file)}:{i + 1}: {lines[i].Trim()}");
                }
            }

            if (hits.Count == 0) return $"No matches for '{query}'.";
            string text = string.Join("\n", hits);
            return hits.Count >= MaxSearchHits ? text + $"\n(stopped at {MaxSearchHits} matches; narrow the query or the folder)" : text;
        }
        catch (Exception ex) { return $"Error searching files: {ex.Message}"; }
    }

    private static bool IsIgnoredPath(string path)
    {
        foreach (string part in path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
            if (part is ".git" or "bin" or "obj" or "node_modules")
                return true;
        return false;
    }

    [ToolDesc("Writes text to a file of the project, creating it and any folders if needed, and replacing the whole file if it exists. Reports whether the file was created or overwritten.")]
    public string WriteFile([ToolDesc("File path relative to the project root.")] string relativePath, [ToolDesc("The complete text to write.")] string content)
    {
        try
        {
            string path = ResolvePath(relativePath);
            bool existed = File.Exists(path);
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.WriteAllText(path, content);
            return existed
                ? $"Overwrote {relativePath} ({content.Length} characters)."
                : $"Created {relativePath} ({content.Length} characters).";
        }
        catch (Exception ex) { return $"Error writing file: {ex.Message}"; }
    }

    [ToolDesc("Adds text to the end of a file of the project, creating the file if needed.")]
    public string AppendToFile([ToolDesc("File path relative to the project root.")] string relativePath, [ToolDesc("The text to add at the end.")] string content)
    {
        try
        {
            string path = ResolvePath(relativePath);
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.AppendAllText(path, content);
            return $"Appended {content.Length} characters to {relativePath}.";
        }
        catch (Exception ex) { return $"Error appending to file: {ex.Message}"; }
    }

    [ToolDesc("Changes one exact piece of text in a file. old_text must appear exactly once in the file, including whitespace and line breaks; the edit is refused otherwise. Read the file first to copy old_text exactly.")]
    public string Edit(
        [ToolDesc("File path relative to the project root.")] string relativePath,
        [ToolDesc("The exact text to replace. It must occur exactly once.")] string oldText,
        [ToolDesc("The text to put in its place.")] string newText)
    {
        try
        {
            if (string.IsNullOrEmpty(oldText)) return "Error: old_text is empty.";
            string path = ResolvePath(relativePath);
            if (!File.Exists(path)) return $"File not found: {relativePath}";

            string text = File.ReadAllText(path);
            int first = text.IndexOf(oldText, StringComparison.Ordinal);
            if (first < 0) return $"No match: old_text was not found in {relativePath}. Read the file and copy the text exactly.";
            if (text.IndexOf(oldText, first + oldText.Length, StringComparison.Ordinal) >= 0)
                return $"Not changed: old_text occurs more than once in {relativePath}. Include more surrounding text so it is unique.";

            File.WriteAllText(path, text.Remove(first, oldText.Length).Insert(first, newText));
            return $"Edited {relativePath}.";
        }
        catch (Exception ex) { return $"Error editing file: {ex.Message}"; }
    }

    [ToolDesc("Deletes one file of the project. Folders are not deleted.")]
    public string DeleteFile([ToolDesc("File path relative to the project root.")] string relativePath)
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!File.Exists(path)) return $"File not found: {relativePath}";
            File.Delete(path);
            return $"Deleted {relativePath}.";
        }
        catch (Exception ex) { return $"Error deleting file: {ex.Message}"; }
    }

    [ToolDesc("Creates an empty file in the project, with any missing folders. Fails if the file already exists.")]
    public string CreateFile([ToolDesc("File path relative to the project root.")] string relativePath)
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (File.Exists(path)) return $"Not created: {relativePath} already exists.";
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.WriteAllText(path, string.Empty);
            return $"Created empty file {relativePath}.";
        }
        catch (Exception ex) { return $"Error creating file: {ex.Message}"; }
    }
}
