using SharpMind;

namespace SharpMind.AgentTools;

public class FileSystemTool
{
    private readonly string _projectRoot;

    public FileSystemTool(string projectRoot)
    {
        _projectRoot = Path.GetFullPath(projectRoot);
    }

    private string ResolvePath(string relativePath)
    {
        string fullPath = Path.GetFullPath(Path.Combine(_projectRoot, relativePath));
        if (!fullPath.StartsWith(_projectRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException($"Access denied: Path {relativePath} is outside the project root.");
        }
        return fullPath;
    }

    [ToolDesc("Lists files and folders in a directory relative to the project root.")]
    public string ListFiles([ToolDesc("The relative path to the directory. Use '.' for root.")] string relativePath = ".")
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!Directory.Exists(path)) return $"Directory not found: {relativePath}";

            var entries = Directory.GetFileSystemEntries(path);
            if (entries.Length == 0) return "Directory is empty.";

            var sb = new System.Text.StringBuilder();
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

    [ToolDesc("Reads the content of a file relative to the project root.")]
    public string ReadFile([ToolDesc("The relative path to the file.")] string relativePath)
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!File.Exists(path)) return $"File not found: {relativePath}";
            return File.ReadAllText(path);
        }
        catch (Exception ex) { return $"Error reading file: {ex.Message}"; }
    }

    [ToolDesc("Writes content to a file relative to the project root. Overwrites existing files.")]
    public string WriteFile([ToolDesc("The relative path to the file.")] string relativePath, [ToolDesc("The text content to write.")] string content)
    {
        try
        {
            string path = ResolvePath(relativePath);
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.WriteAllText(path, content);
            return $"Successfully wrote to {relativePath}.";
        }
        catch (Exception ex) { return $"Error writing file: {ex.Message}"; }
    }

    [ToolDesc("Appends text to a file relative to the project root. Creates the file if it doesn't exist.")]
    public string AppendToFile([ToolDesc("The relative path to the file.")] string relativePath, [ToolDesc("The text content to append.")] string content)
    {
        try
        {
            string path = ResolvePath(relativePath);
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.AppendAllText(path, content);
            return $"Successfully appended to {relativePath}.";
        }
        catch (Exception ex) { return $"Error appending to file: {ex.Message}"; }
    }

    [ToolDesc("Creates an empty file relative to the project root.")]
    public string CreateFile([ToolDesc("The relative path to the file.")] string relativePath)
    {
        try
        {
            string path = ResolvePath(relativePath);
            string? dir = Path.GetDirectoryName(path);
            if (dir != null) Directory.CreateDirectory(dir);
            File.WriteAllText(path, string.Empty);
            return $"Successfully created empty file at {relativePath}.";
        }
        catch (Exception ex) { return $"Error creating file: {ex.Message}"; }
    }

    [ToolDesc("Deletes a file relative to the project root.")]
    public string DeleteFile([ToolDesc("The relative path to the file.")] string relativePath)
    {
        try
        {
            string path = ResolvePath(relativePath);
            if (!File.Exists(path)) return $"File not found: {relativePath}";
            File.Delete(path);
            return $"Successfully deleted {relativePath}.";
        }
        catch (Exception ex) { return $"Error deleting file: {ex.Message}"; }
    }
}
