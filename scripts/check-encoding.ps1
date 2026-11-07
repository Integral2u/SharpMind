#!/usr/bin/env pwsh
# Fails when a text file contains the signatures of UTF-8 text that was
# decoded as Windows-1252 (for example a-euro-quote instead of an em dash).
# These appear in comments, log strings and UI text as visible junk.

param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

$extensions = @(
    '.cs', '.razor', '.xaml', '.html', '.css', '.js', '.ts', '.json', '.md',
    '.txt', '.resx', '.csproj', '.props', '.targets', '.yml', '.yaml', '.sh',
    '.ps1', '.xml', '.config', '.csv', '.rules', '.sln', '.slnf', '.vdproj',
    '.pubxml', '.editorconfig'
)

$markers = @(
    @{ Char = [char]0x00E2; Name = 'U+00E2' },
    @{ Char = [char]0x00C3; Name = 'U+00C3' },
    @{ Char = [char]0x00C2; Name = 'U+00C2' },
    @{ Char = [char]0xFFFD; Name = 'U+FFFD' }
)

$files = Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $extensions -contains $_.Extension -and
        $_.FullName -notmatch '\\(bin|obj|\.git|\.vs|node_modules|TestResults|nupkg|publish)\\' -and
        $_.Length -lt 8MB
    }

$failures = [System.Collections.Generic.List[string]]::new()
foreach ($file in $files) {
    $lines = [IO.File]::ReadAllLines($file.FullName, [Text.UTF8Encoding]::new($false))
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($marker in $markers) {
            if ($lines[$i].Contains($marker.Char)) {
                $failures.Add((
                    "{0}:{1}: {2} ({3})" -f
                        $file.FullName, ($i + 1), $lines[$i].Trim(), $marker.Name))
                break
            }
        }
    }
}

if ($failures.Count -gt 0) {
    $failures | Select-Object -First 50 | ForEach-Object { Write-Output $_ }
    if ($failures.Count -gt 50) {
        Write-Output ("... and " + ($failures.Count - 50) + " more")
    }
    Write-Output ("Encoding check failed: " + $failures.Count + " line(s).")
    exit 1
}

Write-Output ("Encoding check passed (" + $files.Count + " files).")
