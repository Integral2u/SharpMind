using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

public sealed class ChatMLFormatter : IChatPromptFormatter
{
    private readonly string _userStart;
    private readonly string _assistantStart;
    private readonly string _systemStart;
    private readonly string _end;
    private readonly string[] _stopStrings;

    public ChatMLFormatter(string template)
    {
        var tokenMatches = RegexGenerated.ChatMLTokens.Matches(template);// System.Text.RegularExpressions.Regex.Matches(template, @"<\|[^|]+\|>");
        var allTokens = tokenMatches.Select(m => m.Value).Distinct().ToList();

        bool isChatML = allTokens.Any(t => t.Contains("im_start"));
        bool isZephyr = allTokens.Any(t => t == "<|system|>" || t == "<|user|>" || t == "<|assistant|>");

        if (isChatML)
        {
            _systemStart = "<|im_start|>system\n";
            _userStart = "<|im_start|>user\n";
            _assistantStart = "<|im_start|>assistant\n";
            _end = "<|im_end|>";
        }
        else if (isZephyr)
        {
            _systemStart = "<|system|>\n";
            _userStart = "<|user|>\n";
            _assistantStart = "<|assistant|>\n";
            _end = "</s>";
        }
        else
        {
            _userStart = "<|im_start|>user\n";
            _assistantStart = "<|im_start|>assistant\n";
            _systemStart = "<|im_start|>system\n";
            _end = "<|im_end|>";
        }
        _stopStrings = [_end];
    }

    public IReadOnlyList<string> DefaultStopStrings => _stopStrings;

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos, bool enableThinking = false, string? toolsJson = null)
    {
        var sb = new System.Text.StringBuilder();

        if (addBos && tokenizer.BosId >= 0)
            sb.Append(tokenizer.IdToToken(tokenizer.BosId));

        foreach (var msg in history)
        {
            switch (msg.Role)
            {
                case ChatRole.System:
                    sb.Append(_systemStart);
                    sb.Append(msg.Content);
                    sb.Append(_end);
                    break;
                case ChatRole.User:
                    sb.Append(_userStart);
                    sb.Append(msg.Content);
                    sb.Append(_end);
                    break;
                case ChatRole.Agent:
                    sb.Append(_assistantStart);
                    sb.Append(msg.Content);
                    sb.Append(_end);
                    break;
            }
            sb.Append('\n');
        }

        sb.Append(_assistantStart);
        return sb.ToString();
    }
}
