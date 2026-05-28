namespace SharpMind.Inference.Chat.PromptFormatters;

public static class ChatPromptFormatterFactory
{
    public static IChatPromptFormatter Create(string? chatTemplate, string? modelName = null)
    {
        if (string.IsNullOrEmpty(chatTemplate))
        {
           // if (modelName is not null && IsBaseModel(modelName))
            //    return new PassThroughFormatter();
            return new SimpleFormatter();
        }

        // Llama 3.x: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        if (chatTemplate.Contains("start_header_id"))
            return new LlamaFormatter();

        // DeepSeek: <｜User｜> / <｜Assistant｜> special tokens
        if (chatTemplate.Contains("｜User｜") || chatTemplate.Contains("｜Assistant｜"))
            return new DeepSeekFormatter();

        // ChatML: <|im_start|>role / <|im_end|>
        // Zephyr: <|role|> / <|/role|> or {% if message['role'] == 'user' %} with <|user|>
        if (chatTemplate.Contains("im_start") ||
            chatTemplate.Contains("/system") ||
            chatTemplate.Contains("/user") ||
            chatTemplate.Contains("/assistant") ||
            chatTemplate.Contains("<|system|>") ||
            chatTemplate.Contains("<|user|>") ||
            chatTemplate.Contains("<|assistant|>"))
            return new ChatMLFormatter(chatTemplate);

        return new SimpleFormatter();
    }

    private static bool IsBaseModel(string modelName)
    {
        var lower = modelName.ToLowerInvariant();
        return lower.Contains("smollm") || lower.Contains("base");
    }
}
