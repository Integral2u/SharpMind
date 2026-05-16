namespace SharpMind.Inference.Chat.PromptFormatters;

public static class ChatPromptFormatterFactory
{
    public static IChatPromptFormatter Create(string? chatTemplate)
    {
        if (string.IsNullOrEmpty(chatTemplate))
            return new SimpleFormatter();

        // Llama 3.x: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        if (chatTemplate.Contains("start_header_id"))
            return new LlamaFormatter();

        // DeepSeek: <｜User｜> / <｜Assistant｜> special tokens
        if (chatTemplate.Contains("｜User｜") || chatTemplate.Contains("｜Assistant｜"))
            return new DeepSeekFormatter();

        // ChatML: <|im_start|>role / <|im_end|>
        // Zephyr: <|role|> / <|/role|>
        if (chatTemplate.Contains("im_start") ||
            chatTemplate.Contains("/system") ||
            chatTemplate.Contains("/user") ||
            chatTemplate.Contains("/assistant"))
            return new ChatMLFormatter(chatTemplate);

        return new SimpleFormatter();
    }
}
