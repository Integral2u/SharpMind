using System;
using System.Collections.Generic;
using SharpMind.Inference.Chat.PromptFormatters;
using Xunit;

namespace SharpMind.Tests.Chat;

/// <summary>
/// Covers the guard that catches a formatter paired with a model whose vocabulary
/// lacks its turn markers — the failure that makes a model generate past its own
/// answer instead of stopping, with no error anywhere to explain it.
/// </summary>
public class FormatterVocabMismatchTests
{
    private static readonly HashSet<string> QwenVocab =
        new(StringComparer.Ordinal) { "<|im_start|>", "<|im_end|>", "<|endoftext|>" };

    private static readonly HashSet<string> Llama3Vocab =
        new(StringComparer.Ordinal) { "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>" };

    [Fact]
    public void Llama3FormatterOnQwenVocab_IsReported()
    {
        string? msg = ChatPromptFormatterFactory.DescribeVocabMismatch(
            new Llama3Formatter(), QwenVocab.Contains);

        Assert.NotNull(msg);
        Assert.Contains("<|eot_id|>", msg);
    }

    [Fact]
    public void Llama3FormatterOnLlama3Vocab_IsSilent()
    {
        Assert.Null(ChatPromptFormatterFactory.DescribeVocabMismatch(
            new Llama3Formatter(), Llama3Vocab.Contains));
    }

    [Fact]
    public void ChatMLFormatterOnLlama3Vocab_IsReported()
    {
        string? msg = ChatPromptFormatterFactory.DescribeVocabMismatch(
            new ChatMLFormatter("<|im_start|><|im_end|>"), Llama3Vocab.Contains);

        Assert.NotNull(msg);
        Assert.Contains("<|im_end|>", msg);
    }

    [Fact]
    public void ChatMLFormatterOnQwenVocab_IsSilent()
    {
        Assert.Null(ChatPromptFormatterFactory.DescribeVocabMismatch(
            new ChatMLFormatter("<|im_start|><|im_end|>"), QwenVocab.Contains));
    }

    /// <summary>
    /// Formats whose stop strings are ordinary text are supposed to tokenize as
    /// text. Reporting those would make the warning noise and train people to
    /// ignore it, so they must stay silent against any vocabulary.
    /// </summary>
    [Theory]
    [MemberData(nameof(PlainTextFormatters))]
    public void PlainTextFormatters_AreNeverReported(IChatPromptFormatter formatter)
    {
        Assert.Null(ChatPromptFormatterFactory.DescribeVocabMismatch(formatter, QwenVocab.Contains));
        Assert.Null(ChatPromptFormatterFactory.DescribeVocabMismatch(formatter, Llama3Vocab.Contains));
        Assert.Null(ChatPromptFormatterFactory.DescribeVocabMismatch(formatter, _ => false));
    }

    public static TheoryData<IChatPromptFormatter> PlainTextFormatters() =>
    [
        new AlpacaFormatter(),
        new QuestionAnswerFormatter(),
        new RawTemplateFormatter(),
        new SimpleFormatter(),
    ];
}
