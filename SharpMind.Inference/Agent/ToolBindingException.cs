namespace SharpMind.Inference.Agent;

/// <summary>
/// A model-supplied argument could not be bound to a tool's parameters. The
/// session treats this as a fixable call-shape error and feeds a corrected
/// example back to the model, unlike a tool's own runtime failure.
/// </summary>
internal sealed class ToolBindingException : ArgumentException
{
    public ToolBindingException(string message) : base(message)
    {
    }
}
