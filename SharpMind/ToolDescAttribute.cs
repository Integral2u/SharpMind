namespace SharpMind
{
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Parameter)]
    public sealed class ToolDescAttribute(string text) : Attribute
    {
        public string Text { get; } = text;
    }
}
