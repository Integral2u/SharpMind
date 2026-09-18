namespace SharpMind.Inference.Grammar;

/// <summary>
/// One live position in the pushdown grammar simulation. <see cref="State"/> is
/// an NFA state inside a rule; <see cref="Stack"/> holds, innermost last, the
/// return states of the rules currently awaiting completion. Because each rule
/// call records its own continuation on the stack, the same rule can be used in
/// many places and can recurse without any context leaking between uses.
/// Configurations compare and hash by value (state plus every stack entry).
/// </summary>
internal readonly struct Config : IEquatable<Config>
{
    public readonly int State;
    public readonly int[] Stack;

    public Config(int state, int[] stack)
    {
        State = state;
        Stack = stack;
    }

    public bool Equals(Config other)
    {
        if (State != other.State || Stack.Length != other.Stack.Length)
            return false;
        int[] a = Stack, b = other.Stack;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                return false;
        return true;
    }

    public override bool Equals(object? obj)
        => obj is Config other && Equals(other);

    public override int GetHashCode()
    {
        HashCode h = new();
        h.Add(State);
        int[] a = Stack;
        for (int i = 0; i < a.Length; i++)
            h.Add(a[i]);
        return h.ToHashCode();
    }
}