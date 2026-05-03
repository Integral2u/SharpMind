// ApplyTopP - keeps top tokens whose cumulative probability >= p
private static void ApplyTopP(Span<float> probs, float p)
{
    if (p <= 0f || probs.IsEmpty) return;

    int n = probs.Length;
    
    // Build sorted indices list (descending by probability)
    var indices = new List<int>(n);
    for (int i = 0; i < n; i++)
    {
        if (probs[i] > 0f)
        {
            indices.Add(i);
        }
    }
    
    // Sort descending (simple bubble sort for clarity)
    for (int i = 0; i < indices.Count; i++)
    {
        for (int j = i + 1; j < indices.Count; j++)
        {
            if (probs[indices[i]] < probs[indices[j]])
            {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    // Keep only tokens within cumulative probability threshold
    float cumulative = 0f;
    for (int i = 0; i < indices.Count; i++)
    {
        int idx = indices[i];
        cumulative += probs[idx];
        
        if (cumulative >= p)
        {
            for (int j = i + 1; j < indices.Count; j++)
            {
                probs[indices[j]] = 0f;
            }
            break;
        }
    }
}