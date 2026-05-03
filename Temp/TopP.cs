// ApplyTopP - keeps only tokens with cumulative probability >= p, sorted by prob descending
private static void ApplyTopP(Span<float> probs, float p)
{
    if (p <= 0f || probs.IsEmpty) return;

    int n = probs.Length;
    
    // Build list of indices with non-zero probability
    var indices = new List<int>(n);
    for (int i = 0; i < n; i++)
        if (probs[i] > 0f) indices.Add(i);
    
    // Insertion sort descending by probability  
    for (int i = 1; i < indices.Count; i++)
    {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && probs[indices[j]] < probs[key])
        {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    // Zero out tokens beyond cumulative threshold
    float cumulative = 0f;
    for (int i = 0; i < indices.Count; i++)
    {
        cumulative += probs[indices[i]];
        if (cumulative >= p)
        {
            for (int j = i + 1; j < indices.Count; j++)
                probs[indices[j]] = 0f;
            break;
        }
    }
}
