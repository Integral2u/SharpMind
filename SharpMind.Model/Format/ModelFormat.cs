namespace SharpMind.Model.Format;
public static partial class ModelConverter
{
    /// <summary>Supported external model formats.</summary>
    public enum ModelFormat
    {
        Unknown,
        Gguf,         // llama.cpp
        Pytorch,      // PyTorch checkpoint
    }
}