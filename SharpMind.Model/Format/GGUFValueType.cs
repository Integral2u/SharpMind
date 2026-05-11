namespace SharpMind.Model.Format;
public static partial class GgufLoader
{
    private enum GGUFValueType : uint
    {
        UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5, FLOAT32 = 6,
        BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
    }
}
