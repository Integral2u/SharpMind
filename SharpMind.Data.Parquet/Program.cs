using Parquet;
using System.Linq;

var path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\open-perfectblend\train-00000-of-00006.parquet";
using var stream = File.OpenRead(path);
using var reader = await ParquetReader.CreateAsync(stream);

var rowGroup = await reader.ReadEntireRowGroupAsync(0);
Console.WriteLine($"RowGroup type: {rowGroup.GetType().FullName}");
foreach (var col in rowGroup)
{
    Console.WriteLine($"Column: {col.Field.Name}, Type: {col.Data.GetType().FullName}");
}
