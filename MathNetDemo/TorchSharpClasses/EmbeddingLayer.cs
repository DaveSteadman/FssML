using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class TorchEmbeddingLayer : Module<Tensor, Tensor>
{
    private Tensor weight;

    private int padTokenId;

    // --------------------------------------------------------------------------------------------
    // MARK: Constructors
    // --------------------------------------------------------------------------------------------

    public TorchEmbeddingLayer(string name, int vocabSize, int embeddingDim, int padTokenId = -1)
        : base(name)
    {
        this.padTokenId = padTokenId;
        // Create the embedding weight matrix and initialize with random values.
        weight = Parameter(torch.randn(vocabSize, embeddingDim));

        // If a pad token is defined, set its embedding to zero.
        if (padTokenId >= 0 && padTokenId < vocabSize)
        {
            weight[padTokenId] = torch.zeros(embeddingDim);
        }

        RegisterComponents();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: forward
    // --------------------------------------------------------------------------------------------

    public override Tensor forward(Tensor input)
    {
        Console.WriteLine("Input shape: " + string.Join(", ", input.shape));

        var flattened = input.flatten();
        var selected = weight.index_select(0, flattened);
        var newShape = input.shape.Concat(new long[] { weight.shape[1] }).ToArray();
        var output = selected.reshape(newShape);

        Console.WriteLine("Output shape: " + string.Join(", ", output.shape));
        return output;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string filePath)
    {
        // Use the built-in save method (note: method name is lowercase "save")
        this.save(filePath);
        Console.WriteLine($"State saved to {filePath}");
    }

    public static TorchEmbeddingLayer LoadFromFile(string filePath, string name, int vocabSize, int embeddingDim, int padTokenId = -1)
    {
        var module = new TorchEmbeddingLayer(name, vocabSize, embeddingDim, padTokenId);
        // Use the built-in load method to load state into the module.
        module.load(filePath);
        Console.WriteLine($"State loaded from {filePath}");
        return module;
    }
}
