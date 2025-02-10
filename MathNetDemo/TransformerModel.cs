



using System;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

// Alias for brevity.
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;



public class TransformerModel
{
    public string     DirPath { get; set; } = "";
    public TokenVocab Vocab   { get; set; }

    public TransformerModel(string dirPath)
    {
        DirPath = dirPath;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    public void SaveModel()
    {
        // Determine all the filenames
        string vocabPath = Path.Combine(DirPath, "vocab.json");

        // check the directory exists
        if (!Directory.Exists(DirPath))
            Directory.CreateDirectory(DirPath);

        // Save the model parameters to a JSON file.
        // string modelPath = Path.Combine(DirPath, "model.json");
        // string modelJson = JsonSerializer.Serialize(this);
        // File.WriteAllText(modelPath, modelJson);

        Vocab.SaveToFile(vocabPath);
    }

    // --------------------------------------------------------------------------------------------

    public static TransformerModel LoadModel(string modelPath)
    {
        TransformerModel model = new TransformerModel(modelPath);

        // Determine all the filenames
        string vocabPath = Path.Combine(modelPath, "vocab.json");

        // Load the model parameters from a JSON file.
        model.Vocab = TokenVocab.LoadFromFile(vocabPath);

        return model;
    }
}