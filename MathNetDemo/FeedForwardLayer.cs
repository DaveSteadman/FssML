using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class FeedForwardLayer
{
    // Input/output dimension (d_model) and hidden dimension (d_ff)
    private int d_model;
    private int d_ff;

    // Weight matrices and bias vectors for the two linear transformations
    public MatrixF W1 { get; private set; }
    public VectorF b1 { get; private set; }
    public MatrixF W2 { get; private set; }
    public VectorF b2 { get; private set; }

    /// <summary>
    /// Constructs the feed-forward layer.
    /// d_model: the input and output dimension (e.g. embedding size)
    /// d_ff: the hidden layer dimension (typically 4 times d_model)
    /// </summary>
    public FeedForwardLayer(int d_model, int d_ff)
    {
        this.d_model = d_model;
        this.d_ff    = d_ff;

        // Initialize weights randomly.
        // In practice you might want to initialize with a specific distribution or scale.
        W1 = MatrixF.Build.Random(d_model, d_ff);
        b1 = VectorF.Build.Dense(d_ff, 0.0f);
        W2 = MatrixF.Build.Random(d_ff, d_model);
        b2 = VectorF.Build.Dense(d_model, 0.0f);
    }

    // ReLU activation function.
    private float ReLU(float x)
    {
        return x > 0 ? x : 0;
    }

    /// <summary>
    /// Forward pass for the feed-forward layer.
    /// The input is a matrix of shape [numTokens x d_model].
    /// The output is a matrix of the same shape.
    /// </summary>
    // Usage: MatrixF output = feedForwardLayer.Forward(input);
    public MatrixF Forward(MatrixF input)
    {
        // First linear layer: compute (input * W1 + b1).
        // Resulting shape: [numTokens x d_ff]
        MatrixF hiddenLinear = input * W1;
        // Add bias and apply ReLU activation element-wise.
        for (int i = 0; i < hiddenLinear.RowCount; i++)
        {
            for (int j = 0; j < hiddenLinear.ColumnCount; j++)
            {
                hiddenLinear[i, j] = ReLU(hiddenLinear[i, j] + b1[j]);
            }
        }

        // Second linear layer: compute (hidden * W2 + b2).
        // This projects back to the original dimension, shape: [numTokens x d_model]
        MatrixF output = hiddenLinear * W2;
        for (int i = 0; i < output.RowCount; i++)
        {
            for (int j = 0; j < output.ColumnCount; j++)
            {
                output[i, j] = output[i, j] + b2[j];
            }
        }

        return output;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string path)
    {
        using (var writer = File.CreateText(path))
        {
            // Write the model dimension.
            writer.WriteLine(d_model);
            writer.WriteLine(d_ff);

            // Write the weights and biases.
            writer.WriteLine(JsonSerializer.Serialize(W1));
            writer.WriteLine(JsonSerializer.Serialize(b1));
            writer.WriteLine(JsonSerializer.Serialize(W2));
            writer.WriteLine(JsonSerializer.Serialize(b2));
        }
    }

    // --------------------------------------------------------------------------------------------

    public static FeedForwardLayer LoadFromFile(string path)
    {
        using (var reader = File.OpenText(path))
        {
            // Read the model dimension.
            int d_model = int.Parse(reader.ReadLine());
            int d_ff    = int.Parse(reader.ReadLine());

            // Create a new FeedForwardLayer instance.
            FeedForwardLayer layer = new FeedForwardLayer(d_model, d_ff);

            // Read the weights and biases.
            layer.W1 = JsonSerializer.Deserialize<MatrixF>(reader.ReadLine());
            layer.b1 = JsonSerializer.Deserialize<VectorF>(reader.ReadLine());
            layer.W2 = JsonSerializer.Deserialize<MatrixF>(reader.ReadLine());
            layer.b2 = JsonSerializer.Deserialize<VectorF>(reader.ReadLine());

            return layer;
        }
    }
}

