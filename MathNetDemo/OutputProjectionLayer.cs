
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

// Helper class for serialization.
public class OutputProjectionLayerParameters
{
    public int InputDim { get; set; }
    public int OutputDim { get; set; }
    public float[][] Weights { get; set; }
    public float[] Biases { get; set; }
}

public class OutputProjectionLayer
{
    public int     InputDim  { get; }
    public int     OutputDim { get; }
    public MatrixF Weights   { get; }
    public VectorF Biases    { get; }

    public OutputProjectionLayer(int inputDim, int outputDim)
    {
        InputDim  = inputDim;
        OutputDim = outputDim;
        // Initialize the weight matrix randomly in [-0.1, 0.1]

        Weights = DenseMatrix.CreateRandom(inputDim, outputDim, new ContinuousUniform(-0.1f, 0.1f));

        // Initialize biases to zero
        Biases = DenseVector.Create(OutputDim, 0f);
    }

    public int ParamCount()
    {
        return InputDim * OutputDim + OutputDim;
    }

    // Forward pass: multiplies the input matrix (numSamples x InputDim) by Weights
    // and adds Biases row-wise.
    public MatrixF Forward(MatrixF input)
    {
        // Multiply: (numSamples x InputDim) * (InputDim x OutputDim)
        MatrixF output = input * Weights; // shape: (numSamples, OutputDim)
        // Add biases to each row.
        for (int i = 0; i < output.RowCount; i++)
        {
            output.SetRow(i, output.Row(i) + Biases);
        }
        return output;
    }

    // Apply softmax row-wise. This converts each row of logits into probabilities.
    public static MatrixF Softmax(MatrixF logits)
    {
        MatrixF result = DenseMatrix.Create(logits.RowCount, logits.ColumnCount, 0f);
        for (int i = 0; i < logits.RowCount; i++)
        {
            VectorF row = logits.Row(i);
            // Subtract max for numerical stability.
            float max = row.Maximum();
            VectorF expRow = row.Map(v => MathF.Exp(v - max));
            float sum = expRow.Sum();
            result.SetRow(i, expRow / sum);
        }
        return result;
    }

    // Save the OutputProjectionLayer's parameters to a file.
    public static void Save(OutputProjectionLayer layer, string filename)
    {
        // Create a serializable object.
        var parameters = new OutputProjectionLayerParameters
        {
            InputDim = layer.InputDim,
            OutputDim = layer.OutputDim,
            Weights = new float[layer.Weights.RowCount][],
            Biases = layer.Biases.ToArray()
        };

        // Convert the Weights matrix to a jagged array.
        for (int i = 0; i < layer.Weights.RowCount; i++)
        {
            parameters.Weights[i] = layer.Weights.Row(i).ToArray();
        }

        // Serialize with indentation for readability.
        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(parameters, options);
        File.WriteAllText(filename, json);
    }

    // Load the OutputProjectionLayer's parameters from a file.
    public static OutputProjectionLayer Load(string filename)
    {
        string json = File.ReadAllText(filename);
        OutputProjectionLayerParameters parameters = JsonSerializer.Deserialize<OutputProjectionLayerParameters>(json);

        // Create a new OutputProjectionLayer with the saved dimensions.
        OutputProjectionLayer layer = new OutputProjectionLayer(parameters.InputDim, parameters.OutputDim);

        // Override the random weights with the loaded values.
        for (int i = 0; i < parameters.InputDim; i++)
        {
            for (int j = 0; j < parameters.OutputDim; j++)
            {
                layer.Weights[i, j] = parameters.Weights[i][j];
            }
        }

        // Update the biases.
        for (int i = 0; i < parameters.OutputDim; i++)
        {
            layer.Biases[i] = parameters.Biases[i];
        }

        return layer;
    }

}