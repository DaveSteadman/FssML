using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>


public class FeedForwardLayer
{
    public int InputDim { get; }
    public int HiddenDim { get; }
    public int OutputDim { get; }

    // Two dense layers, one mapping InputDim -> HiddenDim and the other mapping HiddenDim -> OutputDim.
    public DenseLayer Dense1 { get; }
    public DenseLayer Dense2 { get; }

    public FeedForwardLayer(int inputDim, int hiddenDim, int outputDim)
    {
        InputDim = inputDim;
        HiddenDim = hiddenDim;
        OutputDim = outputDim;
        Dense1 = new DenseLayer(inputDim, hiddenDim);
        Dense2 = new DenseLayer(hiddenDim, outputDim);
    }

    // Forward pass: applies the first dense layer, a ReLU activation, then the second dense layer.
    public MatrixF Forward(MatrixF input)
    {
        // Apply first dense transformation.
        MatrixF hidden = Dense1.Forward(input);

        // Apply non-linear activation (ReLU) element-wise.
        hidden.MapInplace(x => MathF.Max(0, x));

        // Apply second dense transformation.
        MatrixF output = Dense2.Forward(hidden);
        return output;
    }

    // --------------------------------------------------------------------------
    // Save the FeedForwardLayer to a file.
    // The file format is as follows:
    // Line 1: InputDim HiddenDim OutputDim
    // Line 2: "Dense1" marker
    // Next InputDim lines: Each line is HiddenDim space-separated floats (Dense1 weights)
    // Next line: Dense1 biases (HiddenDim floats)
    // Next line: "Dense2" marker
    // Next HiddenDim lines: Each line is OutputDim space-separated floats (Dense2 weights)
    // Last line: Dense2 biases (OutputDim floats)
    // --------------------------------------------------------------------------
    public static void Save(FeedForwardLayer layer, string path)
    {
        using (var writer = new StreamWriter(path))
        {
            // Write the dimensions.
            writer.WriteLine($"{layer.InputDim} {layer.HiddenDim} {layer.OutputDim}");

            // Write Dense1 weights.
            writer.WriteLine("Dense1");
            for (int i = 0; i < layer.InputDim; i++)
            {
                writer.WriteLine(string.Join(" ", layer.Dense1.Weights.Row(i).ToArray()));
            }
            // Write Dense1 biases.
            writer.WriteLine(string.Join(" ", layer.Dense1.Biases.ToArray()));

            // Write Dense2 weights.
            writer.WriteLine("Dense2");
            for (int i = 0; i < layer.HiddenDim; i++)
            {
                writer.WriteLine(string.Join(" ", layer.Dense2.Weights.Row(i).ToArray()));
            }
            // Write Dense2 biases.
            writer.WriteLine(string.Join(" ", layer.Dense2.Biases.ToArray()));
        }
    }

    // --------------------------------------------------------------------------
    // Load a FeedForwardLayer from a file using the format above.
    // --------------------------------------------------------------------------
    public static FeedForwardLayer Load(string path)
    {
        using (var reader = new StreamReader(path))
        {
            // Read dimensions.
            string[] header = reader.ReadLine().Split(' ');
            int inputDim = int.Parse(header[0]);
            int hiddenDim = int.Parse(header[1]);
            int outputDim = int.Parse(header[2]);

            FeedForwardLayer layer = new FeedForwardLayer(inputDim, hiddenDim, outputDim);

            // Expect a "Dense1" marker.
            string dense1Marker = reader.ReadLine().Trim();
            if (dense1Marker != "Dense1")
                throw new Exception("Expected 'Dense1' marker in file.");

            // Read Dense1 weights.
            for (int i = 0; i < inputDim; i++)
            {
                string[] weightLine = reader.ReadLine().Split(' ');
                for (int j = 0; j < hiddenDim; j++)
                {
                    layer.Dense1.Weights[i, j] = float.Parse(weightLine[j]);
                }
            }
            // Read Dense1 biases.
            string[] dense1Biases = reader.ReadLine().Split(' ');
            for (int i = 0; i < hiddenDim; i++)
            {
                layer.Dense1.Biases[i] = float.Parse(dense1Biases[i]);
            }

            // Expect a "Dense2" marker.
            string dense2Marker = reader.ReadLine().Trim();
            if (dense2Marker != "Dense2")
                throw new Exception("Expected 'Dense2' marker in file.");

            // Read Dense2 weights.
            for (int i = 0; i < hiddenDim; i++)
            {
                string[] weightLine = reader.ReadLine().Split(' ');
                for (int j = 0; j < outputDim; j++)
                {
                    layer.Dense2.Weights[i, j] = float.Parse(weightLine[j]);
                }
            }
            // Read Dense2 biases.
            string[] dense2Biases = reader.ReadLine().Split(' ');
            for (int i = 0; i < outputDim; i++)
            {
                layer.Dense2.Biases[i] = float.Parse(dense2Biases[i]);
            }

            return layer;
        }
    }
}
