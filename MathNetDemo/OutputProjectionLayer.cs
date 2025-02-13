
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public class OutputProjectionLayer
{
    public int     InputDim  { get; }
    public int     OutputDim { get; }
    public MatrixF Weights   { get; set; }
    public VectorF Biases    { get; set; }

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

    // Turn the Forward matrix into a predicted token ID.
    public int PredictNextToken(MatrixF forwardMatrix)
    {
        MatrixF probs = Softmax(forwardMatrix);
        VectorF lastRow = probs.Row(probs.RowCount - 1);
        return lastRow.MaximumIndex();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    // Save the OutputProjectionLayer's parameters to a file.
    public void SaveToFile(string filename)
    {
        using (var writer = new StreamWriter(filename))
        {
            string header = $"{InputDim} {OutputDim}";
            writer.WriteLine(header);

            string weightsString = MatrixOperations.MatrixToString(Weights);
            string biasesString = MatrixOperations.VectorToString(Biases);

            writer.WriteLine(weightsString);
            writer.WriteLine(biasesString);
        }
    }

    // Load the OutputProjectionLayer's parameters from a file.
    public static OutputProjectionLayer LoadFromFile(string filename)
    {
        using (var reader = new StreamReader(filename))
        {
            // Read the header line that contains the dimensions.
            string[] dims = reader.ReadLine().Split(' ');
            int inputDim  = int.Parse(dims[0]);
            int outputDim = int.Parse(dims[1]);

            string? w_string = reader.ReadLine();
            string? b_string = reader.ReadLine();

            // check the strings are valid
            if (w_string == null || b_string == null)
                throw new ArgumentException("Input processing error");

            MatrixF newW1;
            VectorF newB1;

            bool w_read_ok = (MatrixOperations.TryStringToMatrix(w_string, out newW1!));
            bool b_read_ok = (MatrixOperations.TryStringToVector(b_string, out newB1!));

            if (!w_read_ok || !b_read_ok )
                 throw new ArgumentException("Matrix parsing error");

            // Create a new FeedForwardLayer instance.
            OutputProjectionLayer layer = new OutputProjectionLayer(inputDim, outputDim);

            if (w_read_ok) layer.Weights = newW1!;
            if (b_read_ok) layer.Biases = newB1!;

            return layer;
        }

    }

}