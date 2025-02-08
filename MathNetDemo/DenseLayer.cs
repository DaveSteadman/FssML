
using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public class DenseLayer
{
    public int     InputDim  { get; }
    public int     OutputDim { get; }
    public MatrixF Weights   { get; }
    public VectorF Biases    { get; }

    public DenseLayer(int inputDim, int outputDim)
    {
        InputDim  = inputDim;
        OutputDim = outputDim;
        // Initialize the weight matrix randomly in [-0.1, 0.1]
        //Weights = DenseMatrix.CreateRandom<float>(            inputDim, outputDim,            new ContinuousUniform(-0.1f, 0.1f));

        Weights = DenseMatrix.CreateRandom(inputDim, outputDim, new ContinuousUniform(-0.1f, 0.1f));

        // Initialize biases to zero
        Biases = DenseVector.Create(OutputDim, 0f);
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
}