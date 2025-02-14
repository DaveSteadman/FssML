using System;
using System.IO;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class OutputProjectionLayer
{
    public int InputDim  { get; }
    public int OutputDim { get; }
    public MatrixF Weights { get; set; }
    public VectorF Biases  { get; set; }

    public OutputProjectionLayer(int inputDim, int outputDim)
    {
        InputDim  = inputDim;
        OutputDim = outputDim;

        // Initialize weights randomly in [-1, 1]
        Weights = DenseMatrix.CreateRandom(inputDim, outputDim, new ContinuousUniform(-1f, 1f));

        // Initialize biases to zero.
        Biases = DenseVector.Create(OutputDim, 0f);
    }

    public int ParamCount() => InputDim * OutputDim + OutputDim;

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    /// <summary>
    /// Forward pass: multiplies the input matrix (numSamples x InputDim) by Weights,
    /// and adds Biases to each row. Returns a matrix of logits with shape (numSamples, OutputDim).
    /// </summary>
    public MatrixF Forward(MatrixF input)
    {
        if (input.ColumnCount != InputDim)
            throw new ArgumentException($"Expected input with {InputDim} columns, but got {input.ColumnCount}.");

        MatrixF output = input * Weights;
        for (int i = 0; i < output.RowCount; i++)
        {
            output.SetRow(i, output.Row(i) + Biases);
        }
        return output;
    }

    /// <summary>
    /// Given the forward output (a matrix of logits with shape [inputSize, OutputDim]),
    /// aggregates the rows (using average pooling) to produce a single vector of length OutputDim,
    /// applies softmax, and returns the token ID (i.e. the index of the highest probability).
    /// </summary>
    // public int PredictNextToken(MatrixF forwardMatrix)
    // {
    //     // forwardMatrix: [inputSize x OutputDim]
    //     MatrixF logits = Forward(forwardMatrix);

    //     // Aggregate the logits by averaging across all rows.
    //     // This results in a single vector of length OutputDim.
    //     VectorF aggregated = logits.RowSums().Divide(logits.RowCount);

    //     // Apply softmax to get probabilities.
    //     VectorF probabilities = Softmax(aggregated);

    //     // Return the index with the maximum probability.
    //     return ArgMax(probabilities);
    // }

    public int PredictNextToken(MatrixF forwardMatrix)
    {
        MatrixF logits = Forward(forwardMatrix);
        // Use ColumnSums() to aggregate logits into a single vector of length OutputDim.
        VectorF aggregated = logits.ColumnSums().Divide(logits.RowCount);
        VectorF probabilities = Softmax(aggregated);
        return ArgMax(probabilities);
    }


    // Return a vector of rankings for the next token, starting at 0 for the most likely token, and increasing for each next-most-likely token.
    public VectorF NextTokenRankings(MatrixF forwardMatrix)
    {
        // Compute the logits using the forward pass.
        MatrixF logits = Forward(forwardMatrix);
        // Aggregate logits by averaging across all rows.
        VectorF aggregated = logits.RowSums().Divide(logits.RowCount);
        // Compute probabilities using softmax.
        VectorF probabilities = Softmax(aggregated);

        int vocabSize = OutputDim;
        // Create an array of indices [0, 1, ..., vocabSize-1]
        int[] indices = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            indices[i] = i;
        }
        // Sort indices in descending order based on probabilities.
        Array.Sort(indices, (a, b) => probabilities[b].CompareTo(probabilities[a]));

        // Build a ranking vector:
        // For each token, its rank is its position in the sorted order.
        // The most likely token (highest probability) gets rank 0.
        float[] ranking = new float[vocabSize];
        for (int rank = 0; rank < vocabSize; rank++)
        {
            ranking[indices[rank]] = rank;
        }

        // Convert the ranking array to a MathNet vector and return.
        return DenseVector.Create(vocabSize, i => ranking[i]);
    }

    // Returns an array of tuples, where each tuple contains a token ID and its probability,
    // sorted in descending order of probability.
    public (int tokenId, float probability)[] TopNTokens(MatrixF forwardMatrix, int n)
    {
        // Compute logits via the forward pass.
        MatrixF logits = Forward(forwardMatrix);

        // Aggregate logits by averaging each column.
        // This produces a vector of length OutputDim.
        VectorF aggregated = logits.ColumnSums().Divide(logits.RowCount);

        // Apply softmax to convert logits into a probability distribution.
        VectorF probabilities = Softmax(aggregated);

        int vocabSize = OutputDim;
        // Create an array of all token indices.
        int[] indices = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            indices[i] = i;
        }

        // Sort indices in descending order based on probabilities.
        Array.Sort(indices, (a, b) => probabilities[b].CompareTo(probabilities[a]));

        int topN = Math.Min(n, vocabSize);
        (int tokenId, float probability)[] topTokens = new (int, float)[topN];
        for (int i = 0; i < topN; i++)
        {
            topTokens[i] = (indices[i], probabilities[indices[i]]);
        }

        return topTokens;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Mutation
    // --------------------------------------------------------------------------------------------

    public void SetRandom()
    {
        Weights = DenseMatrix.CreateRandom(InputDim, OutputDim, new ContinuousUniform(-1f, 1f));
        Biases  = DenseVector.Create(OutputDim, 0f);
    }

    public void AddNoise(float absOffset)
    {
        // Create noise matrices.
        MatrixF noiseW = DenseMatrix.CreateRandom(InputDim, OutputDim, new ContinuousUniform(-absOffset, absOffset));
        VectorF noiseB = DenseVector.CreateRandom(OutputDim, new ContinuousUniform(-absOffset, absOffset));

        // Add noise to the weights and biases.
        Weights += noiseW;
        Biases  += noiseB;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Training
    // --------------------------------------------------------------------------------------------

    // For a Loss function:
    // - Score the right selected token, its magnitude is proportional to the probability of the token.

    public float Loss(MatrixF forwardMatrix, int targetTokenID)
    {
        float retScore = 0f;

        MatrixF logits        = Forward(forwardMatrix);
        VectorF aggregated    = logits.ColumnSums().Divide(logits.RowCount);
        VectorF vocabRankings = Softmax(aggregated);

        // check the sizes
        if (vocabRankings.Count != OutputDim)
            throw new ArgumentException($"Size mismatch: vocabRankings.Count ({vocabRankings.Count}) != OutputDim ({OutputDim})");
        if (targetTokenID < 0 || targetTokenID >= OutputDim)
            throw new ArgumentException($"Invalid target token ID: {targetTokenID} in vocab size {OutputDim}");

        // Report the max and min values in the vocabRankings vector.
        Console.WriteLine($"Max: {vocabRankings.Maximum()} Min: {vocabRankings.Minimum()}");

        // Score the correctly selected token.
        var topTokens = TopNTokens(forwardMatrix, 5);
        if (topTokens.Length != 5)
            throw new ArgumentException("TopNTokens did not return 5 tokens");
        if (topTokens[0].tokenId == targetTokenID)
        {
            retScore += 50f * topTokens[0].probability;

            // score the gap to the second token
            retScore += 10f * (topTokens[0].probability - topTokens[1].probability);
        }
        else // else, score it negatively on its ranking
        {
            retScore -= (1 - vocabRankings[targetTokenID]) * 10f;
        }

        return retScore;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Util
    // --------------------------------------------------------------------------------------------

    /// <summary>
    /// Returns the index of the maximum element in the vector.
    /// </summary>
    private int ArgMax(VectorF vector)
    {
        int maxIndex = 0;
        float maxValue = vector[0];
        for (int i = 1; i < vector.Count; i++)
        {
            if (vector[i] > maxValue)
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /// <summary>
    /// Applies softmax to a vector.
    /// </summary>
    private static VectorF Softmax(VectorF vector)
    {
        float max = vector.Maximum();
        VectorF exp = vector.Map(v => MathF.Exp(v - max));
        float sum = exp.Sum();
        return exp.Divide(sum);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string filename)
    {
        using (var writer = new StreamWriter(filename))
        {
            string header = $"{InputDim} {OutputDim}";
            writer.WriteLine(header);

            string weightsString = MatrixOperations.MatrixToString(Weights);
            string biasesString  = MatrixOperations.VectorToString(Biases);

            writer.WriteLine(weightsString);
            writer.WriteLine(biasesString);
        }
    }

    public static OutputProjectionLayer LoadFromFile(string filename)
    {
        using (var reader = new StreamReader(filename))
        {
            string[] dims = reader.ReadLine().Split(' ');
            int inputDim  = int.Parse(dims[0]);
            int outputDim = int.Parse(dims[1]);

            string? w_string = reader.ReadLine();
            string? b_string = reader.ReadLine();

            if (w_string == null || b_string == null)
                throw new ArgumentException("Input processing error");

            MatrixF newW;
            VectorF newB;

            bool w_read_ok = MatrixOperations.TryStringToMatrix(w_string, out newW!);
            bool b_read_ok = MatrixOperations.TryStringToVector(b_string, out newB!);

            if (!w_read_ok || !b_read_ok)
                throw new ArgumentException("Matrix parsing error");

            OutputProjectionLayer layer = new OutputProjectionLayer(inputDim, outputDim);
            layer.Weights = newW!;
            layer.Biases  = newB!;
            return layer;
        }
    }
}
