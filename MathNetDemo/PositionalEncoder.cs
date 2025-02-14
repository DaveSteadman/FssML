using System;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class PositionalEncoder
{
    private int sequenceLength;
    private int embeddingDim;

    private MatrixF positionalEncoding;
    public MatrixF EncodingMatrix => positionalEncoding;

    /// <summary>
    /// Initializes a new instance of the PositionalEncoder class
    /// and computes the positional encoding matrix.
    /// </summary>
    /// <param name="sequenceLength">The number of tokens in the sequence.</param>
    /// <param name="embeddingDim">The embedding dimension (d_model).</param>
    public PositionalEncoder(int sequenceLength, int embeddingDim)
    {
        this.sequenceLength     = sequenceLength;
        this.embeddingDim       = embeddingDim;
        this.positionalEncoding = CreatePositionalEncoding();
    }

    /// <summary>
    /// Creates the positional encoding matrix using the sinusoidal formulas.
    /// </summary>
    /// <returns>A MatrixF containing the positional encodings.</returns>
    private MatrixF CreatePositionalEncoding()
    {
        MatrixF pe = DenseMatrix.Build.Dense(sequenceLength, embeddingDim);
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                // Ensure that pairs (2i and 2i+1) share the same exponent.
                int index = i / 2;
                double exponent = (2.0 * index) / embeddingDim;
                double denominator = Math.Pow(10000, exponent);
                double angle = pos / denominator;

                pe[pos, i] = (i % 2 == 0)
                    ? (float)Math.Sin(angle)
                    : (float)Math.Cos(angle);
            }
        }
        return pe;
    }

    /// <summary>
    /// Applies the stored positional encoding to the provided embeddings.
    /// </summary>
    /// <param name="embeddings">A matrix of token embeddings with the same shape as the stored positional encoding.</param>
    /// <returns>A new matrix where the positional encoding is added element-wise.</returns>
    public MatrixF ApplyPositionalEncoding(MatrixF embeddings)
    {
        if (embeddings == null)
            throw new ArgumentNullException(nameof(embeddings));

        bool rowMatch = (embeddings.RowCount    == positionalEncoding.RowCount);
        bool colMatch = (embeddings.ColumnCount == positionalEncoding.ColumnCount);

        if (!rowMatch || !colMatch)
        {
            string inputDimensionStr    = $"[{embeddings.RowCount} x {embeddings.ColumnCount}]";
            string encodingDimensionStr = $"[{positionalEncoding.RowCount} x {positionalEncoding.ColumnCount}]";
            throw new ArgumentException($"The dimensions of the embeddings must match the stored positional encoding matrix // input {inputDimensionStr} // encoding {encodingDimensionStr}");
        }
        return embeddings + positionalEncoding;
    }

    /// <summary>
    /// Recreates the positional encoding matrix with new dimensions.
    /// </summary>
    /// <param name="newSequenceLength">The new sequence length.</param>
    /// <param name="newEmbeddingDim">The new embedding dimension.</param>
    public void RecreatePositionalEncoding(int newSequenceLength, int newEmbeddingDim)
    {
        this.sequenceLength = newSequenceLength;
        this.embeddingDim = newEmbeddingDim;
        this.positionalEncoding = CreatePositionalEncoding();
    }
}
