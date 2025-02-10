
using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public static class PositionalEncoder
{
    /// <summary>
    /// Creates a positional encoding matrix of shape (sequenceLength x embeddingDim)
    /// using the sinusoidal formulas from the transformer paper.
    /// </summary>
    /// <param name="sequenceLength">The number of tokens in the sequence.</param>
    /// <param name="embeddingDim">The embedding dimension (d_model).</param>
    /// <returns>A MatrixF with the positional encodings.</returns>
    // PositionalEncoder.GetPositionalEncoding(5, 4)
    public static MatrixF GetPositionalEncoding(int sequenceLength, int embeddingDim)
    {
        // Create a dense matrix to hold the positional encodings.
        MatrixF pe = DenseMatrix.Build.Dense(sequenceLength, embeddingDim);

        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                // Use floor division to ensure that pairs (2i and 2i+1) share the same exponent.
                int index = i / 2;
                // Compute the exponent: 2*index/embeddingDim.
                double exponent = (2.0 * index) / embeddingDim;
                double denominator = Math.Pow(10000, exponent);
                double angle = pos / denominator;

                if (i % 2 == 0)
                    pe[pos, i] = (float)Math.Sin(angle);
                else
                    pe[pos, i] = (float)Math.Cos(angle);
            }
        }
        return pe;
    }

    /// <summary>
    /// Given an embeddings matrix (shape: sequenceLength x embeddingDim),
    /// returns a new matrix where the positional encoding is added element-wise.
    /// </summary>
    /// <param name="embeddings">The token embeddings matrix.</param>
    /// <returns>The combined embeddings with positional encodings added.</returns>
    public static MatrixF ApplyPositionalEncoding(MatrixF embeddings)
    {
        if (embeddings == null)
            throw new ArgumentNullException(nameof(embeddings));

        int seqLength = embeddings.RowCount;
        int embedDim = embeddings.ColumnCount;

        // Get the positional encoding matrix.
        MatrixF posEnc = GetPositionalEncoding(seqLength, embedDim);

        // Return element-wise addition. MathNet supports operator+ for matrices.
        return embeddings + posEnc;
    }
}
