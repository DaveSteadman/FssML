using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public class EmbeddingLayer
{
    public int VocabSize           { get; private set; }
    public int EmbeddingDim        { get; private set; }
    public MatrixF EmbeddingMatrix { get; private set; }

    public EmbeddingLayer(int vocabSize, int embeddingDim)
    {
        VocabSize       = vocabSize;
        EmbeddingDim    = embeddingDim;
        // Initialize the embedding matrix with small random float values.
        EmbeddingMatrix = DenseMatrix.Build.Random(vocabSize, embeddingDim, new ContinuousUniform(-0.1f, 0.1f));
    }

    /// <summary>
    /// Given a list of token IDs, returns the corresponding embedding vectors.
    /// </summary>
    public List<VectorF> Lookup(List<int> tokenIds)
    {
        var embeddings = new List<VectorF>();
        foreach (var id in tokenIds)
        {
            if (id >= 0 && id < VocabSize)
                embeddings.Add(EmbeddingMatrix.Row(id));
            else
                embeddings.Add(DenseVector.Build.Dense(EmbeddingDim, 0.0f));
        }
        return embeddings;
    }
}

