using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public class EmbeddingLayer
{
    public int     VocabSize       { get; private set; }
    public int     EmbeddingDim    { get; private set; }
    public MatrixF EmbeddingMatrix { get; private set; }

    private static readonly Random random = new Random();

    // --------------------------------------------------------------------------------------------
    // MARK: Constructor
    // --------------------------------------------------------------------------------------------

    public EmbeddingLayer(int vocabSize, int embeddingDim)
    {
        VocabSize       = vocabSize;
        EmbeddingDim    = embeddingDim;

        // Create and initialize the embedding matrix with random float values.
        EmbeddingMatrix = DenseMatrix.Build.Random(vocabSize, embeddingDim, new ContinuousUniform(-0.1f, 0.1f));
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    public EmbeddingLayer DeepCopy()
    {
        EmbeddingLayer newLayer = new EmbeddingLayer(VocabSize, EmbeddingDim);
        newLayer.EmbeddingMatrix = EmbeddingMatrix.Clone();
        return newLayer;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Query
    // --------------------------------------------------------------------------------------------

    public int ParamCount()
    {
        return VocabSize * EmbeddingDim;
    }

    // --------------------------------------------------------------------------------------------

    // Given a token ID, returns the corresponding embedding vectors.
    public VectorF Lookup(int tokenId)
    {
        if (tokenId >= 0 && tokenId < VocabSize)
            return EmbeddingMatrix.Row(tokenId);
        else
            return DenseVector.Build.Dense(EmbeddingDim, 0.0f);
    }

    // --------------------------------------------------------------------------------------------

    // Given a list of token IDs (such as a whole input string in token IDs), returns the corresponding embedding vectors.
    public List<VectorF> LookupList(List<int> tokenIds)
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

    // An evolution of LookupList, returning a matrix of tokens x embeddings rather than a list of embeddings.
    public MatrixF LookupListToMatrix(List<int> tokenIds)
    {
        // Write out the size params for debugging purposes.
        Console.WriteLine("VocabSize: " + VocabSize);
        Console.WriteLine("EmbeddingDim: " + EmbeddingDim);
        Console.WriteLine("tokenIds.Count: " + tokenIds.Count);


        Console.WriteLine("LookupList2");
        MatrixF embeddings = DenseMatrix.Build.Dense(tokenIds.Count, EmbeddingDim);
        for (int i = 0; i < tokenIds.Count; i++)
        {
            int id = tokenIds[i];
            if (id >= 0 && id < VocabSize)
                embeddings.SetRow(i, EmbeddingMatrix.Row(id));
            else
                embeddings.SetRow(i, DenseVector.Build.Dense(EmbeddingDim, 0.0f));
        }
        return embeddings;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Change
    // --------------------------------------------------------------------------------------------

    public void SetRandom(float min, float max)
    {
        EmbeddingMatrix = DenseMatrix.Build.Random(VocabSize, EmbeddingDim, new ContinuousUniform(min, max));
    }

    // --------------------------------------------------------------------------------------------

    public void AddNoise(float absOffset)
    {
        float halfOffset = (float)absOffset / 2f;
        for (int i = 0; i < VocabSize; i++)
        {
            for (int j = 0; j < EmbeddingDim; j++)
            {
                float offset = (float)(random.NextDouble() * absOffset - halfOffset);
                EmbeddingMatrix[i, j] += offset;
            }
        }
    }

    // --------------------------------------------------------------------------------------------

    public void Normalize(float min, float max)
    {
        // float minVal = EmbeddingMatrix.Enumerate().Min();
        // float maxVal = EmbeddingMatrix.Enumerate().Max();
        // float range  = maxVal - minVal;

        // for (int i = 0; i < VocabSize; i++)
        // {
        //     for (int j = 0; j < EmbeddingDim; j++)
        //     {
        //         EmbeddingMatrix[i, j] = (EmbeddingMatrix[i, j] - minVal) / range * (max - min) + min;
        //     }
        // }

        EmbeddingMatrix.TanhNormalize();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string path)
    {
        using (var writer = new StreamWriter(path))
        {
            writer.WriteLine($"{VocabSize} {EmbeddingDim}");
            for (int i = 0; i < VocabSize; i++)
            {
                writer.WriteLine(string.Join(" ", EmbeddingMatrix.Row(i).ToArray()));
            }
        }
    }

    // --------------------------------------------------------------------------------------------

    public static EmbeddingLayer LoadFromFile(string path)
    {
        using (var reader = new StreamReader(path))
        {
            string[] header       = reader.ReadLine().Split(' ');
            int      vocabSize    = int.Parse(header[0]);
            int      embeddingDim = int.Parse(header[1]);

            if (vocabSize    == 0) vocabSize    = 1;
            if (embeddingDim == 0) embeddingDim = 1;

            EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);

            for (int i = 0; i < vocabSize; i++)
            {
                string[] line = reader.ReadLine().Split(' ');
                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddingLayer.EmbeddingMatrix[i, j] = float.Parse(line[j]);
                }
            }

            return embeddingLayer;
        }
    }
}

