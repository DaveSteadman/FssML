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
        EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
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
        // Console.WriteLine("VocabSize: " + VocabSize);
        // Console.WriteLine("EmbeddingDim: " + EmbeddingDim);
        // Console.WriteLine("tokenIds.Count: " + tokenIds.Count);

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
        EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
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
        EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
    }

    public void AddLimitedNoise(float absOffset, float percentChanged)
    {
        float halfOffset = (float)absOffset / 2f;
        for (int i = 0; i < VocabSize; i++)
        {
            for (int j = 0; j < EmbeddingDim; j++)
            {
                if (random.NextDouble() < percentChanged)
                {
                    float offset = (float)(random.NextDouble() * absOffset - halfOffset);
                    EmbeddingMatrix[i, j] += offset;
                }
                // else: parameter remains unchanged.
            }
        }
        EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
    }

    // --------------------------------------------------------------------------------------------

    public MatrixF CreateNoise(float absOffset)
    {
        float halfOffset = (float)absOffset / 2f;
        MatrixF noise = DenseMatrix.Build.Random(VocabSize, EmbeddingDim, new ContinuousUniform(-halfOffset, halfOffset));
        return noise;
    }

    public MatrixF CreateLimitedNoise(float absOffset, float percentChanged)
    {
        float halfOffset = (float)absOffset / 2f;
        MatrixF noise = DenseMatrix.Build.Random(VocabSize, EmbeddingDim, new ContinuousUniform(-halfOffset, halfOffset));
        for (int i = 0; i < VocabSize; i++)
        {
            for (int j = 0; j < EmbeddingDim; j++)
            {
                if (random.NextDouble() >= percentChanged)
                {
                    noise[i, j] = 0.0f;
                }
            }
        }
        return noise;
    }

    public void ApplyNoise(MatrixF noise)
    {
        EmbeddingMatrix = EmbeddingMatrix.Add(noise);
        //EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
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

        EmbeddingMatrix = EmbeddingMatrix.TanhNormalize();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Report
    // --------------------------------------------------------------------------------------------

    public string Report()
    {
        return $"VocabSize={VocabSize}, EmbeddingDim={EmbeddingDim}, MatrixShape(RowxCol)={EmbeddingMatrix.RowCount}x{EmbeddingMatrix.ColumnCount} CheckSum={CheckSum()}";
    }

    public float CheckSum()
    {
        return EmbeddingMatrix.RowSums().Sum();
    }

    public static bool Compare(EmbeddingLayer a, EmbeddingLayer b)
    {
        bool sizeComp = a.VocabSize == b.VocabSize && a.EmbeddingDim == b.EmbeddingDim;
        bool matrixComp = a.EmbeddingMatrix.Equals(b.EmbeddingMatrix);
        return sizeComp && matrixComp;
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

    // --------------------------------------------------------------------------------------------
    // MARK: Binary Load Save
    // --------------------------------------------------------------------------------------------

    public void SaveToBinary(string path)
    {
        // Add retry logic to avoid file access conflicts
        const int maxRetries = 10;
        const int delayMs = 100;
        int retries = 0;
        while (true)
        {
            try
            {
                using (var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.None)))
                {
                    writer.Write(VocabSize);
                    writer.Write(EmbeddingDim);
                    for (int i = 0; i < VocabSize; i++)
                    {
                        for (int j = 0; j < EmbeddingDim; j++)
                        {
                            writer.Write(EmbeddingMatrix[i, j]);
                        }
                    }
                }
                break;
            }
            catch (IOException)
            {
                if (++retries >= maxRetries)
                    throw;
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }

    // --------------------------------------------------------------------------------------------

    public static EmbeddingLayer LoadFromBinary(string path)
    {
        // Add retry logic to avoid file access conflicts
        const int maxRetries = 10;
        const int delayMs = 100;
        int retries = 0;
        while (true)
        {
            try
            {
                using (var reader = new BinaryReader(File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read)))
                {
                    int vocabSize    = reader.ReadInt32();
                    int embeddingDim = reader.ReadInt32();

                    EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);

                    for (int i = 0; i < vocabSize; i++)
                    {
                        for (int j = 0; j < embeddingDim; j++)
                        {
                            embeddingLayer.EmbeddingMatrix[i, j] = reader.ReadSingle();
                        }
                    }

                    return embeddingLayer;
                }
            }
            catch (System.IO.IOException)
            {
                if (++retries >= maxRetries)
                    throw;
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }

}

