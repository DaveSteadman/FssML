using System;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

// Alias for brevity.
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class SelfAttentionNoise
{
    public List<MatrixF> W_qn { get; set; } = new();
    public List<MatrixF> W_kn { get; set; } = new();
    public List<MatrixF> W_vn { get; set; } = new();
    public MatrixF W_on { get; set; }
}


public class SelfAttention
{
    // The dimensionality of the model (and of the query, key, value vectors).
    public int InputLen { get; private set;}
    public int ModelDim { get; private set; }
    public int NumHeads { get; private set; }

    // Weight matrices per head
    public List<MatrixF> W_q { get; private set; } = new();
    public List<MatrixF> W_k { get; private set; } = new();
    public List<MatrixF> W_v { get; private set; } = new();

    // Output projection weight matrix. Shape: (ModelDim*NumHeads, ModelDim)
    public MatrixF W_o { get; private set; }

    private static readonly Random random = new Random();

    // --------------------------------------------------------------------------------------------

    // Creates a new self-attention layer with the given model dimensionality.
    // modelDim: The size of the model and each token’s embedding vector.</param>
    public SelfAttention(int inputLen, int modelDim, int numHeads = 1)
    {
        InputLen = inputLen;
        ModelDim = modelDim;
        NumHeads = numHeads;

        // Create a uniform distribution for small random initialization.
        float rangeMin = -1f;
        float rangeMax =  1f;

        for (int h = 0; h < NumHeads; h++)
        {
            W_q.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            W_k.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            W_v.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        W_o = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    public SelfAttention DeepCopy()
    {
        SelfAttention newLayer = new SelfAttention(InputLen, ModelDim, NumHeads);
        for (int h = 0; h < NumHeads; h++)
        {
            newLayer.W_q.Add(W_q[h].Clone());
            newLayer.W_k.Add(W_k[h].Clone());
            newLayer.W_v.Add(W_v[h].Clone());
        }
        newLayer.W_o = W_o.Clone();
        return newLayer;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Mutation
    // --------------------------------------------------------------------------------------------

    public void SetRandomWeights(float min, float max)
    {
        // Create a uniform distribution for small random initialization.
        float rangeMin = min;
        float rangeMax = max;

        W_q.Clear();
        W_k.Clear();
        W_v.Clear();
        for (int h = 0; h < NumHeads; h++)
        {
            W_q.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            W_k.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            W_v.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        W_o = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
    }

    public void AddNoise(float absOffset)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        // A different offset is applied to each value in each matrix.
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        List<MatrixF> offsetq = new();
        List<MatrixF> offsetk = new();
        List<MatrixF> offsetv = new();
        for (int h = 0; h < NumHeads; h++)
        {
            offsetq.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetk.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetv.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        for (int h = 0; h < NumHeads; h++)
        {
            W_q[h] += offsetq[h];
            W_k[h] += offsetk[h];
            W_v[h] += offsetv[h];
        }
        W_o += offseto;

        NormalizeWeights();
    }


    public void AddLimitedNoise(float absOffset, float percentChanged)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        List<MatrixF> offsetq = new();
        List<MatrixF> offsetk = new();
        List<MatrixF> offsetv = new();
        for (int h = 0; h < NumHeads; h++)
        {
            offsetq.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetk.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetv.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        // For each element in each matrix, only keep the noise if a random draw is less than percentChanged.
        for (int h = 0; h < NumHeads; h++)
        {
            for (int i = 0; i < ModelDim; i++)
            {
                for (int j = 0; j < ModelDim; j++)
                {
                    if (random.NextDouble() >= percentChanged)
                        offsetq[h][i, j] = 0f;
                    if (random.NextDouble() >= percentChanged)
                        offsetk[h][i, j] = 0f;
                    if (random.NextDouble() >= percentChanged)
                        offsetv[h][i, j] = 0f;
                }
            }
        }
        for (int i = 0; i < ModelDim * NumHeads; i++)
        {
            for (int j = 0; j < ModelDim; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    offseto[i, j] = 0f;
            }
        }

        // Apply the (sparsified) noise offsets.
        for (int h = 0; h < NumHeads; h++)
        {
            W_q[h] += offsetq[h];
            W_k[h] += offsetk[h];
            W_v[h] += offsetv[h];
        }
        W_o += offseto;

        // NormalizeWeights();
    }


    public void NormalizeWeights()
    {
        for (int h = 0; h < NumHeads; h++)
        {
            W_q[h].TanhNormalize();
            W_k[h].TanhNormalize();
            W_v[h].TanhNormalize();
        }
        W_o.TanhNormalize();
    }

    public SelfAttentionNoise CreateNoise(float absOffset)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        // Initialize weight matrices for each head.
        List<MatrixF> offsetq = new();
        List<MatrixF> offsetk = new();
        List<MatrixF> offsetv = new();
        for (int h = 0; h < NumHeads; h++)
        {
            offsetq.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetk.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetv.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        SelfAttentionNoise noise = new SelfAttentionNoise();
        noise.W_qn = offsetq;
        noise.W_kn = offsetk;
        noise.W_vn = offsetv;
        noise.W_on = offseto;

        return noise;
    }

    public SelfAttentionNoise CreateLimitedNoise(float absOffset, float percentChanged)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        List<MatrixF> offsetq = new();
        List<MatrixF> offsetk = new();
        List<MatrixF> offsetv = new();
        for (int h = 0; h < NumHeads; h++)
        {
            offsetq.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetk.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
            offsetv.Add(DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax)));
        }
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim * NumHeads, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        // For each element in each matrix, only keep the noise if a random draw is less than percentChanged.
        for (int h = 0; h < NumHeads; h++)
        {
            for (int i = 0; i < ModelDim; i++)
            {
                for (int j = 0; j < ModelDim; j++)
                {
                    if (random.NextDouble() >= percentChanged)
                        offsetq[h][i, j] = 0f;
                    if (random.NextDouble() >= percentChanged)
                        offsetk[h][i, j] = 0f;
                    if (random.NextDouble() >= percentChanged)
                        offsetv[h][i, j] = 0f;
                }
            }
        }
        for (int i = 0; i < ModelDim * NumHeads; i++)
        {
            for (int j = 0; j < ModelDim; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    offseto[i, j] = 0f;
            }
        }

        SelfAttentionNoise noise = new SelfAttentionNoise();
        noise.W_qn = offsetq;
        noise.W_kn = offsetk;
        noise.W_vn = offsetv;
        noise.W_on = offseto;

        return noise;
    }

    public void ApplyNoise(SelfAttentionNoise noise)
    {
        for (int h = 0; h < NumHeads; h++)
        {
            W_q[h] += noise.W_qn[h];
            W_k[h] += noise.W_kn[h];
            W_v[h] += noise.W_vn[h];
        }
        W_o += noise.W_on;

        NormalizeWeights();
    }


    // --------------------------------------------------------------------------------------------

    public int ParamCount()
    {
        int count = 0;
        for (int h = 0; h < NumHeads; h++)
        {
            count += W_q[h].RowCount * W_q[h].ColumnCount;
            count += W_k[h].RowCount * W_k[h].ColumnCount;
            count += W_v[h].RowCount * W_v[h].ColumnCount;
        }
        count += W_o.RowCount * W_o.ColumnCount;
        return count;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    // Applies the self-attention mechanism to the given input.
    // The input matrix with shape (sequenceLength, ModelDim) where each row is an embedded token.
    // The output matrix after applying self-attention. Its shape is (sequenceLength, ModelDim).
    public MatrixF Forward(MatrixF input)
    {
        // Compute queries, keys, and values.
        // If input has shape (n, ModelDim) and W_x has shape (ModelDim, ModelDim), then the result is (n, ModelDim).
        List<MatrixF> headContexts = new();
        for (int h = 0; h < NumHeads; h++)
        {
            MatrixF Q = input * W_q[h];
            MatrixF K = input * W_k[h];
            MatrixF V = input * W_v[h];

            MatrixF scores = Q * K.Transpose();
            float scale = 1.0f / MathF.Sqrt(ModelDim);
            scores = scores.Multiply(scale);
            MatrixF attentionWeights = SoftmaxRows(scores);
            MatrixF context = attentionWeights * V;
            headContexts.Add(context);
        }

        int n = input.RowCount;
        MatrixF concat = DenseMatrix.Create(n, ModelDim * NumHeads, 0f);
        for (int h = 0; h < NumHeads; h++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < ModelDim; j++)
                {
                    concat[i, h * ModelDim + j] = headContexts[h][i, j];
                }
            }
        }

        MatrixF output = concat * W_o;

        return output;
    }

    // --------------------------------------------------------------------------------------------

    // Applies the softmax function to each row of the given matrix.
    private MatrixF SoftmaxRows(MatrixF matrix)
    {
        // Clone the input to avoid modifying the original matrix.
        MatrixF result = matrix.Clone();

        for (int i = 0; i < matrix.RowCount; i++)
        {
            // Get the i-th row.
            VectorF row = matrix.Row(i);

            // For numerical stability, subtract the row maximum.
            float max = row.Maximum();
            VectorF expRow = row.Subtract(max).Map(x => MathF.Exp(x));

            // Sum of exponentials.
            float sum = expRow.Sum();

            // Normalize to form the softmax distribution.
            VectorF softmaxRow = expRow.Divide(sum);

            // Copy the normalized values back into the result matrix.
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                result[i, j] = softmaxRow[j];
            }
        }

        return result;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Report
    // --------------------------------------------------------------------------------------------

    public string Report()
    {
        return $"SelfAttention // Heads: {NumHeads} // Input Length: {InputLen} // Model Dimension: {ModelDim} // Parameter Count: {ParamCount()} // O Shape: {W_o.RowCount}x{W_o.ColumnCount} // Checksum: {CheckSum()}";
    }

    public float CheckSum()
    {
        float sum = 0f;
        for (int h = 0; h < NumHeads; h++)
        {
            sum += W_q[h].RowSums().Sum();
            sum += W_k[h].RowSums().Sum();
            sum += W_v[h].RowSums().Sum();
        }
        sum += W_o.RowSums().Sum();
        return sum;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------

    // Save the self-attention layer to a file.
    public void SaveToFile(string path)
    {
        using (var writer = new StreamWriter(path))
        {
            // Write the model dimension.
            writer.WriteLine(InputLen);
            writer.WriteLine(ModelDim);
            writer.WriteLine(NumHeads);

            for (int h = 0; h < NumHeads; h++)
            {
                writer.WriteLine(MatrixOperations.MatrixToString(W_q[h]));
                writer.WriteLine(MatrixOperations.MatrixToString(W_k[h]));
                writer.WriteLine(MatrixOperations.MatrixToString(W_v[h]));
            }

            writer.WriteLine(MatrixOperations.MatrixToString(W_o));
        }
    }

    // --------------------------------------------------------------------------------------------

    // Load a self-attention layer from a file.
    public static SelfAttention LoadFromFile(string path)
    {
        using (var reader = new StreamReader(path))
        {
            // Read the model dimension.
            int inputLen = int.Parse(reader.ReadLine());
            int modelDim = int.Parse(reader.ReadLine());
            int numHeads = int.Parse(reader.ReadLine());

            SelfAttention layer = new SelfAttention(inputLen, modelDim, numHeads);

            for (int h = 0; h < numHeads; h++)
            {
                string? q_string = reader.ReadLine();
                string? k_string = reader.ReadLine();
                string? v_string = reader.ReadLine();
                if (q_string == null || k_string == null || v_string == null)
                    throw new ArgumentException("Input processing error");
                MatrixOperations.TryStringToMatrix(q_string, out var qMat);
                MatrixOperations.TryStringToMatrix(k_string, out var kMat);
                MatrixOperations.TryStringToMatrix(v_string, out var vMat);
                layer.W_q[h] = qMat!;
                layer.W_k[h] = kMat!;
                layer.W_v[h] = vMat!;
            }
            string? o_string = reader.ReadLine();
            if (o_string == null)
                throw new ArgumentException("Input processing error");
            MatrixOperations.TryStringToMatrix(o_string, out var oMat);
            layer.W_o = oMat!;

            return layer;
        }
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Binary Load Save
    // --------------------------------------------------------------------------------------------

    // Save the self-attention layer to a binary file.
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
                    writer.Write(InputLen);
                    writer.Write(ModelDim);
                    writer.Write(NumHeads);

                    for (int h = 0; h < NumHeads; h++)
                    {
                        for (int i = 0; i < ModelDim; i++)
                        {
                            for (int j = 0; j < ModelDim; j++)
                            {
                                writer.Write(W_q[h][i, j]);
                            }
                        }

                        for (int i = 0; i < ModelDim; i++)
                        {
                            for (int j = 0; j < ModelDim; j++)
                            {
                                writer.Write(W_k[h][i, j]);
                            }
                        }

                        for (int i = 0; i < ModelDim; i++)
                        {
                            for (int j = 0; j < ModelDim; j++)
                            {
                                writer.Write(W_v[h][i, j]);
                            }
                        }
                    }

                    for (int i = 0; i < ModelDim * NumHeads; i++)
                    {
                        for (int j = 0; j < ModelDim; j++)
                        {
                            writer.Write(W_o[i, j]);
                        }
                    }
                }
                break;
            }
            catch (System.IO.IOException)
            {
                if (++retries >= maxRetries)
                    throw;
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }

    // --------------------------------------------------------------------------------------------

    // Load a self-attention layer from a binary file.

    public static SelfAttention LoadFromBinary(string path)
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
                    int inputLen = reader.ReadInt32();
                    int modelDim = reader.ReadInt32();
                    int numHeads = reader.ReadInt32();

                    SelfAttention layer = new SelfAttention(inputLen, modelDim, numHeads);

                    for (int h = 0; h < numHeads; h++)
                    {
                        for (int i = 0; i < modelDim; i++)
                        {
                            for (int j = 0; j < modelDim; j++)
                            {
                                layer.W_q[h][i, j] = reader.ReadSingle();
                            }
                        }

                        for (int i = 0; i < modelDim; i++)
                        {
                            for (int j = 0; j < modelDim; j++)
                            {
                                layer.W_k[h][i, j] = reader.ReadSingle();
                            }
                        }

                        for (int i = 0; i < modelDim; i++)
                        {
                            for (int j = 0; j < modelDim; j++)
                            {
                                layer.W_v[h][i, j] = reader.ReadSingle();
                            }
                        }
                    }

                    for (int i = 0; i < modelDim * numHeads; i++)
                    {
                        for (int j = 0; j < modelDim; j++)
                        {
                            layer.W_o[i, j] = reader.ReadSingle();
                        }
                    }

                    return layer;
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
