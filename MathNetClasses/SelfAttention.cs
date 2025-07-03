using System;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

// Alias for brevity.
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class SelfAttentionNoise
{
    public MatrixF W_qn { get; set; }
    public MatrixF W_kn { get; set; }
    public MatrixF W_vn { get; set; }
    public MatrixF W_on { get; set; }
}


public class SelfAttention
{
    // The dimensionality of the model (and of the query, key, value vectors).
    public int InputLen { get; private set;}
    public int ModelDim { get; private set; }

    // Weight matrix for computing queries. Shape: (ModelDim, ModelDim)
    public MatrixF W_q { get; private set; }

    // Weight matrix for computing keys. Shape: (ModelDim, ModelDim)
    public MatrixF W_k { get; private set; }

    // Weight matrix for computing values. Shape: (ModelDim, ModelDim)
    public MatrixF W_v { get; private set; }

    // Output projection weight matrix. Shape: (ModelDim, ModelDim)
    public MatrixF W_o { get; private set; }

    private static readonly Random random = new Random();

    // --------------------------------------------------------------------------------------------

    // Creates a new self-attention layer with the given model dimensionality.
    // modelDim: The size of the model and each token’s embedding vector.</param>
    public SelfAttention(int inputLen, int modelDim)
    {
        InputLen = inputLen;
        ModelDim = modelDim;

        // Create a uniform distribution for small random initialization.
        float rangeMin = -1f;
        float rangeMax =  1f;

        // Initialize weight matrices.
        W_q = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_k = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_v = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_o = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    public SelfAttention DeepCopy()
    {
        SelfAttention newLayer = new SelfAttention(InputLen, ModelDim);
        newLayer.W_q = W_q.Clone();
        newLayer.W_k = W_k.Clone();
        newLayer.W_v = W_v.Clone();
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

        // Initialize weight matrices.
        W_q = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_k = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_v = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        W_o = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
    }

    public void AddNoise(float absOffset)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        // A different offset is applied to each value in each matrix.
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        // Initialize weight matrices.
        MatrixF offsetq = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetk = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetv = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        // Apply the offset to each element in each matrix.
        W_q += offsetq;
        W_k += offsetk;
        W_v += offsetv;
        W_o += offseto;

        NormalizeWeights();
    }


    public void AddLimitedNoise(float absOffset, float percentChanged)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        // Generate initial offset matrices with random noise.
        MatrixF offsetq = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetk = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetv = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        // For each element in each matrix, only keep the noise if a random draw is less than percentChanged.
        for (int i = 0; i < ModelDim; i++)
        {
            for (int j = 0; j < ModelDim; j++)
            {
                if (random.NextDouble() >= percentChanged)
                {
                    offsetq[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offsetk[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offsetv[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offseto[i, j] = 0f;
                }
            }
        }

        // Apply the (sparsified) noise offsets.
        W_q += offsetq;
        W_k += offsetk;
        W_v += offsetv;
        W_o += offseto;

        // NormalizeWeights();
    }


    public void NormalizeWeights()
    {
        W_q.TanhNormalize();
        W_k.TanhNormalize();
        W_v.TanhNormalize();
        W_o.TanhNormalize();
    }

    public SelfAttentionNoise CreateNoise(float absOffset)
    {
        // The offset for each value is plus/minus a random value in the range [-absOffset, absOffset].
        float rangeMin = -absOffset;
        float rangeMax = absOffset;

        // Initialize weight matrices.
        MatrixF offsetq = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetk = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetv = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

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

        // Generate initial offset matrices with random noise.
        MatrixF offsetq = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetk = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offsetv = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));
        MatrixF offseto = DenseMatrix.Build.Random(ModelDim, ModelDim, new ContinuousUniform(rangeMin, rangeMax));

        // For each element in each matrix, only keep the noise if a random draw is less than percentChanged.
        for (int i = 0; i < ModelDim; i++)
        {
            for (int j = 0; j < ModelDim; j++)
            {
                if (random.NextDouble() >= percentChanged)
                {
                    offsetq[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offsetk[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offsetv[i, j] = 0f;
                }
                if (random.NextDouble() >= percentChanged)
                {
                    offseto[i, j] = 0f;
                }
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
        W_q += noise.W_qn;
        W_k += noise.W_kn;
        W_v += noise.W_vn;
        W_o += noise.W_on;

        NormalizeWeights();
    }


    // --------------------------------------------------------------------------------------------

    public int ParamCount()
    {
        return W_q.RowCount * W_q.ColumnCount +
               W_k.RowCount * W_k.ColumnCount +
               W_v.RowCount * W_v.ColumnCount +
               W_o.RowCount * W_o.ColumnCount;
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
        MatrixF Q = input * W_q;
        MatrixF K = input * W_k;
        MatrixF V = input * W_v;

        // Compute raw attention scores with Q * K^T.
        // This gives a score matrix of shape (n, n).
        MatrixF scores = Q * K.Transpose();

        // Scale the scores by 1/sqrt(ModelDim) for numerical stability.
        float scale = 1.0f / MathF.Sqrt(ModelDim);
        scores = scores.Multiply(scale);

        // Apply softmax to each row to obtain attention weights.
        MatrixF attentionWeights = SoftmaxRows(scores);

        // Multiply the attention weights by V to get the context (weighted sum).
        MatrixF context = attentionWeights * V;

        // Apply a final linear projection.
        MatrixF output = context * W_o;

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
        return $"SelfAttention // Input Length: {InputLen} // Model Dimension: {ModelDim} // Parameter Count: {ParamCount()} // Weight Shapes (RowxCol): [Q: {W_q.RowCount}x{W_q.ColumnCount}, K: {W_k.RowCount}x{W_k.ColumnCount}, V: {W_v.RowCount}x{W_v.ColumnCount}, O: {W_o.RowCount}x{W_o.ColumnCount}] // Checksum: {CheckSum()}";
    }

    public float CheckSum()
    {
        return W_q.RowSums().Sum() + W_k.RowSums().Sum() + W_v.RowSums().Sum() + W_o.RowSums().Sum();
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

            string q_string = MatrixOperations.MatrixToString(W_q);
            string k_string = MatrixOperations.MatrixToString(W_k);
            string v_string = MatrixOperations.MatrixToString(W_v);
            string o_string = MatrixOperations.MatrixToString(W_o);

            writer.WriteLine(q_string);
            writer.WriteLine(k_string);
            writer.WriteLine(v_string);
            writer.WriteLine(o_string);
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

            string? q_string = reader.ReadLine();
            string? k_string = reader.ReadLine();
            string? v_string = reader.ReadLine();
            string? o_string = reader.ReadLine();

            // check the strings are valid
            if (q_string == null || k_string == null || v_string == null || o_string == null)
                throw new ArgumentException("Input processing error");

            MatrixF newQ;
            MatrixF newK;
            MatrixF newV;
            MatrixF newO;

            // Load each weight matrix and assign to the corresponding property.
            bool q_read_ok = (MatrixOperations.TryStringToMatrix(q_string, out newQ!));
            bool k_read_ok = (MatrixOperations.TryStringToMatrix(k_string, out newK!));
            bool v_read_ok = (MatrixOperations.TryStringToMatrix(v_string, out newV!));
            bool o_read_ok = (MatrixOperations.TryStringToMatrix(o_string, out newO!));

            if (!q_read_ok || !k_read_ok || !v_read_ok || !o_read_ok)
                 throw new ArgumentException("Matrix parsing error");

            // Create a new SelfAttention instance.
            // (The constructor will initialize random weights, but we overwrite them below.)
            SelfAttention layer = new SelfAttention(inputLen, modelDim);

            if (q_read_ok) layer.W_q = newQ!;
            if (k_read_ok) layer.W_k = newK!;
            if (v_read_ok) layer.W_v = newV!;
            if (o_read_ok) layer.W_o = newO!;

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

                    for (int i = 0; i < ModelDim; i++)
                    {
                        for (int j = 0; j < ModelDim; j++)
                        {
                            writer.Write(W_q[i, j]);
                        }
                    }

                    for (int i = 0; i < ModelDim; i++)
                    {
                        for (int j = 0; j < ModelDim; j++)
                        {
                            writer.Write(W_k[i, j]);
                        }
                    }

                    for (int i = 0; i < ModelDim; i++)
                    {
                        for (int j = 0; j < ModelDim; j++)
                        {
                            writer.Write(W_v[i, j]);
                        }
                    }

                    for (int i = 0; i < ModelDim; i++)
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

                    SelfAttention layer = new SelfAttention(inputLen, modelDim);

                    for (int i = 0; i < modelDim; i++)
                    {
                        for (int j = 0; j < modelDim; j++)
                        {
                            layer.W_q[i, j] = reader.ReadSingle();
                        }
                    }

                    for (int i = 0; i < modelDim; i++)
                    {
                        for (int j = 0; j < modelDim; j++)
                        {
                            layer.W_k[i, j] = reader.ReadSingle();
                        }
                    }

                    for (int i = 0; i < modelDim; i++)
                    {
                        for (int j = 0; j < modelDim; j++)
                        {
                            layer.W_v[i, j] = reader.ReadSingle();
                        }
                    }

                    for (int i = 0; i < modelDim; i++)
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
