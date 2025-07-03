using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;

public class FeedForwardLayerNoise
{
    public MatrixF W1n { get; set; }
    public VectorF b1n { get; set; }
    public MatrixF W2n { get; set; }
    public VectorF b2n { get; set; }
}

public class FeedForwardLayer
{
    // Input/output dimension (d_model) and hidden dimension (d_ff)
    private int d_model;
    private int d_ff;

    private static readonly Random random = new Random();

    // Weight matrices and bias vectors for the two linear transformations
    public MatrixF W1 { get; private set; }
    public VectorF b1 { get; private set; }
    public MatrixF W2 { get; private set; }
    public VectorF b2 { get; private set; }

    public int ParamCount()
    {
        return d_model * d_ff + d_ff + d_ff * d_model + d_model;
    }

    public string Report()
    {
        return $"FeedForwardLayer // d_model: {d_model} // d_ff: {d_ff} // CheckSum: {CheckSum()}";
    }

    public float CheckSum()
    {
        float sum = W1.RowSums().Sum() + b1.Sum() + W2.RowSums().Sum() + b2.Sum();
        return sum;
    }

    /// <summary>
    /// Constructs the feed-forward layer.
    /// d_model: the input and output dimension (e.g. embedding size)
    /// d_ff: the hidden layer dimension (typically 4 times d_model)
    /// </summary>
    public FeedForwardLayer(int d_model, int d_ff)
    {
        this.d_model = d_model;
        this.d_ff    = d_ff;

        // Initialize weights randomly.
        // In practice you might want to initialize with a specific distribution or scale.
        W1 = MatrixF.Build.Random(d_model, d_ff);
        b1 = VectorF.Build.Dense(d_ff, 0.0f);
        W2 = MatrixF.Build.Random(d_ff, d_model);
        b2 = VectorF.Build.Dense(d_model, 0.0f);
    }

    // ReLU activation function.
    private float ReLU(float x)
    {
        return x > 0 ? x : 0;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    public FeedForwardLayer DeepCopy()
    {
        FeedForwardLayer newLayer = new FeedForwardLayer(d_model, d_ff);
        newLayer.W1 = W1.Clone();
        newLayer.b1 = b1.Clone();
        newLayer.W2 = W2.Clone();
        newLayer.b2 = b2.Clone();
        return newLayer;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Mutation
    // --------------------------------------------------------------------------------------------


    public void SetRandom()
    {
        W1 = MatrixF.Build.Random(d_model, d_ff);
        b1 = VectorF.Build.Dense(d_ff, 0.0f);
        W2 = MatrixF.Build.Random(d_ff, d_model);
        b2 = VectorF.Build.Dense(d_model, 0.0f);
    }

    public void AddNoise(float absOffset)
    {
        // create a noise matrix
        MatrixF W1_noise = DenseMatrix.Build.Random(d_model, d_ff, new ContinuousUniform(-absOffset, absOffset));
        MatrixF W2_noise = DenseMatrix.Build.Random(d_ff, d_model, new ContinuousUniform(-absOffset, absOffset));

        // add noise to the weights
        W1 = W1 + W1_noise;
        W2 = W2 + W2_noise;

        // create a noise vector
        VectorF b1_noise = DenseVector.Build.Random(d_ff, new ContinuousUniform(-absOffset, absOffset));
        VectorF b2_noise = DenseVector.Build.Random(d_model, new ContinuousUniform(-absOffset, absOffset));

        // add noise to the biases
        b1 = b1 + b1_noise;
        b2 = b2 + b2_noise;
    }

    public void AddLimitedNoise(float absOffset, float percentChanged)
    {
        MatrixF W1_noise = DenseMatrix.Build.Random(d_model, d_ff, new ContinuousUniform(-absOffset, absOffset));
        MatrixF W2_noise = DenseMatrix.Build.Random(d_ff, d_model, new ContinuousUniform(-absOffset, absOffset));
        VectorF b1_noise = DenseVector.Build.Random(d_ff, new ContinuousUniform(-absOffset, absOffset));
        VectorF b2_noise = DenseVector.Build.Random(d_model, new ContinuousUniform(-absOffset, absOffset));

        for (int i = 0; i < d_model; i++)
        {
            for (int j = 0; j < d_ff; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    W1_noise[i, j] = 0f;
            }
        }

        for (int i = 0; i < d_ff; i++)
        {
            for (int j = 0; j < d_model; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    W2_noise[i, j] = 0f;
            }
        }

        for (int i = 0; i < d_ff; i++)
        {
            if (random.NextDouble() >= percentChanged)
                b1_noise[i] = 0f;
        }

        for (int i = 0; i < d_model; i++)
        {
            if (random.NextDouble() >= percentChanged)
                b2_noise[i] = 0f;
        }

        W1 += W1_noise;
        W2 += W2_noise;
        b1 += b1_noise;
        b2 += b2_noise;
    }

    // --------------------------------------------------------------------------------------------

    public FeedForwardLayerNoise CreateNoise(float absOffset)
    {
        // create a noise matrix
        MatrixF W1_noise = DenseMatrix.Build.Random(d_model, d_ff, new ContinuousUniform(-absOffset, absOffset));
        MatrixF W2_noise = DenseMatrix.Build.Random(d_ff, d_model, new ContinuousUniform(-absOffset, absOffset));

        // create a noise vector
        VectorF b1_noise = DenseVector.Build.Random(d_ff, new ContinuousUniform(-absOffset, absOffset));
        VectorF b2_noise = DenseVector.Build.Random(d_model, new ContinuousUniform(-absOffset, absOffset));

        FeedForwardLayerNoise noise = new FeedForwardLayerNoise();
        noise.W1n = W1_noise;
        noise.b1n = b1_noise;
        noise.W2n = W2_noise;
        noise.b2n = b2_noise;

        return noise;
    }

    public FeedForwardLayerNoise CreateLimitedNoise(float absOffset, float percentChanged)
    {
        // create a noise matrix
        MatrixF W1_noise = DenseMatrix.Build.Random(d_model, d_ff, new ContinuousUniform(-absOffset, absOffset));
        MatrixF W2_noise = DenseMatrix.Build.Random(d_ff, d_model, new ContinuousUniform(-absOffset, absOffset));

        // create a noise vector
        VectorF b1_noise = DenseVector.Build.Random(d_ff, new ContinuousUniform(-absOffset, absOffset));
        VectorF b2_noise = DenseVector.Build.Random(d_model, new ContinuousUniform(-absOffset, absOffset));

        for (int i = 0; i < d_model; i++)
        {
            for (int j = 0; j < d_ff; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    W1_noise[i, j] = 0f;
            }
        }

        for (int i = 0; i < d_ff; i++)
        {
            for (int j = 0; j < d_model; j++)
            {
                if (random.NextDouble() >= percentChanged)
                    W2_noise[i, j] = 0f;
            }
        }

        for (int i = 0; i < d_ff; i++)
        {
            if (random.NextDouble() >= percentChanged)
                b1_noise[i] = 0f;
        }

        for (int i = 0; i < d_model; i++)
        {
            if (random.NextDouble() >= percentChanged)
                b2_noise[i] = 0f;
        }

        FeedForwardLayerNoise noise = new FeedForwardLayerNoise();
        noise.W1n = W1_noise;
        noise.b1n = b1_noise;
        noise.W2n = W2_noise;
        noise.b2n = b2_noise;

        return noise;
    }

    // --------------------------------------------------------------------------------------------

    public void AddNoise(FeedForwardLayerNoise noise)
    {
        // add noise to the weights
        W1 = W1 + noise.W1n;
        W2 = W2 + noise.W2n;

        // add noise to the biases
        b1 = b1 + noise.b1n;
        b2 = b2 + noise.b2n;
    }

    public void ApplyNoise(FeedForwardLayerNoise noise)
    {
        W1 += noise.W1n;
        b1 += noise.b1n;
        W2 += noise.W2n;
        b2 += noise.b2n;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------


    /// <summary>
    /// Forward pass for the feed-forward layer.
    /// The input is a matrix of shape [numTokens x d_model].
    /// The output is a matrix of the same shape.
    /// </summary>
    // Usage: MatrixF output = feedForwardLayer.Forward(input);
    public MatrixF Forward(MatrixF input)
    {
        // First linear layer: compute (input * W1 + b1).
        // Resulting shape: [numTokens x d_ff]
        MatrixF hiddenLinear = input * W1;
        // Add bias and apply ReLU activation element-wise.
        for (int i = 0; i < hiddenLinear.RowCount; i++)
        {
            for (int j = 0; j < hiddenLinear.ColumnCount; j++)
            {
                hiddenLinear[i, j] = ReLU(hiddenLinear[i, j] + b1[j]);
            }
        }

        // Second linear layer: compute (hidden * W2 + b2).
        // This projects back to the original dimension, shape: [numTokens x d_model]
        MatrixF output = hiddenLinear * W2;
        for (int i = 0; i < output.RowCount; i++)
        {
            for (int j = 0; j < output.ColumnCount; j++)
            {
                output[i, j] = output[i, j] + b2[j];
            }
        }

        return output;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string path)
    {
        using (var writer = new StreamWriter(path))
        {
            string header = $"{d_model} {d_ff}";
            writer.WriteLine(header);

            string w1_string = MatrixOperations.MatrixToString(W1);
            string b1_string = MatrixOperations.VectorToString(b1);
            string w2_string = MatrixOperations.MatrixToString(W2);
            string b2_string = MatrixOperations.VectorToString(b2);

            writer.WriteLine(w1_string);
            writer.WriteLine(b1_string);
            writer.WriteLine(w2_string);
            writer.WriteLine(b2_string);
        }
    }

    // --------------------------------------------------------------------------------------------

    public static FeedForwardLayer LoadFromFile(string path)
    {
        using (var reader = File.OpenText(path))
        {
            // Read the model dimension.
            string[] header       = reader.ReadLine().Split(' ');
            int      new_d_model  = int.Parse(header[0]);
            int      new_d_ff     = int.Parse(header[1]);

            string? w1_string = reader.ReadLine();
            string? b1_string = reader.ReadLine();
            string? w2_string = reader.ReadLine();
            string? b2_string = reader.ReadLine();

            // check the strings are valid
            if (w1_string == null || b1_string == null || w2_string == null || b2_string == null)
                throw new ArgumentException("Input processing error");

            MatrixF newW1;
            VectorF newB1;
            MatrixF newW2;
            VectorF newB2;

            // Load each weight matrix and assign to the corresponding property.
            bool w1_read_ok = (MatrixOperations.TryStringToMatrix(w1_string, out newW1!));
            bool b1_read_ok = (MatrixOperations.TryStringToVector(b1_string, out newB1!));
            bool w2_read_ok = (MatrixOperations.TryStringToMatrix(w2_string, out newW2!));
            bool b2_read_ok = (MatrixOperations.TryStringToVector(b2_string, out newB2!));

            if (!w1_read_ok || !b1_read_ok || !w2_read_ok || !b2_read_ok)
                 throw new ArgumentException("Matrix parsing error");

            // Create a new FeedForwardLayer instance.
            FeedForwardLayer layer = new FeedForwardLayer(new_d_model, new_d_ff);

            if (w1_read_ok) layer.W1 = newW1!;
            if (b1_read_ok) layer.b1 = newB1!;
            if (w2_read_ok) layer.W2 = newW2!;
            if (b2_read_ok) layer.b2 = newB2!;

            return layer;
        }
    }

    // -------------------------------------------------------------------------
    // MARK: Binary Load Save
    // -------------------------------------------------------------------------

    public void SaveToBinary(string path)
    {
        const int maxRetries = 10;
        const int delayMs = 100;
        int retries = 0;
        while (true)
        {
            try
            {
                using (var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.None)))
                {
                    writer.Write(d_model);
                    writer.Write(d_ff);
                    for (int i = 0; i < d_model; i++)
                    {
                        for (int j = 0; j < d_ff; j++)
                        {
                            writer.Write(W1[i, j]);
                        }
                    }
                    for (int i = 0; i < d_ff; i++)
                    {
                        writer.Write(b1[i]);
                    }
                    for (int i = 0; i < d_ff; i++)
                    {
                        for (int j = 0; j < d_model; j++)
                        {
                            writer.Write(W2[i, j]);
                        }
                    }
                    for (int i = 0; i < d_model; i++)
                    {
                        writer.Write(b2[i]);
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

    public static FeedForwardLayer LoadFromBinary(string path)
    {
        const int maxRetries = 10;
        const int delayMs = 100;
        int retries = 0;
        while (true)
        {
            try
            {
                using (var reader = new BinaryReader(File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read)))
                {
                    int new_d_model = reader.ReadInt32();
                    int new_d_ff = reader.ReadInt32();
                    FeedForwardLayer layer = new FeedForwardLayer(new_d_model, new_d_ff);
                    for (int i = 0; i < new_d_model; i++)
                    {
                        for (int j = 0; j < new_d_ff; j++)
                        {
                            layer.W1[i, j] = reader.ReadSingle();
                        }
                    }
                    for (int i = 0; i < new_d_ff; i++)
                    {
                        layer.b1[i] = reader.ReadSingle();
                    }
                    for (int i = 0; i < new_d_ff; i++)
                    {
                        for (int j = 0; j < new_d_model; j++)
                        {
                            layer.W2[i, j] = reader.ReadSingle();
                        }
                    }
                    for (int i = 0; i < new_d_model; i++)
                    {
                        layer.b2[i] = reader.ReadSingle();
                    }
                    return layer;
                }
            }
            catch (IOException)
            {
                if (++retries >= maxRetries)
                    throw;
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }
}

