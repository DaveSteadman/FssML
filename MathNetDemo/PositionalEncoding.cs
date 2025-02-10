using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;  // For Single precision (float)
using MathNet.Numerics.Distributions;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

public static class PositionalEncoder
{
    public static float[,] GetPositionalEncoding(int seqLength, int dModel)
    {
        float[,] pe = new float[seqLength, dModel];
        for (int pos = 0; pos < seqLength; pos++)
        {
            for (int i = 0; i < dModel; i++)
            {
                // Compute the denominator: 10000^(i/dModel) for even indices,
                // use the same value for the odd index immediately after.
                double denominator = Math.Pow(10000, (2 * (i / 2)) / (double)dModel);
                double angle = pos / denominator;

                if (i % 2 == 0)
                    pe[pos, i] = (float)Math.Sin(angle);
                else
                    pe[pos, i] = (float)Math.Cos(angle);
            }
        }
        return pe;
    }

    public
}