using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

public static class VectorExtensions
{
    /// <summary>
    /// Returns a ranking vector for the input vector such that the highest value receives rank 1, the second highest rank 2, etc.
    /// </summary>
    public static Vector<float> Rank(this Vector<float> vector)
    {
        int n = vector.Count;
        // Create an array of indices [0, 1, 2, ..., n-1]
        int[] indices = Enumerable.Range(0, n).ToArray();
        // Sort indices by corresponding vector value descending
        Array.Sort(indices, (i, j) => vector[j].CompareTo(vector[i]));

        // Create an array to hold the rank for each position.
        float[] ranks = new float[n];
        for (int rank = 0; rank < n; rank++)
        {
            ranks[indices[rank]] = rank + 1;  // rank starting at 1
        }
        return Vector<float>.Build.DenseOfArray(ranks);
    }
}
