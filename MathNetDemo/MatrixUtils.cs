using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;  // For MatrixF
using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;

public static class MatrixOperations
{
    /// <summary>
    /// Normalizes all elements of the matrix to lie within the specified [newMin, newMax] range.
    /// </summary>
    /// <param name="matrix">The input matrix.</param>
    /// <param name="newMin">The new minimum value.</param>
    /// <param name="newMax">The new maximum value.</param>
    /// <returns>A new normalized matrix.</returns>
    // Usage: MatrixF normalizedMatrix = MatrixOperations.Normalize(matrix, -1.0f, 1.0f);
    public static MatrixF Normalize(this MatrixF matrix, float newMin, float newMax)
    {
        // Get all elements and compute their min and max.
        var values = matrix.Enumerate().ToArray();
        float oldMin = values.Min();
        float oldMax = values.Max();

        // Avoid division by zero if all values are the same.
        if (Math.Abs(oldMax - oldMin) < 1e-8)
        {
            return DenseMatrix.Build.Dense(matrix.RowCount, matrix.ColumnCount, newMin);
        }

        float oldRange = oldMax - oldMin;
        float newRange = newMax - newMin;
        MatrixF result = DenseMatrix.Build.Dense(matrix.RowCount, matrix.ColumnCount);

        for (int i = 0; i < matrix.RowCount; i++)
        {
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                result[i, j] = ((matrix[i, j] - oldMin) / oldRange) * newRange + newMin;
            }
        }
        return result;
    }
}
