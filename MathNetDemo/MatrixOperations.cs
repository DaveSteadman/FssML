using System;
using System.Globalization;
using System.Linq;
using System.Text;

using MathNet.Numerics.LinearAlgebra.Single;  // For MatrixF

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;


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

    /// <summary>
    /// Applies a hyperbolic tangent (tanh) normalization to each element of the matrix.
    /// This function is useful for softly constraining values so that they remain within (-1, 1)
    /// while preserving the relative differences among small values.
    /// </summary>
    /// <param name="matrix">The input matrix.</param>
    /// <param name="scale">
    /// An optional scaling factor applied to each element before applying tanh.
    /// For small scale values, the function remains nearly linear; for larger scale values,
    /// the tanh saturates more aggressively. Default is 1.0f.
    /// </param>
    /// <returns>A new matrix with each element transformed by the tanh function.</returns>
    // Usage: MatrixF tanhNormalizedMatrix = MatrixOperations.TanhNormalize(matrix);
    //        MatrixF tanhNormalizedMatrixScaled = MatrixOperations.TanhNormalize(matrix, 0.5f);
    public static MatrixF TanhNormalize(this MatrixF matrix, float scale = 1.0f)
    {
        // Option 1: Using MathNet's Map function for conciseness.
        return matrix.Map(x => MathF.Tanh(scale * x));

        // Option 2: Explicit iteration (uncomment if you prefer loops)
        /*
        MatrixF result = DenseMatrix.Build.Dense(matrix.RowCount, matrix.ColumnCount);
        for (int i = 0; i < matrix.RowCount; i++)
        {
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                result[i, j] = MathF.Tanh(scale * matrix[i, j]);
            }
        }
        return result;
        */
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------

    // Helper method to load a matrix.
    public static MatrixF LoadMatrix(StreamReader reader)
    {
        // Read the header line that contains the dimensions.
        string[] dims = reader.ReadLine().Split(' ');
        int rows = int.Parse(dims[0]);
        int cols = int.Parse(dims[1]);

        // Create a new dense matrix.
        MatrixF matrix = DenseMatrix.Build.Dense(rows, cols);

        // Read each row.
        for (int i = 0; i < rows; i++)
        {
            string[] tokens = reader.ReadLine().Split(' ');
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = float.Parse(tokens[j]);
            }
        }
        return matrix;
    }


    /// <summary>
    /// Converts a MatrixF to a string using the format "rows;columns;v1,v2,...,vN".
    /// </summary>
    // Usage: string matrixString = MatrixOperations.MatrixToString(matrix);
    public static string MatrixToString(MatrixF matrix)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
        StringBuilder sb = new StringBuilder();

        sb.Append(rows.ToString(CultureInfo.InvariantCulture));
        sb.Append(';');
        sb.Append(cols.ToString(CultureInfo.InvariantCulture));
        sb.Append(';');

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sb.Append(matrix[i, j].ToString(CultureInfo.InvariantCulture));
                // Append a comma between elements (but not after the last element)
                if (!(i == rows - 1 && j == cols - 1))
                    sb.Append(',');
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// Attempts to parse a string into a MatrixF.
    /// Expected format: "rows;columns;v1,v2,...,vN".
    /// Returns true if successful; otherwise, false.
    /// </summary>
    // Usage: bool success = (MatrixOperations.TryStringToMatrix(matrixString, out MatrixF? matrix))
    public static bool TryStringToMatrix(string data, out MatrixF? matrix)
    {
        matrix = null;
        if (string.IsNullOrWhiteSpace(data))
            return false;

        // Expecting three parts: rows, columns, and comma-separated values.
        string[] parts = data.Split(';');
        if (parts.Length != 3)
            return false;

        if (!int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int rows))
            return false;
        if (!int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int cols))
            return false;

        string[] valueStrings = parts[2].Split(',');
        if (valueStrings.Length != rows * cols)
            return false;

        float[] values = new float[rows * cols];
        for (int i = 0; i < valueStrings.Length; i++)
        {
            if (!float.TryParse(valueStrings[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float val))
                return false;
            values[i] = val;
        }

        matrix = MatrixF.Build.Dense(rows, cols, values);
        return true;
    }

    // --------------------------------------------------------------------------------------------

    // Now matching routines for VectorF

    /// <summary>
    /// Converts a VectorF to a string using the format "v1,v2,...,vN".
    /// </summary>
    /// <param name="vector">The input vector.</param>
    /// <returns>A string representation of the vector.</returns>
    // Usage: string vectorString = MatrixOperations.VectorToString(vector);

    public static string VectorToString(VectorF vector)
    {
        int count = vector.Count;
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < count; i++)
        {
            sb.Append(vector[i].ToString(CultureInfo.InvariantCulture));
            // Append a comma between elements (but not after the last element)
            if (i < count - 1)
                sb.Append(',');
        }
        return sb.ToString();
    }

    /// <summary>
    /// Attempts to parse a string into a VectorF.
    /// Expected format: "v1,v2,...,vN".
    /// Returns true if successful; otherwise, false.
    /// </summary>
    /// <param name="data">The input string.</param>
    /// <param name="vector">The resulting vector.</param>
    /// <returns>True if the parsing was successful; otherwise, false.</returns>
    /// <remarks>If the parsing fails, the vector parameter will be set to null.</remarks>
    /// <example>
    /// <code>
    // if (MatrixOperations.TryStringToVector(vectorString, out VectorF? vector))
    public static bool TryStringToVector(string data, out VectorF? vector)
    {
        vector = null;
        if (string.IsNullOrWhiteSpace(data))
            return false;

        string[] valueStrings = data.Split(',');
        float[] values = new float[valueStrings.Length];
        for (int i = 0; i < valueStrings.Length; i++)
        {
            if (!float.TryParse(valueStrings[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float val))
                return false;
            values[i] = val;
        }

        vector = VectorF.Build.DenseOfArray(values);
        return true;
    }


}
