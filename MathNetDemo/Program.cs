
// Setup Instructions
// ==================
// dotnet new console -n MathNetDemo
// cd MathNetDemo
// dotnet add package MathNet.Numerics
// dotnet run

using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>

namespace MathNetDemo
{
    class Program
    {
        public static void DemoBPE()
        {
            // Load the input string from file
            string input = File.ReadAllText("SampleStr.txt");

            // Write out tge first 100 characters of the input string
            Console.WriteLine("Original input: " + input.Substring(0, Math.Min(100, input.Length)));

            // Build the initial vocabulary and tokenize the string (character-by-character).
            Dictionary<string, int> vocabulary = BPETokenizer.BuildInitialVocabulary(input);
            List<string> tokens = BPETokenizer.Tokenize(input);

            // Writeout the first 100 tokens
            Console.WriteLine($"Initial vocabulary: {vocabulary.Count} tokens");
            Console.WriteLine(string.Join(" ", tokens.Take(100)));

            // Write first 100 token IDs from the dictionary
            Console.WriteLine("\nToken IDs:");
            Console.WriteLine(BPETokenizer.VocabularyToString(vocabulary, 100));

            // Perform several BPE iterations and update both tokens and vocabulary
            int iterationCount = 4800;
            for (int i = 0; i < iterationCount; i++)
            {
                // Write a debug line every 100 iterations
                if (i % 1000 == 0)
                    Console.WriteLine($"Performing BPE iteration: {i + 1}");

                // Update tokens and vocabulary using a BPE merge iteration.
                (tokens, vocabulary) = BPETokenizer.ApplyBPEIteration(tokens, vocabulary);

                // Break out of the loop if the vocabulary meets a target number
                if (vocabulary.Count >= 10000)
                {
                    Console.WriteLine($"Complete: {vocabulary.Count} tokens after {i + 1} iterations");
                    break;
                }
            }

            // Writeout a mix of 100 tokens and token IDs - sampling the vocabulary
            Random random = new Random();
            Console.WriteLine($"Final Vocab Examples: {vocabulary.Count} tokens");
            Console.WriteLine(BPETokenizer.VocabularyToString(vocabulary, 100));

            // Writeout the first 100 tokens
            Console.WriteLine($"Initial vocabulary: {vocabulary.Count} tokens");
            Console.WriteLine(string.Join(" ", tokens.Take(100)));
        }

        static void DemoMatrix()
        {
            // VECTOR OPERATIONS
            // Create two vectors using DenseVector
            Vector<double> vectorA = DenseVector.OfArray(new double[] { 1.0, 2.0, 3.0 });
            Vector<double> vectorB = DenseVector.OfArray(new double[] { 4.0, 5.0, 6.0 });

            // Compute the dot product of vectorA and vectorB
            double dotProduct = vectorA.DotProduct(vectorB);
            Console.WriteLine("Dot Product of vectorA and vectorB: " + dotProduct);

            // MATRIX OPERATIONS
            // Create a 2x3 matrix (matrixA)
            Matrix<double> matrixA = DenseMatrix.OfArray(new double[,] {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            // Create a 3x2 matrix (matrixB)
            Matrix<double> matrixB = DenseMatrix.OfArray(new double[,] {
                { 7, 8 },
                { 9, 10 },
                { 11, 12 }
            });

            // Multiply matrixA by matrixB resulting in a 2x2 matrix
            Matrix<double> matrixProduct = matrixA * matrixB;
            Console.WriteLine("Result of matrixA * matrixB:");
            Console.WriteLine(matrixProduct.ToString());

            // VECTOR-MATRIX MULTIPLICATION
            // Multiply matrixA (2x3) by a 3-element vector
            Vector<double> vectorC = DenseVector.OfArray(new double[] { 1, 2, 3 });
            Vector<double> resultVector = matrixA * vectorC;
            Console.WriteLine("Result of matrixA * vectorC:");
            Console.WriteLine(resultVector.ToString());
        }

        public static void DemoEmbeddings()
        {
            // Example: Assume your BPE tokenizer produced a vocabulary of 100 tokens.
            int vocabSize = 100;
            int embeddingDim = 16;  // Choose the embedding dimension as a hyperparameter.

            // Create an embedding layer using Single precision.
            EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);

            // Example token IDs obtained from your BPE tokenizer.
            List<int> tokenIds = new List<int> { 5, 20, 3, 10, 99, 20 };

            // Look up embeddings for these token IDs.
            List<VectorF> embeddings = embeddingLayer.Lookup(tokenIds);

            Console.WriteLine("Embeddings for token IDs:");
            for (int i = 0; i < tokenIds.Count; i++)
            {
                Console.WriteLine($"Token ID {tokenIds[i]}: {embeddings[i].ToString()}");
            }
        }

        static void Main(string[] args)
        {
            //DemoBPE();
            //DemoMatrix();
            DemoEmbeddings();
        }
    }
}

