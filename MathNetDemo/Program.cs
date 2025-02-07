
// Setup Instructions
// ==================
// dotnet new console -n MathNetDemo
// cd MathNetDemo
// dotnet add package MathNet.Numerics
// dotnet run

// Update environment
// ==================
// dotnet workload list
// dotnet workload search
// dotnet workload update

using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
//using MathNet.Numerics.LinearAlgebra.Double;

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
                if (vocabulary.Count >= 1000)
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

            Dictionary<string, int> vocabTrimmed = BPETokenizer.LimitVocabSize(vocabulary, 1000);
            BPETokenizer.SaveVocabularyToFile(vocabTrimmed, "./vocab.json");
        }

        // --------------------------------------------------------------------------------------------

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

        // --------------------------------------------------------------------------------------------

        public static void DemoEmbeddings()
        {
            // Load vocab
            Dictionary<string, int> vocabulary = BPETokenizer.LoadVocabularyFromFile("./vocab.json");

            // Example: Assume your BPE tokenizer produced a vocabulary of 100 tokens.
            int vocabSize    = vocabulary.Count;
            int embeddingDim = 16;  // Choose the embedding dimension as a hyperparameter.

            // Load the input string from file
            string input = File.ReadAllText("SampleStr.txt");
            List<string> tokList   = BPETokenizer.TokenizeUsingVocabulary(input, vocabulary);
            List<int>    tokIdList = BPETokenizer.TokenIdsUsingVocabulary(input, vocabulary);

            // Create an embedding layer using Single precision.
            EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
            embeddingLayer.SetRandom(-0.5f, 0.5f);

            int numToks = 15;
            for (int i = 0; i < numToks; i++)
            {
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            }

            // Create a sub-list of the first 50 tokens IDs
            List<int> tokenIds = tokIdList.Take(5).ToList();

            // Look up embeddings for these token IDs.
            List<VectorF> embeddings = embeddingLayer.LookupList(tokenIds);

            Console.WriteLine("Embeddings for token IDs:");
            for (int i = 0; i < tokenIds.Count; i++)
            {
                Console.WriteLine($"Token ID {tokenIds[i]}: {embeddings[i].ToString()}");
            }

            // Save the embeddings
            EmbeddingLayer.Save(embeddingLayer, "./embeddings.json");
            Console.WriteLine($"Embeddings saved to file. Total {embeddingLayer.ParamCount()} parameters.");
        }

        // --------------------------------------------------------------------------------------------

        public static void DemoSelfAttention()
        {


            // Load the vocabulary
            Dictionary<string, int> vocabulary = BPETokenizer.LoadVocabularyFromFile("./vocab.json");

            int vocabSize    = vocabulary.Count;
            int embeddingDim = 16;

            // // Tokenize the input string using the vocabulary
            // List<string> tokens   = BPETokenizer.TokenizeUsingVocabulary(input, vocabulary);
            // List<int>    tokenIds = BPETokenizer.TokenIdsUsingVocabulary(input, vocabulary);

            // // Create an embedding layer
            bool createEmbeddings = true;
            EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);

            if (createEmbeddings)
            {
                embeddingLayer.SetRandom(-0.2f, 0.2f);
                EmbeddingLayer.Save(embeddingLayer, "./embeddings.json");
            }
            else
            {
                embeddingLayer = EmbeddingLayer.Load("./embeddings.json");
            }

            // // Create a self-attention layer
            SelfAttention selfAttention = new SelfAttention(embeddingDim);

            // Create a pass of a single strong through the system.
            {
                // Load the input string from file
                string input = File.ReadAllText("SampleStr.txt");
                List<string> tokList   = BPETokenizer.TokenizeUsingVocabulary(input, vocabulary);
                List<int>    tokIdList = BPETokenizer.TokenIdsUsingVocabulary(input, vocabulary);

                int numTokens = 10;
                List<int>     tokenIdsSub = tokIdList.Take(numTokens).ToList();
                List<VectorF> embeddings  = embeddingLayer.LookupList(tokenIdsSub);
                MatrixF       inputMatrix = DenseMatrix.OfRowVectors(embeddings);

                for (int i = 0; i < numTokens; i++)
                {
                    Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
                }

            }

            // // Perform self-attention on the input matrix
            // MatrixF outputMatrix = selfAttention.Forward(inputMatrix);

            // // Save the self-attention layer
            // SelfAttention.Save(selfAttention, "./self-attention.json");
            // Console.WriteLine("Self-attention layer saved to file.");

            // // Load the self-attention layer
            // SelfAttention loadedLayer = SelfAttention.Load("./self-attention.json");
            // Console.WriteLine("Self-attention layer loaded from file.");

            // // Perform self-attention on the input matrix
            // MatrixF loadedOutputMatrix = loadedLayer.Forward(inputMatrix);

            // // Compare the output matrices
            // Console.WriteLine("Output matrices are equal: " + outputMatrix.Equals(loadedOutputMatrix));



            SelfAttention.Save(selfAttention, "./self-attention.json");
        }

        static void Main(string[] args)
        {
            //DemoBPE();
            //DemoMatrix();
            //DemoEmbeddings();
            DemoSelfAttention();
        }
    }
}

