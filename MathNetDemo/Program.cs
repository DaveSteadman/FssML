
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

// Codespace Commands
// ==================
// git commit -am "recent progress"
// git push

using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

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
                if (vocabulary.Count >= 1200)
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

            // A save/load/save cycle flushes out probe tokens in the dictionary, such as control characters.
            // We then trim to the right length.
            // Any token we can't find will be replaced with the <unk> token.

            BPETokenizer.SaveVocabularyToFile(vocabulary, "./vocab.json");
            Dictionary<string, int> vocabLoaded = BPETokenizer.LoadVocabularyFromFile("./vocab.json");
            Dictionary<string, int> vocabTrimmed = BPETokenizer.LimitVocabSize(vocabLoaded, 1000);
            BPETokenizer.SaveVocabularyToFile(vocabTrimmed, "./vocab.json");
        }

        // --------------------------------------------------------------------------------------------

        static void DemoMatrix()
        {
            // VECTOR OPERATIONS
            // Create two vectors using DenseVector
            Vector<float> vectorA = DenseVector.OfArray(new float[] { 1.0f, 2.0f, 3.0f });
            Vector<float> vectorB = DenseVector.OfArray(new float[] { 4.0f, 5.0f, 6.0f });

            // Compute the dot product of vectorA and vectorB
            float dotProduct = vectorA.DotProduct(vectorB);
            Console.WriteLine("Dot Product of vectorA and vectorB: " + dotProduct);

            // MATRIX OPERATIONS
            // Create a 2x3 matrix (matrixA)
            Matrix<float> matrixA = DenseMatrix.OfArray(new float[,] {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            // Create a 3x2 matrix (matrixB)
            Matrix<float> matrixB = DenseMatrix.OfArray(new float[,] {
                { 7, 8 },
                { 9, 10 },
                { 11, 12 }
            });

            // Multiply matrixA by matrixB resulting in a 2x2 matrix
            Matrix<float> matrixProduct = matrixA * matrixB;
            Console.WriteLine("Result of matrixA * matrixB:");
            Console.WriteLine(matrixProduct.ToString());

            // VECTOR-MATRIX MULTIPLICATION
            // Multiply matrixA (2x3) by a 3-element vector
            Vector<float> vectorC = DenseVector.OfArray(new float[] { 1, 2, 3 });
            Vector<float> resultVector = matrixA * vectorC;
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

            Console.WriteLine($"Vocabulary size: {vocabSize}, Embedding dimension: {embeddingDim}");

            // Create an embedding layer using Single precision.
            EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
            embeddingLayer.SetRandom(-0.5f, 0.5f);

            // Debug print the first 15 tokens and their IDs
            int numToks = 15;
            for (int i = 0; i < numToks; i++)
            {
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            }
            Console.WriteLine();

            // Generate embeddings for every token in the vocabulary
            List<VectorF> embeddings = new List<VectorF>();
            foreach (var token in vocabulary.Keys)
            for (int i = 0; i < vocabSize; i++)
            {
                int tokenId = i; // vocabulary[token];
                VectorF embedding = embeddingLayer.Lookup(tokenId);
                embeddings.Add(embedding);
            }
            Console.WriteLine($"Embeddings for tokens (count: {embeddings.Count}):");

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

                // Perform self-attention on the input matrix
                MatrixF outputMatrix = selfAttention.Forward(inputMatrix);

                // Report the output matrix dimensions
                Console.WriteLine($"Output matrix: {outputMatrix.RowCount} x {outputMatrix.ColumnCount}");
            }

            SelfAttention.Save(selfAttention, "./self-attention.json");
        }

        public static void DemoDenseOutput()
        {
            // Load vocabulary from file.
            Dictionary<string, int> vocabulary = BPETokenizer.LoadVocabularyFromFile("./vocab.json");
            int vocabSize    = vocabulary.Count;
            int embeddingDim = 16; // as chosen for embeddings and self-attention
            int numTokens    = 10;    // number of tokens (rows) to simulate

            EmbeddingLayer embeddingLayer = EmbeddingLayer.Load("./embeddings.json");
            SelfAttention  selfAttention  = SelfAttention.Load("./self-attention.json");

            // Load input string and tokenize.
            string input = File.ReadAllText("SampleStr.txt");
            List<string> tokList     = BPETokenizer.TokenizeUsingVocabulary(input, vocabulary);
            List<int>    tokIdList   = BPETokenizer.TokenIdsUsingVocabulary(input, vocabulary);

            // Debug print the first 15 tokens and their IDs
            for (int i = 0; i < numTokens; i++)
            {
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            }
            Console.WriteLine();

            // Take the first 'numTokens' tokens to process.
            List<int> tokenIdsSub = tokIdList.Take(numTokens).ToList();

            // Look up embeddings and form the input matrix of shape (numTokens, embeddingDim).
            List<VectorF> embList = embeddingLayer.LookupList(tokenIdsSub);
            MatrixF inputMatrix = DenseMatrix.OfRowVectors(embList);

            // Self-attention forward pass.
            // Takes in and outputs a matrix of shape (numTokens, embeddingDim).
            MatrixF selfAttOutput = selfAttention.Forward(inputMatrix);

            // Validate the shape of the output matrix.
            Console.WriteLine($"Self-attention output: {selfAttOutput.RowCount} x {selfAttOutput.ColumnCount}");

            // Create the dense layer to project from embeddingDim to vocabSize.
            DenseLayer denseLayer = new DenseLayer(embeddingDim, vocabSize);

            // Compute logits: each row corresponds to a token, and has vocabSize columns.
            MatrixF logits = denseLayer.Forward(selfAttOutput);

            // Optionally, apply softmax to get probability distributions.
            MatrixF probabilities = DenseLayer.Softmax(logits);

            // Select the last token's probability distribution.
            VectorF lastTokenProbabilities = probabilities.Row(probabilities.RowCount - 1);

            // Compute the argmax of the last token's probabilities.
            int predictedIndex = ArgMax(lastTokenProbabilities);

            // Optionally, convert the predicted index back into a token string.
            // Since your vocabulary is a Dictionary<string, int>, we can search for the matching key.
            string predictedToken = vocabulary.FirstOrDefault(kvp => kvp.Value == predictedIndex).Key;
            if (predictedToken == null)
            {
                predictedToken = "<unk>";
            }

            Console.WriteLine($"Predicted next token (index {predictedIndex}): {predictedToken}");
        }

        // Helper method to compute the index of the maximum element in a vector.
        public static int ArgMax(VectorF v)
        {
            int maxIndex = 0;
            float maxVal = v[0];
            for (int i = 1; i < v.Count; i++)
            {
                if (v[i] > maxVal)
                {
                    maxVal = v[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }


        static void Main(string[] args)
        {
            //DemoBPE();
            //DemoMatrix();
            //DemoEmbeddings();
            //DemoSelfAttention();
            DemoDenseOutput();
        }
    }
}

