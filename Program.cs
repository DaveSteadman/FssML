
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
// git commit -am "500k progress"
// git add <filename>
// git push

using System;
using System.Text;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;  // Alias for Vector<float>


// Issues:
// - Serialisation / Model recreation
// - Training: Better scoring

namespace MathNetDemo
{
    class Program
    {

        // --------------------------------------------------------------------------------------------
        // MARK: v0.5
        // --------------------------------------------------------------------------------------------

        public static void DemoModel100K_Create()
        {
            string modeldirname = "./Model_100K_V3";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            // Check if model directory exists and prompt user
            if (System.IO.Directory.Exists(modeldirname))
            {
                Console.Write($"WARNING: Directory '{modeldirname}' already exists. Overwrite? (Y/N): ");
                var key = Console.ReadKey();
                Console.WriteLine();
                if (key.KeyChar != 'Y' && key.KeyChar != 'y')
                {
                    Console.WriteLine("Aborted model creation.");
                    return;
                }
            }

            TrainingFramework.CreateInitialModel(
                modeldirname,
                vocabSize: 2500,
                inputSize: 10,
                embeddingSize: 50);
        }

        public static void DemoModel100K_Train()
        {
            string modeldirname = "./Model_100K_V3";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            // TrainingFramework.CreateInitialModel(
            //     modeldirname,
            //     vocabSize: 1000,
            //     embeddingSize: 45,
            //     inputSize: 10);


            // Load the input string from file
            TransformerModel model = TransformerModel.LoadModel(modeldirname);
            List<string> tokList = model.Vocab!.TokenizeToStrings(input);
            List<int> tokIdList = model.Vocab!.TokenizeToIds(input);




            // Debug print the first 15 tokens and their IDs
            for (int i = 0; i < 15; i++)
            {
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            }
            Console.WriteLine();

            // print the predicted 16th token from the bigram model

            Console.WriteLine($"Count: {model.Bigram!.GetNumAssociations()}");


            int startPredId = 15;
            int prevTokId = tokIdList[startPredId];

            for (int predcount = 0; predcount < 10; predcount++)
            {
                int tokPos = startPredId + predcount;
                int nextTokId = model.Bigram!.GetNextTokenIdProbabilistic(prevTokId);
                Console.WriteLine($"Actual: pos {tokPos} = [{tokList[tokPos]}: {tokIdList[tokPos]}] ");
                Console.WriteLine($"Bigram: {prevTokId}->[{nextTokId}: >{model.Vocab!.GetTokenString(nextTokId)}<] ");

                prevTokId = nextTokId;
            }



            // int j = 15;
            // int nextId = model.Bigram!.GetNextTokenId(tokIdList[j-1]);
            // Console.WriteLine($"Actual: [{tokList[j]}: {tokIdList[j]}] ");
            // Console.WriteLine($"Bigram: {tokIdList[j-1]}->[{nextId}: >{model.Vocab!.GetTokenString(nextId)}<] ");

            int numCycles = 100;
            for (int i = 1; i <= numCycles; i++)
            {
                Console.WriteLine($"\n\n---- TRAIN {i}/{numCycles} ----------------\n");

                TrainingFramework.TrainModel(modeldirname, input);

                Console.WriteLine($"\n\n---- RUN {i}/{numCycles} ----------------\n");

                TransformerModel model2 = TransformerModel.LoadModel(modeldirname);
                Console.WriteLine($"\n\n---- RUN {i}/{numCycles} ----------------\n");

                TrainingFramework.NextTokens(model2, "With many tears they singled out the whitened", 10);

                if (!TrainingFramework.validRun)
                    break;
            }

            //TrainingFramework.TrainModel_Backprop(modeldirname, input);
        }

        // --------------------------------------------------------------------------------------------

        public static void DemoModel500K_Create()
        {
            string modeldirname = "./Model_500K_V3";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            TrainingFramework.CreateInitialModel(
                modeldirname,
                vocabSize: 3500,
                inputSize: 20,
                embeddingSize: 60);
        }

        public static void DemoModel500K_Train()
        {
            string modeldirname = "./Model_500K_V3";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            // Load the input string from file
            TransformerModel model = TransformerModel.LoadModel(modeldirname);
            List<string> tokList = model.Vocab!.TokenizeToStrings(input);
            List<int> tokIdList = model.Vocab!.TokenizeToIds(input);

            // Debug print the first 15 tokens and their IDs
            for (int i = 0; i < 15; i++)
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            Console.WriteLine();

            // print the predicted 16th token from the bigram model
            Console.WriteLine($"Count: {model.Bigram!.GetNumAssociations()}");

            int startPredId = 15;
            int prevTokId = tokIdList[startPredId];

            for (int predcount = 0; predcount < 10; predcount++)
            {
                int tokPos = startPredId + predcount;
                int nextTokId = model.Bigram!.GetNextTokenIdProbabilistic(prevTokId);
                Console.WriteLine($"Actual: pos {tokPos} = [{tokList[tokPos]}: {tokIdList[tokPos]}] ");
                Console.WriteLine($"Bigram: {prevTokId}->[{nextTokId}: >{model.Vocab!.GetTokenString(nextTokId)}<] ");

                prevTokId = nextTokId;
            }



            // int j = 15;
            // int nextId = model.Bigram!.GetNextTokenId(tokIdList[j-1]);
            // Console.WriteLine($"Actual: [{tokList[j]}: {tokIdList[j]}] ");
            // Console.WriteLine($"Bigram: {tokIdList[j-1]}->[{nextId}: >{model.Vocab!.GetTokenString(nextId)}<] ");

            int numCycles = 100;
            for (int i = 1; i <= numCycles; i++)
            {
                Console.WriteLine($"\n\n---- TRAIN {i}/{numCycles} ----------------\n");

                TrainingFramework.TrainModel(modeldirname, input);

                Console.WriteLine($"\n\n---- RUN {i}/{numCycles} ----------------\n");

                TransformerModel model2 = TransformerModel.LoadModel(modeldirname);

                Console.WriteLine($"With many tears they singled out the whitened bones of their loved comrade and laid them within a golden urn.");
                TrainingFramework.NextTokens(model2, "you shall now pay me in full", 10);

                if (!TrainingFramework.validRun)
                    break;
            }

            //TrainingFramework.TrainModel_Backprop(modeldirname, input);
        }

        // --------------------------------------------------------------------------------------------
        // MARK: Main
        // --------------------------------------------------------------------------------------------

        static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                switch (args[0].ToLower())
                {
                    case "create":
                        {
                            string modeldirname = "./Model_100K_V3";
                            if (System.IO.Directory.Exists(modeldirname))
                            {
                                Console.Write($"WARNING: Directory '{modeldirname}' already exists. Overwrite? (Y/N): ");
                                var key = Console.ReadKey();
                                Console.WriteLine();
                                if (key.KeyChar != 'Y' && key.KeyChar != 'y')
                                {
                                    Console.WriteLine("Aborted model creation.");
                                    return;
                                }
                            }
                            DemoModel100K_Create();
                        }
                        break;
                    case "train":
                        DemoModel100K_Train();
                        break;
                    // Add more cases as needed
                    default:
                        Console.WriteLine("Unknown command.");
                        break;
                }
            }
            else
            {
                Console.WriteLine("Please provide a command: create or train");
            }
        }
    }
}