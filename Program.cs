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

        public static string CurrModelDirname = "./Model_100K_V3";

        // --------------------------------------------------------------------------------------------
        // MARK: Create
        // --------------------------------------------------------------------------------------------

        public static void DemoModel100K_Create()
        {
            string modeldirname = CurrModelDirname;
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

        // --------------------------------------------------------------------------------------------
        // MARK: Train
        // --------------------------------------------------------------------------------------------

        public static void DemoModel100K_Train()
        {
            string modeldirname = CurrModelDirname;
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

                Console.WriteLine("Most experts would likely agree that while we may see incremental, recursive improvements over coming decades, achieving a model that autonomously self‑improves without any supervision is still a long‑term research goal.");
                TrainingFramework.NextTokens(model2, "Most experts would likely agree that while we may see", 10);

                if (!TrainingFramework.validRun)
                    break;
            }

            //TrainingFramework.TrainModel_Backprop(modeldirname, input);
        }


        // --------------------------------------------------------------------------------------------
        // MARK: Predict
        // --------------------------------------------------------------------------------------------

        public static void DemoModel_Predict(string prompt)
        {
            string modeldirname = CurrModelDirname;

            // Check if model directory exists
            if (!System.IO.Directory.Exists(modeldirname))
            {
                Console.WriteLine($"ERROR: Model directory '{modeldirname}' does not exist.");
                Console.WriteLine("Please create a model first using: dotnet run create");
                return;
            }

            try
            {
                // Load the model
                TransformerModel model = TransformerModel.LoadModel(modeldirname);
                Console.WriteLine($"Model loaded: {model.Report()}");
                Console.WriteLine();

                // Generate predictions
                Console.WriteLine($"Prompt: \"{prompt}\"");
                Console.WriteLine("Generating next 10 tokens...");
                Console.WriteLine();

                TrainingFramework.NextTokens(model, prompt, 10);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to load model or generate predictions: {ex.Message}");
                Console.WriteLine("Please ensure the model exists and is properly trained.");
            }
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
                    case "predict":
                        {
                            if (args.Length < 2)
                            {
                                Console.WriteLine("Please provide a prompt after 'predict'");
                                Console.WriteLine("Example: dotnet run predict \"Most experts would likely agree\"");
                                return;
                            }

                            // Join all arguments after "predict" as the prompt
                            string prompt = string.Join(" ", args.Skip(1));
                            DemoModel_Predict(prompt);
                        }
                        break;
                    // Add more cases as needed
                    default:
                        Console.WriteLine("Unknown command.");
                        break;
                }
            }
            else
            {
                Console.WriteLine("Please provide a command: create, train, or predict");
            }
        }
    }
}