
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


        public static void DemoModel100K()
        {
            string modeldirname = "./Model_100K-V2";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            TrainingFramework.CreateInitialModel(
                modeldirname,
                vocabSize: 1000,
                embeddingSize: 45,
                inputSize: 10);


            // Load the input string from file
            TransformerModel model = TransformerModel.LoadModel(modeldirname);
            List<string> tokList   = model.Vocab!.TokenizeToStrings(input);
            List<int>    tokIdList = model.Vocab!.TokenizeToIds(input);




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

            for (int predcount=0; predcount<10; predcount++)
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

                TransformerModel model2  = TransformerModel.LoadModel(modeldirname);
                TrainingFramework.NextTokens(model2, "you shall now pay me in full", 10);

                if (!TrainingFramework.validRun)
                    break;
            }

            //TrainingFramework.TrainModel_Backprop(modeldirname, input);
        }

        // --------------------------------------------------------------------------------------------

        public static async Task DemoModel500K()
        {
            string modeldirname = "./Model_500K";
            string textFilepath = "SampleStr.txt";
            string input = File.ReadAllText(textFilepath);

            // TrainingFramework.CreateInitialModel(
            //     modeldirname,
            //     vocabSize: 2500,
            //     inputSize: 20,
            //     embeddingSize: 50);

            // Load the input string from file
            TransformerModel model = TransformerModel.LoadModel(modeldirname);
            List<string> tokList   = model.Vocab!.TokenizeToStrings(input);
            List<int>    tokIdList = model.Vocab!.TokenizeToIds(input);

            // Debug print the first 15 tokens and their IDs
            for (int i = 0; i < 15; i++)
                Console.Write($"[{tokList[i]}: {tokIdList[i]}] ");
            Console.WriteLine();

            // print the predicted 16th token from the bigram model
            Console.WriteLine($"Count: {model.Bigram!.GetNumAssociations()}");

            int startPredId = 15;
            int prevTokId = tokIdList[startPredId];

            for (int predcount=0; predcount<10; predcount++)
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

                TransformerModel model2  = TransformerModel.LoadModel(modeldirname);
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
            //DemoBPE();
            //DemoMatrix();
            //DemoEmbeddings();
            //DemoSelfAttention();
            //DemoDenseOutput();

            //DemoTokenVocab();
            //DemoEmbeddings2();
            //DemoPositionalEncoding();

            //DemoMakeModel();

            //DemoFirstModelRun();

            DemoModel100K();
            //DemoModel500K();
            //DemoTinyML();
        }
    }
}

