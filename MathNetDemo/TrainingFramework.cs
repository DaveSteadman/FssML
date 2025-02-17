using System;
using System.Collections.Generic;


// TrainingFramework:
// - Creates/Loads a model, performs training passes and serializes the model back to disk
// - Outputs to a loss log file so we can plot progress


// Class for a single training input/output
public class TrainingInput
{
    public List<int> InputTokenIdList = new List<int>();
    public int       ExpectedOutputTokenId;
}

public static class TrainingFramework
{
    public static void CreateInitialModel(string dirname)
    {
        var model = new TransformerModel(dirname);

        model.Create01_CreateVocab("./SampleStr.txt", 1000);
        model.Create02_CreateEmbedding(16);
        model.Create03_CreatePositionalEncoding(10);
        model.Create04_CreateSelfAttention();
        model.Create05_CreateFeedForward();
        model.Create06_CreateOutputProjection();
        model.SaveModel();
    }

    // TrainingFramework.TrainModel
    public static void TrainModel(string modeldirname, string trainingdata)
    {
        // Load the model
        TransformerModel model = TransformerModel.LoadModel(modeldirname);
        Console.WriteLine($"Model Report: {model.Report()}");

        // Crte the training data
        List<TrainingInput> trainData = ConstructTrainingPass(model, trainingdata);

        // Get the training score for this model
        float baselinePredictionScore = 0f;
        foreach (TrainingInput input in trainData)
        {
            baselinePredictionScore += model.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
        }

        // Output the baseline loss
        Console.WriteLine($"\nBaseline score: {baselinePredictionScore}\n");


        int numPasses = 1;

        for (int i = 0; i < numPasses; i++)
        {
            Console.WriteLine($"--- Training Pass {i} // {baselinePredictionScore} --- ");

            // Create a deep copy of the model
            TransformerModel modelMutation = model.DeepCopy();
            modelMutation.AddNoise(0.1f);

            Console.WriteLine($"Checksums: Original {model.CheckSum()} // Mutation {modelMutation.CheckSum()}");

            // Run the training again, looking for a better (higher) score
            float newPredictionScore = 0f;
            foreach (TrainingInput input in trainData)
            {
                newPredictionScore += modelMutation.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
            }


            // If the new model is better, save it
            if (newPredictionScore > baselinePredictionScore)
            {
                //model = modelMutation;
                //baselinePredictionScore = newPredictionScore;

                modelMutation.DirPath = "./Model_003";

                Console.WriteLine($"IMPROVEMENT: Saving new model: Score {newPredictionScore}");
                modelMutation.SaveModel();
            }
            else
            {
                Console.WriteLine("New score: " + newPredictionScore);
            }
        }
    }

    private static List<TrainingInput> ConstructTrainingPass(TransformerModel model, string trainingdata)
    {
        List<TrainingInput> trainingPass = new List<TrainingInput>();

        // Convert input into tokens
        List<int> tokenIds = model.Vocab!.TokenizeToIds(trainingdata);
        int padTokId = model.Vocab!.GetTokenId("<PAD>");
        int maxTrainingEntries = 50;

        // Loop through the input list of tokens, setting up the training data of "windows" of
        // whole sets of tokens
        int windowSize = model.ModelDetails.InputLen;

        for (int i = 0; i < tokenIds.Count - windowSize; i++)
        {
            // Create a training input
            TrainingInput trainingInput = new TrainingInput();
            trainingInput.InputTokenIdList.AddRange(tokenIds.GetRange(i, windowSize));
            trainingInput.ExpectedOutputTokenId = tokenIds[i + windowSize];

            trainingPass.Add(trainingInput);

            if (trainingPass.Count >= maxTrainingEntries)
                break;
        }

        // List the first ten training inputs
        for (int i = 0; i < 10; i++)
        {
            Console.WriteLine($"Training Input {i}: {model.Vocab.DebugTokenList(trainingPass[i].InputTokenIdList)} -> {model.Vocab.GetTokenString(trainingPass[i].ExpectedOutputTokenId)}");
        }

        return trainingPass;
    }

}