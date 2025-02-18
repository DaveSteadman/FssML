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
        model.Create02_CreateEmbedding(32);
        model.Create03_CreatePositionalEncoding(10);
        model.Create04_CreateSelfAttention();
        model.Create05_CreateFeedForward();
        model.Create06_CreateOutputProjection();
        model.SaveModel();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: TRAIN01
    // --------------------------------------------------------------------------------------------

    // TrainingFramework.TrainModel
    public static void TrainModel(string modeldirname, string trainingdata)
    {
        // Start the timer
        RunTimer timer = new RunTimer();

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


        int numPasses = 500;
        float noiseVal = 1f;

        List<float> last10Scores = new List<float>();

        for (int i = 0; i < numPasses; i++)
        {
            // Create a deep copy of the model
            TransformerModel modelMutation = model.DeepCopy();
            modelMutation.AddNoise(noiseVal, i);

            //Console.WriteLine($"Checksums: Original {model.CheckSum()} // Mutation {modelMutation.CheckSum()}");

            // Run the training again, looking for a better (higher) score
            float newPredictionScore = 0f;
            foreach (TrainingInput input in trainData)
            {
                newPredictionScore += modelMutation.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
            }

            // If the new model is better, save it
            if (newPredictionScore > baselinePredictionScore)
            {
                // noiseVal = (newPredictionScore - baselinePredictionScore) / 10f;

                // Save the new model
                model = modelMutation;
                baselinePredictionScore = newPredictionScore;


                model.DirPath = "./Model_005";
                model.SaveModel();

            }
            // else
            // {
            //     if (noiseVal > 0.00001f)
            //         noiseVal *= 0.95f;
            // }

            Console.WriteLine($"--- Training Pass {i} // {baselinePredictionScore} // {newPredictionScore} // {noiseVal:F3} --- ");

            // Break if there has been a console keypress
            if (Console.KeyAvailable)
                break;

            // Update the last 10 scores
            last10Scores.Add(newPredictionScore);
            if (last10Scores.Count > 10)
                last10Scores.RemoveAt(0);

            // If the average score, plus the min max range, is less than the baselinePredictionScore, increase the noise
            float avgScore = last10Scores.Average();
            float minScore = last10Scores.Min();
            float maxScore = last10Scores.Max();
            float scoreRange = maxScore - minScore;
            if (minScore + scoreRange < baselinePredictionScore)
            {
                noiseVal *= 1.1f;
            }
            else
            {
                noiseVal *= 0.9f;
            }
            noiseVal = 1f;
        }

        // Output the elapsed time
        Console.WriteLine($"Elapsed time: {timer.ElapsedSeconds:F3} seconds");
    }

    // --------------------------------------------------------------------------------------------
    // MARK: TRAIN02 - Backprop
    // --------------------------------------------------------------------------------------------

        public static void TrainModel_Backprop(string modeldirname, string trainingdata)
    {
        // Start the timer
        RunTimer timer = new RunTimer();

        // Load the model
        TransformerModel model = TransformerModel.LoadModel(modeldirname);
        Console.WriteLine($"Model Report: {model.Report()}");

        // Create the training data
        List<TrainingInput> trainData = ConstructTrainingPass(model, trainingdata, 100);

        // perform the forward pass for one line, to produce a single output.
        var embeddings        = model.Embedding!.LookupListToMatrix(trainData[0].InputTokenIdList);
        var encodedEmbeddings = model.PositionalEnc!.ApplyPositionalEncoding(embeddings);
        var selfAttOutput     = model.SelfAtt!.Forward(encodedEmbeddings);

        var logits = model.OutputProjection!.RawOutputs(selfAttOutput);
        var hotOne = model.OutputProjection!.HotOne(trainData[0].ExpectedOutputTokenId);

        var outputLayerNudges = model.OutputProjection!.ComputeOutputNudge(logits, hotOne);


        // // Get the training score for this model
        // float baselinePredictionScore = 0f;
        // foreach (TrainingInput input in trainData)
        // {
        //     baselinePredictionScore += model.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
        // }

        // // Output the baseline loss
        // Console.WriteLine($"\nBaseline score: {baselinePredictionScore}\n");


        // int numPasses = 500;
        // float noiseVal = 1f;

        // List<float> last10Scores = new List<float>();

        // for (int i = 0; i < numPasses; i++)
        // {
        //     // Create a deep copy of the model
        //     TransformerModel modelMutation = model.DeepCopy();
        //     modelMutation.AddNoise(noiseVal, i);

        //     //Console.WriteLine($"Checksums: Original {model.CheckSum()} // Mutation {modelMutation.CheckSum()}");

        //     // Run the training again, looking for a better (higher) score
        //     float newPredictionScore = 0f;
        //     foreach (TrainingInput input in trainData)
        //     {
        //         newPredictionScore += modelMutation.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
        //     }

        //     // If the new model is better, save it
        //     if (newPredictionScore > baselinePredictionScore)
        //     {
        //         // noiseVal = (newPredictionScore - baselinePredictionScore) / 10f;

        //         // Save the new model
        //         model = modelMutation;
        //         baselinePredictionScore = newPredictionScore;


        //         model.DirPath = "./Model_005";
        //         model.SaveModel();

        //     }
        //     // else
        //     // {
        //     //     if (noiseVal > 0.00001f)
        //     //         noiseVal *= 0.95f;
        //     // }

        //     Console.WriteLine($"--- Training Pass {i} // {baselinePredictionScore} // {newPredictionScore} // {noiseVal:F3} --- ");

        //     // Break if there has been a console keypress
        //     if (Console.KeyAvailable)
        //         break;

        //     // Update the last 10 scores
        //     last10Scores.Add(newPredictionScore);
        //     if (last10Scores.Count > 10)
        //         last10Scores.RemoveAt(0);

        //     // If the average score, plus the min max range, is less than the baselinePredictionScore, increase the noise
        //     float avgScore = last10Scores.Average();
        //     float minScore = last10Scores.Min();
        //     float maxScore = last10Scores.Max();
        //     float scoreRange = maxScore - minScore;
        //     if (minScore + scoreRange < baselinePredictionScore)
        //     {
        //         noiseVal *= 1.1f;
        //     }
        //     else
        //     {
        //         noiseVal *= 0.9f;
        //     }
        //     noiseVal = 1f;
        // }

        // Output the elapsed time
        Console.WriteLine($"Elapsed time: {timer.ElapsedSeconds:F3} seconds");
    }


    // --------------------------------------------------------------------------------------------
    // MARK: Train Data
    // --------------------------------------------------------------------------------------------

    private static List<TrainingInput> ConstructTrainingPass(TransformerModel model, string trainingdata, int numEntries = 50)
    {
        List<TrainingInput> trainingPass = new List<TrainingInput>();

        // Convert input into tokens
        List<int> tokenIds = model.Vocab!.TokenizeToIds(trainingdata);
        int padTokId = model.Vocab!.GetTokenId("<PAD>");

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

            // if (trainingPass.Count >= numEntries)
            //     break;
        }

        Random rnd = new Random();
        List<TrainingInput> randomTrainingSample = trainingPass.OrderBy(x => rnd.Next()).Take(numEntries).ToList();

        // List the first ten training inputs
        for (int i = 0; i < 10; i++)
        {
            Console.WriteLine($"Training Input {i}: {model.Vocab.DebugTokenList(randomTrainingSample[i].InputTokenIdList)} -> {model.Vocab.GetTokenString(randomTrainingSample[i].ExpectedOutputTokenId)}");
        }

        //return trainingPass;
        return randomTrainingSample;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    public static void NextTokens(TransformerModel model, string promptstr, int numTokens)
    {
        // tokenise the text
        List<int> tokenIds    = model.Vocab!.TokenizeToIds(promptstr);
        List<int> finalTokens = new List<int>(tokenIds);

        for (int i = 0; i < numTokens; i++)
        {
            // Get the last <input length> of tokens
            List<int> inputTokenIds       = finalTokens.TakeLast(model.ModelDetails.InputLen).ToList();

            // Get the last <input length> of tokens
            //List<int> inputTokenIds       = finalTokens.GetRange(finalTokens.Count - model.ModelDetails.InputLen, model.ModelDetails.InputLen);
            List<int> paddedInputTokenIds = new List<int>(inputTokenIds);

            // Add the <PAD> token if the input is too short
            while (paddedInputTokenIds.Count < model.ModelDetails.InputLen)
                paddedInputTokenIds.Add(model.Vocab.GetTokenId("<PAD>"));

            int nextTokId = model.PredictNextToken(paddedInputTokenIds);
            finalTokens.Add(nextTokId);
        }

        // Loop and output all the tokens
        Console.WriteLine($"Prompt: {promptstr}");
        Console.WriteLine($"Tokens: {model.Vocab.DebugTokenList(finalTokens)}");
    }
}
