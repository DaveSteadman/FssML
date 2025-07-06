using System;
using System.Collections.Generic;
using System.Collections.Concurrent;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;
using System.Runtime.CompilerServices;  // Alias for Vector<float>


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
    public static bool validRun = false;

    // --------------------------------------------------------------------------------------------
    // MARK: Static thread-safe queue
    // --------------------------------------------------------------------------------------------

    private static readonly ConcurrentQueue<ModelNoise> NoiseQueue = new ConcurrentQueue<ModelNoise>();

    // Append a value from any thread
    public static void Enqueue(ModelNoise value)
    {
        NoiseQueue.Enqueue(value);
    }

    // Remove a value in a thread-safe manner
    public static bool TryDequeue(out ModelNoise value)
    {
        return NoiseQueue.TryDequeue(out value);
    }

    // Check if the queue has any content
    public static bool HasContent()
    {
        return !NoiseQueue.IsEmpty;
    }

    // --------------------------------------------------------------------------------------------

    public static void CreateInitialModel(
        string dirname,
        int vocabSize = 2500,
        int inputSize = 10,
        int embeddingSize = 100,
        float noiseVal = 0.1f,
        float percentToChange = 1f)
    {
        var model = new TransformerModel(dirname);

        model.Create01_CreateVocab("./SampleStr.txt", vocabSize);
        model.Create01_CreateBigram("./SampleStr.txt");
        model.Create02_CreateEmbedding(embeddingSize);
        model.Create03_CreatePositionalEncoding(inputSize);
        model.Create04_CreateSelfAttention();
        model.Create05_CreateFeedForward();
        model.Create04b_CreateSelfAttention2();
        model.Create05b_CreateFeedForward2();
        model.Create06_CreateOutputProjection();
        model.Create07_SetupNoise(noiseVal, percentToChange);
        model.SaveModel();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: TRAIN01
    // --------------------------------------------------------------------------------------------

    // TrainingFramework.TrainModel
    public static void TrainModel(string modeldirname, string trainingdata)
    {
        // Start the timer, reset the flag
        RunTimer timer = new RunTimer();
        validRun = true;

        // Load the model
        TransformerModel model = TransformerModel.LoadModel(modeldirname);
        Console.WriteLine($"Model Report: {model.Report()}");

        // Crte the training data
        List<TrainingInput> trainData = ConstructTrainingPass(model, trainingdata, 50);
        //List<TrainingInput> trainData = ConstructFullTrainingPass(model, trainingdata);

        // Get the training score for this model
        float baselinePredictionScore = 0f;
        foreach (TrainingInput input in trainData)
        {
            baselinePredictionScore += model.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
        }

        // Output the baseline loss
        Console.WriteLine($"\nBaseline score: {baselinePredictionScore}\n");

        float initialScore = baselinePredictionScore;


        // Run multiple instances of TrainModelThread in parallel
        int numThreads = 10;
        int numPasses = 2;

        float noiseVal = model.ModelDetails.NoiseVal; // e.g. 0.12 means +/-12% of the noise value
        float percentToChange = model.ModelDetails.PercentChange; // e.g. 1.0 means +/-100% of the percent change

        // float noiseVal = 0.12f;
        // float percentToChange = 1.0f;

        // Use a fraction of the value for delta
        float noiseFrac = 0.5f; // e.g. 0.5 means +/-50% of noiseVal
        float percentFrac = 0.5f; // e.g. 0.5 means +/-50% of percentToChange

        // Create lists of the noise and percent values
        List<float> noiseValues = new List<float>();
        List<float> percentValues = new List<float>();

        for (int i = 0; i < numThreads; i++)
        {
            float noiseDelta = noiseVal * noiseFrac;
            float percentDelta = percentToChange * percentFrac;

            float noiseStep = (noiseDelta * 2) / numThreads;
            float noiseThreadVal = (noiseVal - noiseDelta) + (i * noiseStep);

            float percentStep = (percentDelta * 2) / numThreads;
            float percentThreadVal = (percentToChange + percentDelta) - (i * percentStep); // Apply the largest change, to the smallest percentage

            noiseValues.Add(noiseThreadVal);
            percentValues.Add(percentThreadVal);
        }


        var tasks = new List<Task<(TransformerModel newModel, float newScore)>>();


        for (int i = 0; i < numThreads; i++)
        {
            int threadID = i;
            tasks.Add(TrainModelThreadAsync(model, trainData, baselinePredictionScore, numPasses, threadID, noiseValues[i], percentValues[i]));

        }

        // Block until all threads are done
        Task.WaitAll(tasks.ToArray());

        // Find the best result
        var bestResult = tasks.Select(t => t.Result).OrderByDescending(r => r.newScore).First();
        var (newmodel, newscore) = bestResult;
        Console.WriteLine($"New Model Score: {newscore}");

        // Write the new score to the logfile
        newmodel.AppendLog(newmodel.ModelDetails.NumIterations, newscore);

        // Save the model if its better
        if (newscore > baselinePredictionScore)
        {
            model = newmodel;
            model.DirPath = modeldirname;
            model.SaveModel();
        }

        Console.WriteLine($"\n\nApplying winning noise: {NoiseQueue.Count} items\n\n");

        // Apply all successful noise directly to the model without re-scoring
        float retScore = baselinePredictionScore;

        float totalSuccessNoise = 0f;
        float totalSuccessPercent = 0f;
        while (TryDequeue(out ModelNoise noise))
        {
            if (!validRun) break;
            if (Console.KeyAvailable) { Console.WriteLine("Keystroke detected: ValidRun set false"); validRun = false; }

            model.ModelDetails.NumIterations++;
            model.ApplyNoise(noise);
            model.DirPath = modeldirname;
            model.SaveModel();

            totalSuccessNoise = max(totalSuccessNoise, noise.recordedNoise);
            totalSuccessPercent = max(totalSuccessPercent, noise.recordedPercentChange);
        }

        // // Calculate the average noise and percent change (only if there were successes)
        // float avgSuccessNoise = successCount > 0 ? totalSuccessNoise / successCount : 0f;
        // float avgSuccessPercent = successCount > 0 ? totalSuccessPercent / successCount : 0f;

        // Blend the model's noise and percent change values towards the best found values - takes out some of the volatility
        model.ModelDetails.NoiseVal = halfwayto(model.ModelDetails.NoiseVal, totalSuccessNoise);
        model.ModelDetails.PercentChange = halfwayto(model.ModelDetails.PercentChange, totalSuccessPercent);
        model.SaveModel();


        // Clear out any remaining elements in the queue
        while (TryDequeue(out ModelNoise noise)) { }

        float finalScore = retScore;

        // Write the new score to the logfile
        newmodel.AppendLog(newmodel.ModelDetails.NumIterations, finalScore);

        // Output the elapsed time and score
        Console.WriteLine($"Elapsed time: {timer.ElapsedSeconds:F3} seconds // score {initialScore:F3} -> {finalScore:F3} = {finalScore - initialScore:F3}");
    }

    // --------------------------------------------------------------------------------------------
    // MARK: TRAIN01 - Thread
    // --------------------------------------------------------------------------------------------

    // Create a background thread that takes in a model, a training set and a score, and returns a new model with its score

    // Usage:
    // var (newmodel, newscore) = TrainModelThreadAsync(model, trainData, baselinePredictionScore, 100);

    public static async Task<(TransformerModel newmodel, float newScore)> TrainModelThreadAsync(
        TransformerModel model,
        List<TrainingInput> trainData,
        float baselinePredictionScore,
        int numPasses,
        int threadID,
        float noiseVal,
        float percentToChange)
    {
        // Initialise the return values
        TransformerModel retmodel = model.DeepCopy();
        float retScore = baselinePredictionScore;

        if (!validRun) return (retmodel, retScore);

        Console.WriteLine($"Thread {threadID,2} // Starting // Noise: {noiseVal:F4} // Percent: {percentToChange:F4}");

        for (int i = 0; i < numPasses; i++)
        {
            // Pause the thread, being a good citizen with lots of tasks around.
            await Task.Yield();

            if (!validRun) break;
            if (Console.KeyAvailable) { Console.WriteLine("Keystroke detected: ValidRun set false"); validRun = false; }

            // Create a deep copy of the model
            TransformerModel modelMutation = retmodel.DeepCopy();
            retmodel.ModelDetails.NumIterations++;

            // Add the noise
            // float noiseVal = 0.12f;
            // float percentToChange = 1.0f;
            ModelNoise newNoise = modelMutation.CreateLimitedNoise(noiseVal, percentToChange);
            modelMutation.ApplyNoise(newNoise);


            // Run the prediction, looking for a better (higher) score
            float newPredictionScore = 0f;
            foreach (TrainingInput input in trainData)
            {
                newPredictionScore += modelMutation.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
            }

            // If the new model is better, save it
            if (newPredictionScore > retScore)
            {
                // Note the improvement
                float improvement = newPredictionScore - retScore;

                // Save the new model
                retmodel = modelMutation;
                retScore = newPredictionScore;

                // Pad threadID to width 3, right-aligned, space-padded
                Console.WriteLine($"Thread {threadID,2} // Pass {i,2}/{numPasses} // Score {newPredictionScore} ({improvement:F1})");

                NoiseQueue.Enqueue(newNoise);
            }
        }

        Console.WriteLine($"Thread {threadID,2} // Finished");

        return (retmodel, retScore);
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

        Console.WriteLine($"Creating Training Data: {tokenIds.Count} tokens, window size {windowSize}, numEntries {numEntries}");

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

        // Select a random set of data
        Random rnd = new Random();
        //List<TrainingInput> trainingSampleList = trainingPass.OrderBy(x => rnd.Next()).Take(numEntries).ToList();

        // select the first X number of list entries
        List<TrainingInput> trainingSampleList = trainingPass.Take(numEntries).ToList();


        // List the first ten training inputs
        for (int i = 0; i < 10; i++)
        {
            Console.WriteLine($"Training Input {i}: {model.Vocab.DebugTokenList(trainingSampleList[i].InputTokenIdList)} -> {model.Vocab.GetTokenString(trainingSampleList[i].ExpectedOutputTokenId)}");
        }

        //return trainingPass;
        return trainingSampleList;
    }

    // ConstructFullTrainingPass - Uses the entire dataset for training
    private static List<TrainingInput> ConstructFullTrainingPass(TransformerModel model, string trainingdata)
    {
        List<TrainingInput> trainingPass = new List<TrainingInput>();

        // Convert input into tokens
        List<int> tokenIds = model.Vocab!.TokenizeToIds(trainingdata);
        int padTokId = model.Vocab!.GetTokenId("<PAD>");

        // Loop through the input list of tokens, setting up the training data of "windows" of
        // whole sets of tokens
        int windowSize = model.ModelDetails.InputLen;

        Console.WriteLine($"Creating FULL Training Data: {tokenIds.Count} tokens, window size {windowSize}");
        Console.WriteLine($"This will create approximately {tokenIds.Count - windowSize} training examples");

        for (int i = 0; i < tokenIds.Count - windowSize; i++)
        {
            // Create a training input
            TrainingInput trainingInput = new TrainingInput();
            trainingInput.InputTokenIdList.AddRange(tokenIds.GetRange(i, windowSize));
            trainingInput.ExpectedOutputTokenId = tokenIds[i + windowSize];

            trainingPass.Add(trainingInput);

            // Progress indicator for large datasets
            if (trainingPass.Count % 50000 == 0)
            {
                Console.WriteLine($"Generated {trainingPass.Count} training examples...");
            }
        }

        Console.WriteLine($"Full training data created: {trainingPass.Count} examples");

        // List the first ten training inputs for verification
        for (int i = 0; i < Math.Min(10, trainingPass.Count); i++)
        {
            Console.WriteLine($"Training Input {i}: {model.Vocab.DebugTokenList(trainingPass[i].InputTokenIdList)} -> {model.Vocab.GetTokenString(trainingPass[i].ExpectedOutputTokenId)}");
        }

        return trainingPass;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    public static void NextTokens(TransformerModel model, string promptstr, int numTokens)
    {
        // tokenise the text
        List<int> tokenIds = model.Vocab!.TokenizeToIds(promptstr);
        List<int> finalTokens = new List<int>(tokenIds);

        for (int i = 0; i < numTokens; i++)
        {
            // Get the last <input length> of tokens
            List<int> inputTokenIds = finalTokens.TakeLast(model.ModelDetails.InputLen).ToList();

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


    // --------------------------------------------------------------------------------------------

    private static float max(float val1, float val2)
    {
        return val1 > val2 ? val1 : val2;
    }

    private static float halfwayto(float originalval, float newval)
    {
        // Calculate the halfway point between the original value and the new value
        return originalval + (newval - originalval) / 2f;
    }

}
