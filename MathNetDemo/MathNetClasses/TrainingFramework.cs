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
        model.Create06_CreateOutputProjection();
        model.Create07_SetupNoise(noiseVal, percentToChange);
        model.SaveModel();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: TRAIN01
    // --------------------------------------------------------------------------------------------

    // TrainingFramework.TrainModel
    public static async Task TrainModel(string modeldirname, string trainingdata)
    {
        // boilerplate await / yield call for 100ms
        //await System.Threading.Tasks.Task.Delay(100);

        // Start the timer, reset the flag
        RunTimer timer = new RunTimer();
        validRun = true;

        // Load the model
        TransformerModel model = TransformerModel.LoadModel(modeldirname);
        Console.WriteLine($"Model Report: {model.Report()}");

        // Crte the training data
        List<TrainingInput> trainData = ConstructTrainingPass(model, trainingdata, 50);

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
        int numThreads = 40;
        int numPasses  = 30;
        var tasks = new List<Task<(TransformerModel newModel, float newScore)>>();

        for (int i = 0; i < numThreads; i++)
        {
            int threadID = i;
            tasks.Add(TrainModelThreadAsync(model, trainData, baselinePredictionScore, numPasses, threadID));

        }

        // Block until all threads are done
        Task.WaitAll(tasks.ToArray());

        // Find the best result
        var bestResult = tasks.Select(t => t.Result).OrderByDescending(r => r.newScore).First();
        var (newmodel, newscore) = bestResult;
        Console.WriteLine($"New Model Score: {newscore}");

        // Save the model if its better
        if (newscore > baselinePredictionScore)
        {
            model = newmodel;
            model.DirPath = modeldirname;
            model.SaveModel();
        }

        Console.WriteLine($"\n\nApplying winning noise: {NoiseQueue.Count} items\n\n");

        // Apply all successful noise to the model
        float retScore = baselinePredictionScore;
        while (TryDequeue(out ModelNoise noise))
        {
            if (!validRun) break;
            if (Console.KeyAvailable) { Console.WriteLine("Keystroke detected: ValidRun set false"); validRun = false; }

            TransformerModel modelMutation = model.DeepCopy();
            modelMutation.ApplyNoise(noise);

            float newPredictionScore = 0f;
            foreach (TrainingInput input in trainData)
            {
                newPredictionScore += modelMutation.PredictionScore(input.InputTokenIdList, input.ExpectedOutputTokenId);
            }

            // If the new model is better, save it
            if (newPredictionScore > retScore)
            {
                // Save the new model
                model = modelMutation;
                retScore = newPredictionScore;

                Console.WriteLine($"Postthreads noise: // Score {newPredictionScore}");

                model = modelMutation;
                model.DirPath = modeldirname;
                model.SaveModel();
            }
        }
        // Clear out any remaining elements in the queue
        while (TryDequeue(out ModelNoise noise)) { }

        float finalScore = retScore;

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
        int threadID)
    {
        // Initialise the return values
        TransformerModel retmodel = model.DeepCopy();
        float retScore            = baselinePredictionScore;

        if (!validRun) return (retmodel, retScore);

        Console.WriteLine($"Thread {threadID} // Starting");

        for (int i = 0; i < numPasses; i++)
        {
            // Pause the thread, being a good citizen with lots of tasks around.
            await Task.Yield();

            if (!validRun) break;
            if (Console.KeyAvailable) { Console.WriteLine("Keystroke detected: ValidRun set false"); validRun = false; }

            // Create a deep copy of the model
            TransformerModel modelMutation = retmodel.DeepCopy();

            // Add the noise
            float noiseVal = 0.21f;
            float percentToChange = 1.0f;
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
                // Save the new model
                retmodel = modelMutation;
                retScore = newPredictionScore;

                Console.WriteLine($"Thread {threadID} // Pass {i}/{numPasses} // Score {newPredictionScore}");

                NoiseQueue.Enqueue(newNoise);
            }
        }

        Console.WriteLine($"Thread {threadID} // Finished");

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
