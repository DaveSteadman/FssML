




using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using MatrixF = MathNet.Numerics.LinearAlgebra.Matrix<float>;  // Alias for Matrix<float>
using VectorF = MathNet.Numerics.LinearAlgebra.Vector<float>;  // Alias for Vector<float>


public struct TransformerModelFilenames
{
    public string ModelPath            { get; set; }
    public string VocabPath            { get; set; }
    public string BigramPath           { get; set; }
    public string EmbeddingPath        { get; set; }
    public string SelfAttPath          { get; set; }
    public string FeedForwardPath      { get; set; }
    public string SelfAtt2Path         { get; set; }
    public string FeedForward2Path     { get; set; }
    public string OutputProjectionPath { get; set; }
    public string TrainingLogPath      { get; set; }

    public string BinModelPath            { get; set; }
    public string BinVocabPath            { get; set; }
    public string BinEmbeddingPath        { get; set; }
    public string BinSelfAttPath          { get; set; }
    public string BinFeedForwardPath      { get; set; }
    public string BinSelfAtt2Path         { get; set; }
    public string BinFeedForward2Path     { get; set; }
    public string BinOutputProjectionPath { get; set; }

    public TransformerModelFilenames(string dirPath)
    {
        ModelPath               = System.IO.Path.Combine(dirPath, "model.txt");
        VocabPath               = System.IO.Path.Combine(dirPath, "vocab.txt");
        BigramPath              = System.IO.Path.Combine(dirPath, "bigram.txt");
        EmbeddingPath           = System.IO.Path.Combine(dirPath, "embedding.txt");
        SelfAttPath             = System.IO.Path.Combine(dirPath, "selfatt.txt");
        FeedForwardPath         = System.IO.Path.Combine(dirPath, "feedforward.txt");
        SelfAtt2Path            = System.IO.Path.Combine(dirPath, "selfatt2.txt");
        FeedForward2Path        = System.IO.Path.Combine(dirPath, "feedforward2.txt");
        OutputProjectionPath    = System.IO.Path.Combine(dirPath, "outputprojection.txt");
        TrainingLogPath         = System.IO.Path.Combine(dirPath, "traininglog.txt");

        BinModelPath            = System.IO.Path.Combine(dirPath, "model.bin");
        BinVocabPath            = System.IO.Path.Combine(dirPath, "vocab.bin");
        BinEmbeddingPath        = System.IO.Path.Combine(dirPath, "embedding.bin");
        BinSelfAttPath          = System.IO.Path.Combine(dirPath, "selfatt.bin");
        BinFeedForwardPath      = System.IO.Path.Combine(dirPath, "feedforward.bin");
        BinSelfAtt2Path         = System.IO.Path.Combine(dirPath, "selfatt2.bin");
        BinFeedForward2Path     = System.IO.Path.Combine(dirPath, "feedforward2.bin");
        BinOutputProjectionPath = System.IO.Path.Combine(dirPath, "outputprojection.bin");
    }
}

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

public struct TransformerModelDetails
{
    public int   VocabSize;
    public int   EmbeddingDim;
    public int   FFHiddenDim;
    public int   InputLen;
    public float NoiseVal;
    public float PercentChange;
    public int   NumIterations;

    public TransformerModelDetails()
    {
        VocabSize = 0;
        EmbeddingDim = 0;
        FFHiddenDim = 0;
        InputLen = 0;
        NoiseVal = 0f;
        PercentChange = 0f;
        NumIterations = 0;
    }

    public TransformerModelDetails(int vocabSize, int embeddingDim, int ffHiddenDim, int inputLen, float noiseVal, float percentChange)
    {
        VocabSize     = vocabSize;
        EmbeddingDim  = embeddingDim;
        FFHiddenDim   = ffHiddenDim;
        InputLen      = inputLen;
        NoiseVal      = noiseVal;
        PercentChange = percentChange;
        NumIterations = 0;
    }

    // LoadSave
    public void SaveToFile(string filepath)
    {
        using (var writer = new StreamWriter(filepath, false, Encoding.UTF8))
        {
            writer.WriteLine(VocabSize);
            writer.WriteLine(EmbeddingDim);
            writer.WriteLine(FFHiddenDim);
            writer.WriteLine(InputLen);
            writer.WriteLine(NoiseVal.ToString("F4"));
            writer.WriteLine(PercentChange.ToString("F4"));
            writer.WriteLine(NumIterations);
        }
    }

    public static TransformerModelDetails LoadFromFile(string filepath)
    {
        TransformerModelDetails newDetails = new();

        using (var reader = new StreamReader(filepath, Encoding.UTF8))
        {
            newDetails.VocabSize = int.Parse(reader.ReadLine());
            newDetails.EmbeddingDim = int.Parse(reader.ReadLine());
            newDetails.FFHiddenDim = int.Parse(reader.ReadLine());
            newDetails.InputLen = int.Parse(reader.ReadLine());
            newDetails.NoiseVal = float.Parse(reader.ReadLine());
            newDetails.PercentChange = float.Parse(reader.ReadLine());
            newDetails.NumIterations = int.Parse(reader.ReadLine());

        }
        return newDetails;
    }
}

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

public class ModelNoise
{
    public float recordedNoise { set; get; } = 0; // 0 = no noise, 1 = noise applied
    public float recordedPercentChange { set; get; } = 0; // 0 = no change, 1 = all values changed

    public MatrixF embeddingNoise { set; get; }
    public SelfAttentionNoise selfAttNoise                       { set; get; }
    public FeedForwardLayerNoise feedForwardNoise                { set; get; }
    public SelfAttentionNoise selfAtt2Noise                      { set; get; }
    public FeedForwardLayerNoise feedForward2Noise               { set; get; }
    public OutputProjectionLayerNoise outputProjectionLayerNoise { set; get; }
}

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

public class TransformerModel
{
    public string            DirPath     { get; set; } = "";
    public TransformerModelFilenames Filenames;
    public TransformerModelDetails ModelDetails = new TransformerModelDetails();

    public TokenVocab?            Vocab            { get; set; } = null;
    public TokenBigram?           Bigram           { get; set; } = null;
    public EmbeddingLayer?        Embedding        { get; set; } = null;
    public PositionalEncoder?     PositionalEnc    { get; set; } = null;
    public SelfAttention?         SelfAtt          { get; set; } = null;
    public FeedForwardLayer?      FeedForward      { get; set; } = null;
    public SelfAttention?         SelfAtt2         { get; set; } = null;
    public FeedForwardLayer?      FeedForward2     { get; set; } = null;
    public OutputProjectionLayer? OutputProjection { get; set; } = null;

    private Random random = new Random();

    // --------------------------------------------------------------------------------------------
    // MARK: Constructor
    // --------------------------------------------------------------------------------------------

    public TransformerModel(string dirPath)
    {
        DirPath   = dirPath;
        EnsurePathExists(DirPath);

        Filenames = new(DirPath);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Create Model
    // --------------------------------------------------------------------------------------------

    public void Create01_CreateVocab(string vocabFilepath, int targetTokenCount)
    {
        Vocab = new TokenVocab();

        Console.WriteLine("- - - DemoTokenVocab - - - - - -");

        Vocab.SaveToFile(Filenames.VocabPath);

        // Load the input string from file
        string input = File.ReadAllText(vocabFilepath);

        int newTokenPerIteration = 25;
        int currIterationcount = 0;
        int prevCount = 0;
        int targetCount = (int)(targetTokenCount * 1.1); // Overshoot, so we can limit the size later when culling control chars etc
        int maxIterations = (int)((targetCount / newTokenPerIteration) * 1.5);
        while (Vocab.Count < targetCount)
        {
            Vocab.ApplyBPEIteration(input, newTokenPerIteration); // Add 25 tokens at a time
            Console.Write($"{Vocab.Count} ");

            // Break out of the loop in case we aren't making progress, either but loops or by count
            if (currIterationcount > maxIterations) break;
            if (Vocab.Count == prevCount) break;
            currIterationcount++;
            prevCount = Vocab.Count;
        }
        Console.Write($"Done\n");
        Vocab.SaveToFile(Filenames.VocabPath);

        // delay a moment for file save
        System.Threading.Thread.Sleep(100);

        TokenVocab.PerformLimitSizePass(Filenames.VocabPath, targetTokenCount);
        Vocab = TokenVocab.LoadFromFile(Filenames.VocabPath);

        ModelDetails.VocabSize = Vocab.Count;
    }

    public void Create01_CreateBigram(string vocabFilepath)
    {
        // Load the input string from file
        string input = File.ReadAllText(vocabFilepath);
        List<string> tokList   = Vocab!.TokenizeToStrings(input);
        List<int>    tokIdList = Vocab!.TokenizeToIds(input);

        Bigram = new TokenBigram(ModelDetails.VocabSize);
        Bigram.AddTokenLinks(tokIdList);

        // Clear the entries for fixed tokens
        Bigram.ClearLinksForToken(0); // <PAD>
        Bigram.ClearLinksForToken(1); // <UNK>
        Bigram.ClearLinksForToken(2); // <EOS>
    }


    public void Create02_CreateEmbedding(int embeddingDim)
    {
        ModelDetails.EmbeddingDim = embeddingDim;

        if (Vocab == null)
            throw new Exception("Vocab must be created before creating the embedding layer.");

        Embedding = new EmbeddingLayer(Vocab!.Count, ModelDetails.EmbeddingDim);

        Embedding.SaveToFile(Filenames.EmbeddingPath);
    }

    public void Create03_CreatePositionalEncoding(int inputLen)
    {
        ModelDetails.InputLen = inputLen;

        if (Embedding == null)
            throw new Exception("Embedding must be created before creating the positional encoding.");

        // Create the positional encoding matrix.
        // PositionalEncoding = new PositionalEncoding(EmbeddingDim, MaxLength);

        PositionalEnc = new PositionalEncoder(ModelDetails.InputLen, ModelDetails.EmbeddingDim);
    }

    public void Create04_CreateSelfAttention()
    {
        if (Embedding == null)
            throw new Exception("Embedding must be created before creating the self-attention layer.");

        SelfAtt = new SelfAttention(ModelDetails.InputLen, ModelDetails.EmbeddingDim);
    }

    public void Create05_CreateFeedForward()
    {
        if (SelfAtt == null)
            throw new Exception("Self-attention must be created before creating the feed-forward layer.");

        ModelDetails.FFHiddenDim = ModelDetails.EmbeddingDim * 4;
        FeedForward = new FeedForwardLayer(ModelDetails.EmbeddingDim, ModelDetails.FFHiddenDim);
    }

    public void Create04b_CreateSelfAttention2()
    {
        if (FeedForward == null)
            throw new Exception("Feed-forward must be created before creating the second self-attention layer.");

        SelfAtt2 = new SelfAttention(ModelDetails.InputLen, ModelDetails.EmbeddingDim);
    }

    public void Create05b_CreateFeedForward2()
    {
        if (SelfAtt2 == null)
            throw new Exception("Second self-attention must be created before creating the second feed-forward layer.");

        FeedForward2 = new FeedForwardLayer(ModelDetails.EmbeddingDim, ModelDetails.FFHiddenDim);
    }

    public void Create06_CreateOutputProjection()
    {
        //if (FeedForward == null)
        //    throw new Exception("Feed-forward must be created before creating the output projection.");

        OutputProjection = new OutputProjectionLayer(ModelDetails.EmbeddingDim, Vocab!.Count);
    }

    public void Create07_SetupNoise(float noiseVal, float percentChange)
    {
        ModelDetails.NoiseVal      = noiseVal;
        ModelDetails.PercentChange = percentChange;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    // Create a complete duplicate of the model

    public TransformerModel DeepCopy()
    {
        TransformerModel newModel = new TransformerModel(DirPath);

        newModel.ModelDetails     = ModelDetails;

        newModel.Vocab            = Vocab!.DeepCopy();
        newModel.Embedding        = Embedding!.DeepCopy();
        newModel.PositionalEnc    = PositionalEnc!.DeepCopy();
        newModel.SelfAtt          = SelfAtt!.DeepCopy();
        newModel.FeedForward      = FeedForward!.DeepCopy();
        newModel.SelfAtt2         = SelfAtt2!.DeepCopy();
        newModel.FeedForward2     = FeedForward2!.DeepCopy();
        newModel.OutputProjection = OutputProjection!.DeepCopy();

        return newModel;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Mutation
    // --------------------------------------------------------------------------------------------

    // Add random +/- noise to all model parameters.
    public void AddNoise(float absNoiseVal, float percentToChange)
    {
        //float percentToChange = 1f; // 5% of values to change

        // create a random number up to the specified value
        float realNoise   = (float)(random.NextDouble()) * absNoiseVal;
        float realPercent = (float)(random.NextDouble()) * percentToChange;

        float fractionTochange = realPercent / 100f;

        Embedding?.AddLimitedNoise(realNoise, fractionTochange);
        SelfAtt?.AddLimitedNoise(realNoise, fractionTochange);
        FeedForward?.AddLimitedNoise(realNoise, fractionTochange);
        SelfAtt2?.AddLimitedNoise(realNoise, fractionTochange);
        FeedForward2?.AddLimitedNoise(realNoise, fractionTochange);
        OutputProjection?.AddLimitedNoise(realNoise, fractionTochange);
    }

    public ModelNoise CreateLimitedNoise(float absNoiseVal, float percentToChange)
    {
        //float percentToChange = 1f; // 5% of values to change

        // create a random number up to the specified value
        float realNoise   = (float)(random.NextDouble()) * absNoiseVal;
        float realPercent = (float)(random.NextDouble()) * percentToChange;

        float fractionTochange = realPercent / 100f;

        ModelNoise noise = new ModelNoise();

        // record the metrics, so we can analyse the successful results
        noise.recordedNoise         = realNoise;
        noise.recordedPercentChange = realPercent;

        noise.embeddingNoise             = Embedding!.CreateLimitedNoise(realNoise, fractionTochange);
        noise.selfAttNoise               = SelfAtt!.CreateLimitedNoise(realNoise, fractionTochange);
        noise.feedForwardNoise           = FeedForward!.CreateLimitedNoise(realNoise, fractionTochange);
        noise.selfAtt2Noise              = SelfAtt2!.CreateLimitedNoise(realNoise, fractionTochange);
        noise.feedForward2Noise          = FeedForward2!.CreateLimitedNoise(realNoise, fractionTochange);
        noise.outputProjectionLayerNoise = OutputProjection!.CreateLimitedNoise(realNoise, fractionTochange);

        return noise;
    }

    public void ApplyNoise(ModelNoise noise)
    {
        Embedding!.ApplyNoise(noise.embeddingNoise);
        SelfAtt!.ApplyNoise(noise.selfAttNoise);
        FeedForward!.ApplyNoise(noise.feedForwardNoise);
        SelfAtt2!.ApplyNoise(noise.selfAtt2Noise);
        FeedForward2!.ApplyNoise(noise.feedForward2Noise);
        OutputProjection!.ApplyNoise(noise.outputProjectionLayerNoise);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    public string PredictNextToken(string inputText)
    {
        // Tokenize the input text.
        List<string> tokenStrList = Vocab!.TokenizeToStrings(inputText);
        List<int>    tokenIdList  = Vocab!.TokenizeToIds(inputText);

        // trim to the right length, either the last n tokens or the whole list with padding if needed
        if (tokenIdList.Count > ModelDetails.InputLen)
            tokenIdList = tokenIdList.GetRange(tokenIdList.Count - ModelDetails.InputLen, ModelDetails.InputLen);
        while (tokenIdList.Count < ModelDetails.InputLen)
        {
            tokenStrList.Add("<PAD>");
            tokenIdList.Add(Vocab!.GetTokenId("<PAD>"));
        }

        // Print the input tokens and Ids
        Console.WriteLine("Input: {tokenStrList.Count} tokens:");
        for (int i=0; i<tokenIdList.Count; i++)
            Console.Write($"[{tokenStrList[i]}: {tokenIdList[i]}] ");
        Console.Write("\n");

        //AddNoise(0.1f);


        // Get the embeddings for the input tokens.
        var embeddings = Embedding!.LookupListToMatrix(tokenIdList);

        // Print the embedding matrix
        Console.WriteLine("Embeddings:");
        Console.WriteLine(embeddings);

        // Apply the positional encoding to the embeddings.
        var encodedEmbeddings = PositionalEnc!.ApplyPositionalEncoding(embeddings);

        // Print the encoded embeddings
        Console.WriteLine("Encoding:");
        Console.WriteLine(PositionalEnc!.EncodingMatrix);

        // Print the encoded embeddings
        Console.WriteLine("Encoded Embeddings (Embeddings+Encoding):");
        Console.WriteLine(encodedEmbeddings);

        // Apply the self-attention mechanism.
        var selfAttOutput = SelfAtt!.Forward(encodedEmbeddings);

        // Self Attention Output
        Console.WriteLine($"Self-Attention Output:");
        Console.WriteLine(selfAttOutput);

        // Apply the feed-forward layer.
        var feedForwardOutput = FeedForward!.Forward(selfAttOutput);

        // Apply second self-attention and feed-forward layers
        var selfAttOutput2 = SelfAtt2!.Forward(feedForwardOutput);
        var feedForwardOutput2 = FeedForward2!.Forward(selfAttOutput2);

        // Report the next token
        int nextTokenID = OutputProjection!.PredictNextToken(feedForwardOutput2);
        string nextTokenStr = Vocab!.GetTokenString(nextTokenID);
        Console.WriteLine($"Next Token: [{nextTokenStr}: {nextTokenID}]");

        // Report the highest ranked 5 tokens
        var topTokens = OutputProjection.TopNTokens(feedForwardOutput2, 5);
        Console.WriteLine("Top 5 tokens:");
        foreach (var (tokenId, prob) in topTokens)
        {
            string tokenStr = Vocab!.GetTokenString(tokenId);
            Console.WriteLine($"- [{tokenStr}: {tokenId}] with probability {prob:F4}");
        }

        // determine the loss score
        Console.WriteLine($"Loss: {OutputProjection!.Loss(feedForwardOutput2, nextTokenID - 1)}");

        return nextTokenStr;
    }

    public int PredictNextToken(List<int> tokenIdList)
    {
        // Get the embeddings for the input tokens.
        var embeddings = Embedding!.LookupListToMatrix(tokenIdList);

        // Apply the positional encoding to the embeddings.
        var encodedEmbeddings = PositionalEnc!.ApplyPositionalEncoding(embeddings);

        // Apply the self-attention mechanism.
        var selfAttOutput = SelfAtt!.Forward(encodedEmbeddings);
        var feedForwardOutput = FeedForward!.Forward(selfAttOutput);
        var selfAttOutput2 = SelfAtt2!.Forward(feedForwardOutput);
        var feedForwardOutput2 = FeedForward2!.Forward(selfAttOutput2);

        // Report the next token
        int nextTokenID = OutputProjection!.PredictNextToken(feedForwardOutput2);

        return nextTokenID;
    }


    public float PredictionScore(List<int> tokenIdList, int expectedNextTokenId)
    {
        // // Tokenize the input text.
        // List<string> tokenStrList = Vocab!.TokenizeToStrings(inputText);
        // List<int>    tokenIdList  = Vocab!.TokenizeToIds(inputText);

        // // trim to the right length, either the last n tokens or the whole list with padding if needed
        // if (tokenIdList.Count > InputLen)
        //     tokenIdList = tokenIdList.GetRange(tokenIdList.Count - InputLen, InputLen);
        // while (tokenIdList.Count < InputLen)
        // {
        //     tokenStrList.Add("<PAD>");
        //     tokenIdList.Add(Vocab!.GetTokenId("<PAD>"));
        // }

        // // Print the input tokens and Ids
        // Console.WriteLine("Input: {tokenStrList.Count} tokens:");
        // for (int i=0; i<tokenIdList.Count; i++)
        //     Console.Write($"[{tokenStrList[i]}: {tokenIdList[i]}] ");
        // Console.Write("\n");

        // AddNoise(0.1f);


        // Get the embeddings for the input tokens.
        var embeddings = Embedding!.LookupListToMatrix(tokenIdList);

        // Print the embedding matrix
        // Console.WriteLine("Embeddings:");
        // Console.WriteLine(embeddings);

        // Apply the positional encoding to the embeddings.
        var encodedEmbeddings = PositionalEnc!.ApplyPositionalEncoding(embeddings);

        // Print the encoded embeddings
        // Console.WriteLine("Encoding:");
        // Console.WriteLine(PositionalEnc!.EncodingMatrix);

        // Print the encoded embeddings
        // Console.WriteLine("Encoded Embeddings (Embeddings+Encoding):");
        // Console.WriteLine(encodedEmbeddings);

        // Apply the self-attention mechanism.
        var selfAttOutput = SelfAtt!.Forward(encodedEmbeddings);

        // Apply the feed-forward layer.
        var feedForwardOutput = FeedForward!.Forward(selfAttOutput);
        var selfAttOutput2 = SelfAtt2!.Forward(feedForwardOutput);
        var feedForwardOutput2 = FeedForward2!.Forward(selfAttOutput2);

        // Report the next token
        // int nextTokenID = OutputProjection!.PredictNextToken(selfAttOutput);
        // string nextTokenStr = Vocab!.GetTokenString(nextTokenID);
        // Console.WriteLine($"Next Token: [{nextTokenStr}: {nextTokenID}]");

        // // Report the highest ranked 5 tokens
        // var topTokens = OutputProjection.TopNTokens(selfAttOutput, 5);
        // Console.WriteLine("Top 5 tokens:");
        // foreach (var (tokenId, prob) in topTokens)
        // {
        //     string tokenStr = Vocab!.GetTokenString(tokenId);
        //     Console.WriteLine($"- [{tokenStr}: {tokenId}] with probability {prob:F4}");
        // }

        float loss = OutputProjection!.Loss(feedForwardOutput2, expectedNextTokenId);

        return loss;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Report
    // --------------------------------------------------------------------------------------------

    public string Report()
    {
        StringBuilder sb = new StringBuilder();

        sb.AppendLine("Model Report");
        sb.AppendLine("------------");
        sb.AppendLine($"Model Path: {DirPath}");
        sb.AppendLine($"Model Parameters: {ParamCount()}");
        sb.AppendLine($"Objects Present: [Vocab: {Vocab != null}] [Embedding: {Embedding != null}] [Positional Encoding: {PositionalEnc != null}] [Self-Attention: {SelfAtt != null}] [FeedForward: {FeedForward != null}] [Self-Attention2: {SelfAtt2 != null}] [FeedForward2: {FeedForward2 != null}] [Output Projection: {OutputProjection != null}]");
        sb.AppendLine($"Vocab Size: {Vocab!.Count}");
        sb.AppendLine($"Embedding: {Embedding!.Report()}");
        sb.AppendLine($"Positional Encoding: {PositionalEnc!.Report()}");
        sb.AppendLine($"Self-Attention: {SelfAtt!.Report()}");
        sb.AppendLine($"FeedForward: {FeedForward!.Report()}");
        sb.AppendLine($"Self-Attention2: {SelfAtt2!.Report()}");
        sb.AppendLine($"FeedForward2: {FeedForward2!.Report()}");
        sb.AppendLine($"Output Projection: {OutputProjection!.Report()}");

        return sb.ToString();
    }

    public float CheckSum()
    {
        float sum = 0.0f;

        //sum += Vocab!.CheckSum();
        sum += Embedding!.CheckSum();
        sum += PositionalEnc!.CheckSum();
        sum += SelfAtt!.CheckSum();
        sum += FeedForward!.CheckSum();
        sum += SelfAtt2!.CheckSum();
        sum += FeedForward2!.CheckSum();
        sum += OutputProjection!.CheckSum();

        return sum;
    }

    public int ParamCount()
    {
        int count = 0;

        count += Embedding!.ParamCount();
        //count += PositionalEnc!.ParamCount();
        count += SelfAtt!.ParamCount();
        count += FeedForward!.ParamCount();
        count += SelfAtt2!.ParamCount();
        count += FeedForward2!.ParamCount();
        count += OutputProjection!.ParamCount();

        return count;
    }

    public void AppendLog(int cycleCount, float score)
    {
        string logLine = $"{cycleCount}, {score:F4}, {ModelDetails.NoiseVal:F4}, {ModelDetails.PercentChange:F4}\n";
        File.AppendAllText(Filenames.TrainingLogPath, logLine);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    public void EnsurePathExists(string path)
    {
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
    }

    public void SaveModel()
    {
        // check the directory exists
        if (!Directory.Exists(DirPath))
            Directory.CreateDirectory(DirPath);

        // Save the model parameters to a JSON file.
        // string modelPath = Path.Combine(DirPath, "model.json");
        // string modelJson = JsonSerializer.Serialize(this);
        // File.WriteAllText(modelPath, modelJson);

        Filenames = new(DirPath);

        ModelDetails.SaveToFile(Filenames.ModelPath);
        Vocab?.SaveToFile(Filenames.VocabPath);
        Bigram?.SaveToFile(Filenames.BigramPath);
        Embedding?.SaveToFile(Filenames.EmbeddingPath);
        SelfAtt?.SaveToFile(Filenames.SelfAttPath);
        FeedForward?.SaveToFile(Filenames.FeedForwardPath);
        SelfAtt2?.SaveToFile(Filenames.SelfAtt2Path);
        FeedForward2?.SaveToFile(Filenames.FeedForward2Path);
        OutputProjection?.SaveToFile(Filenames.OutputProjectionPath);

        Embedding?.SaveToBinary(Filenames.BinEmbeddingPath);
        SelfAtt?.SaveToBinary(Filenames.BinSelfAttPath);
        FeedForward?.SaveToBinary(Filenames.BinFeedForwardPath);
        SelfAtt2?.SaveToBinary(Filenames.BinSelfAtt2Path);
        FeedForward2?.SaveToBinary(Filenames.BinFeedForward2Path);
        OutputProjection?.SaveToBinary(Filenames.BinOutputProjectionPath);

    }

    // --------------------------------------------------------------------------------------------

    public static TransformerModel LoadModel(string dirPath)
    {
        TransformerModel model = new TransformerModel(dirPath);
        model.ModelDetails = TransformerModelDetails.LoadFromFile(model.Filenames.ModelPath);

        // Load the model parameters from a JSON file.
        model.Vocab            = TokenVocab.LoadFromFile(model.Filenames.VocabPath);

        model.Bigram = TokenBigram.LoadFromFile(model.Filenames.BigramPath);

        //model.Embedding        = EmbeddingLayer.LoadFromFile(model.Filenames.EmbeddingPath);
        model.Embedding = EmbeddingLayer.LoadFromBinary(model.Filenames.BinEmbeddingPath);

        model.Create03_CreatePositionalEncoding(model.ModelDetails.InputLen);

        //model.SelfAtt          = SelfAttention.LoadFromFile(model.Filenames.SelfAttPath);
        model.SelfAtt          = SelfAttention.LoadFromBinary(model.Filenames.BinSelfAttPath);

        model.FeedForward      = FeedForwardLayer.LoadFromBinary(model.Filenames.BinFeedForwardPath);
        model.SelfAtt2         = SelfAttention.LoadFromBinary(model.Filenames.BinSelfAtt2Path);
        model.FeedForward2     = FeedForwardLayer.LoadFromBinary(model.Filenames.BinFeedForward2Path);

        //model.OutputProjection = OutputProjectionLayer.LoadFromFile(model.Filenames.OutputProjectionPath);
        model.OutputProjection = OutputProjectionLayer.LoadFromBinary(model.Filenames.BinOutputProjectionPath);

        return model;
    }

    // --------------------------------------------------------------------------------------------


}