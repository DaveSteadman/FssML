



using System;


public struct TransformerModelFilenames
{
    public string VocabPath            { get; set; }
    public string EmbeddingPath        { get; set; }
    public string SelfAttPath          { get; set; }
    public string FeedForwardPath      { get; set; }
    public string OutputProjectionPath { get; set; }

    public TransformerModelFilenames(string dirPath)
    {
        VocabPath            = System.IO.Path.Combine(dirPath, "vocab.txt");
        EmbeddingPath        = System.IO.Path.Combine(dirPath, "embedding.txt");
        SelfAttPath          = System.IO.Path.Combine(dirPath, "selfatt.txt");
        FeedForwardPath      = System.IO.Path.Combine(dirPath, "feedforward.txt");
        OutputProjectionPath = System.IO.Path.Combine(dirPath, "outputprojection.txt");
    }
}

public class TransformerModel
{
    public string            DirPath     { get; set; } = "";
    public TransformerModelFilenames Filenames;

    public TokenVocab?            Vocab            { get; set; } = null;
    public EmbeddingLayer?        Embedding        { get; set; } = null;
    public PositionalEncoder?     PositionalEnc    { get; set; } = null;
    public SelfAttention?         SelfAtt          { get; set; } = null;
    public FeedForwardLayer?      FeedForward      { get; set; } = null;
    public OutputProjectionLayer? OutputProjection { get; set; } = null;

    public int EmbeddingDim = 0;
    public int FFHiddenDim  = 0;

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
        TokenVocab.PerformLimitSizePass(Filenames.VocabPath, targetTokenCount);
        Vocab = TokenVocab.LoadFromFile(Filenames.VocabPath);
    }

    public void Create02_CreateEmbedding(int embeddingDim)
    {
        EmbeddingDim = embeddingDim;

        if (Vocab == null)
            throw new Exception("Vocab must be created before creating the embedding layer.");

        Embedding = new EmbeddingLayer(Vocab!.Count, EmbeddingDim);

        Embedding.SaveToFile(Filenames.EmbeddingPath);
    }

    public void Create03_CreatePositionalEncoding()
    {
        if (Embedding == null)
            throw new Exception("Embedding must be created before creating the positional encoding.");

        // Create the positional encoding matrix.
        // PositionalEncoding = new PositionalEncoding(EmbeddingDim, MaxLength);

        PositionalEnc = new PositionalEncoder(EmbeddingDim, EmbeddingDim);
    }

    public void Create04_CreateSelfAttention()
    {
        if (Embedding == null)
            throw new Exception("Embedding must be created before creating the self-attention layer.");

        SelfAtt = new SelfAttention(EmbeddingDim);
    }

    public void Create05_CreateFeedForward()
    {
        if (SelfAtt == null)
            throw new Exception("Self-attention must be created before creating the feed-forward layer.");

        FFHiddenDim = EmbeddingDim * 4;
        FeedForward = new FeedForwardLayer(EmbeddingDim, FFHiddenDim);
    }

    public void Create06_CreateOutputProjection()
    {
        if (FeedForward == null)
            throw new Exception("Feed-forward must be created before creating the output projection.");

        OutputProjection = new OutputProjectionLayer(EmbeddingDim, Vocab!.Count);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Prediction
    // --------------------------------------------------------------------------------------------

    public string PredictNextToken(string inputText)
    {
        // Tokenize the input text.
        List<string> TokenStrList = Vocab!.TokenizeToStrings(inputText);
        List<int>    tokenIdList  = Vocab!.TokenizeToIds(inputText);

        // Get the embeddings for the input tokens.
        var embeddings = Embedding!.LookupListToMatrix(tokenIdList);

        // Apply the positional encoding to the embeddings.
        var encodedEmbeddings = PositionalEnc!.ApplyPositionalEncoding(embeddings);

        // Apply the self-attention mechanism.
        var selfAttOutput = SelfAtt!.Forward(encodedEmbeddings);

        // Apply the feed-forward layer.
        //var feedForwardOutput = FeedForward.Apply(selfAttOutput);

        // Apply the output projection layer.
        var outputProjections = OutputProjection!.Forward(selfAttOutput);

        // Get the token ID of the most likely next token.
        // var nextTokenId = Vocab.GetMostLikelyToken(outputProjections);

        // Return the next token as a string.
        // return Vocab.GetToken(nextTokenId);
        return "";
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

        Vocab?.SaveToFile(Filenames.VocabPath);
        Embedding?.SaveToFile(Filenames.EmbeddingPath);
        SelfAtt?.SaveToFile(Filenames.SelfAttPath);
        FeedForward?.SaveToFile(Filenames.FeedForwardPath);
        OutputProjection?.SaveToFile(Filenames.OutputProjectionPath);
    }

    // --------------------------------------------------------------------------------------------

    public static TransformerModel LoadModel(string dirPath)
    {
        TransformerModel model = new TransformerModel(dirPath);

        // Load the model parameters from a JSON file.
        model.Vocab            = TokenVocab.LoadFromFile(model.Filenames.VocabPath);
        model.Embedding        = EmbeddingLayer.LoadFromFile(model.Filenames.EmbeddingPath);

        model.Create03_CreatePositionalEncoding();

        model.SelfAtt          = SelfAttention.LoadFromFile(model.Filenames.SelfAttPath);
        model.FeedForward      = FeedForwardLayer.LoadFromFile(model.Filenames.FeedForwardPath);
        model.OutputProjection = OutputProjectionLayer.LoadFromFile(model.Filenames.OutputProjectionPath);

        return model;
    }

    // --------------------------------------------------------------------------------------------


}