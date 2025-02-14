


// TrainingFramework:
// - Creates/Loads a model, performs training passes and serializes the model back to disk
// - Outputs to a loss log file so we can plot progress

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

    public static void TrainModel(string modeldirname, string trainingdata)
    {
        TransformerModel model  = TransformerModel.LoadModel("./Model_001");


    }


}