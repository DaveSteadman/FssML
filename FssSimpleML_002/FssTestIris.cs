using System;

public class FssTestIris
{
    // Function to set up and create a new network
    public static FssNetwork CreateIrisNetwork()
    {
        // Typical setup for the Iris dataset: 4 inputs, 5 hidden nodes, 3 outputs
        int[] layerSizes = new int[] { 4, 5, 3 };
        FssNetwork network = new FssNetwork(layerSizes);
        Console.WriteLine("Network created with layer sizes: 4 -> 5 -> 3");
        return network;
    }

    // Function to save the network to a file
    public static void SaveNetwork(FssNetwork network, string filePath)
    {
        FssNetworkIO.SaveToFile(network, filePath);
        Console.WriteLine($"Network saved to {filePath}");
    }

    // Function to load the network from a file
    public static FssNetwork LoadNetwork(string filePath)
    {
        FssNetwork network = FssNetworkIO.LoadFromFile(filePath);
        Console.WriteLine($"Network loaded from {filePath}");
        return network;
    }

    // Entry point to test the NetworkIO functionality
    public static void TestNetworkIO()
    {
        string filePath = "iris_network.txt";

        // Step 1: Create a new network
        FssNetwork createdNetwork = CreateIrisNetwork();

        // Step 2: Save the network to a file
        SaveNetwork(createdNetwork, filePath);

        // Step 3: Load the network from the file
        FssNetwork loadedNetwork = LoadNetwork(filePath);

        // Check if the loaded network has the same structure
        Console.WriteLine("Original network's layer sizes:");
        Console.WriteLine(string.Join(" -> ", createdNetwork.LayerSizes));
        Console.WriteLine("Loaded network's layer sizes:");
        Console.WriteLine(string.Join(" -> ", loadedNetwork.LayerSizes));
    }
}
