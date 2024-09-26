using System;
using System.IO;
using System.Text;
using System.Collections.Generic;

public class FssNetworkIO
{
    // Serialize the network into a human-readable format
    public static string Serialize(FssNetwork network)
    {
        StringBuilder sb = new StringBuilder();

        // Serialize layer info
        sb.AppendLine("# Layers");
        sb.AppendLine($"InputLayer: {network.LayerSizes[0]}");
        for (int i = 1; i < network.LayerSizes.Length - 1; i++)
        {
            sb.AppendLine($"HiddenLayer: {network.LayerSizes[i]}");
        }
        sb.AppendLine($"OutputLayer: {network.LayerSizes[network.LayerSizes.Length - 1]}");

        // Serialize nodes
        sb.AppendLine("# Nodes");
        for (int i = 0; i < network.Nodes.Count; i++)
        {
            sb.AppendLine($"Node_{i} Bias: {network.Nodes[i].Bias}");
        }

        // Serialize connections
        sb.AppendLine("# Connections");
        for (int i = 0; i < network.Connections.Count; i++)
        {
            var connection = network.Connections[i];
            sb.AppendLine($"Connection: {connection.FromNodeIndex} -> {connection.ToNodeIndex} Weight: {connection.Weight}");
        }

        return sb.ToString();
    }

    // Deserialize a network from a string
    public static FssNetwork Deserialize(string serializedNetwork)
    {
        StringReader reader = new StringReader(serializedNetwork);
        string? line; // Use nullable type for the line
        List<int> layerSizes = new List<int>();
        List<FssNode> nodes = new List<FssNode>();
        List<FssConnection> connections = new List<FssConnection>();

        // Read layer info
        while ((line = reader.ReadLine()) != null)
        {
            try
            {
                if (line.StartsWith("InputLayer"))
                {
                    layerSizes.Add(int.Parse(line.Split(": ")[1]));
                }
                else if (line.StartsWith("HiddenLayer"))
                {
                    layerSizes.Add(int.Parse(line.Split(": ")[1]));
                }
                else if (line.StartsWith("OutputLayer"))
                {
                    layerSizes.Add(int.Parse(line.Split(": ")[1]));
                }
                else if (line.StartsWith("Node"))
                {
                    // Split the line into parts and check for the keyword "Bias: "
                    string[] biasParts = line.Split(new string[] { "Bias: " }, StringSplitOptions.None);

                    if (biasParts.Length == 2) // Ensure we got two parts (Node and Bias value)
                    {
                        try
                        {
                            double bias = double.Parse(biasParts[1]); // Parse the bias value
                            nodes.Add(new FssNode(bias));
                        }
                        catch (FormatException ex)
                        {
                            Console.WriteLine($"Error parsing bias value in line: {line}");
                            Console.WriteLine($"Exception: {ex.Message}");
                        }
                    }
                    else
                    {
                        throw new Exception($"Unexpected format for Node line: {line}");
                    }
                }
                else if (line.StartsWith("Connection"))
                {
                    // Parse connection info
                    var parts = line.Split(' ');
                    if (parts.Length >= 6)
                    {
                        string nameStr   = parts[0];
                        string startId   = parts[1];
                        string arrow     = parts[2];
                        string endId     = parts[3];
                        string weightStr = parts[4];
                        string weightVal = parts[5];

                        int fromNodeIndex = int.Parse(startId);
                        int toNodeIndex   = int.Parse(endId);
                        double weight     = double.Parse(weightVal);

                        // Check if the node indices are valid
                        if (fromNodeIndex >= 0 && fromNodeIndex < nodes.Count &&
                            toNodeIndex >= 0 && toNodeIndex < nodes.Count)
                        {
                            connections.Add(new FssConnection(fromNodeIndex, toNodeIndex, weight));
                        }
                        else
                        {
                            throw new IndexOutOfRangeException("Connection references invalid node indices.");
                        }
                    }
                    else
                    {
                        throw new Exception($"Unexpected format for Connection line: {line}");
                    }
                }
            }
            catch (Exception ex)
            {
                // Print the problematic line and the exception message
                Console.WriteLine($"Error processing line: {line}");
                Console.WriteLine($"Exception: {ex.Message}");
            }
        }

        // Build the network from the loaded data
        FssNetwork network = new FssNetwork(layerSizes.ToArray());
        network.Nodes = nodes;
        network.Connections = connections;

        return network;
    }


    // Save the serialized network to a file
    public static void SaveToFile(FssNetwork network, string filePath)
    {
        string serializedNetwork = Serialize(network);
        File.WriteAllText(filePath, serializedNetwork);
    }

    // Load a network from a file
    public static FssNetwork LoadFromFile(string filePath)
    {
        string serializedNetwork = File.ReadAllText(filePath);
        return Deserialize(serializedNetwork);
    }
}
