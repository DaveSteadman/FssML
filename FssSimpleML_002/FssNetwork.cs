using System;
using System.Collections.Generic;

public class FssNetwork
{
    public List<FssNode>       Nodes       { get; set; } // All nodes in the network
    public List<FssConnection> Connections { get; set; } // Flat list of all connections
    public int[]               LayerSizes  { get; set; } // Track the size of each layer

    // Constructor to initialize the network based on layer sizes
    public FssNetwork(int[] layerSizes)
    {
        LayerSizes  = layerSizes;
        Nodes       = new List<FssNode>();
        Connections = new List<FssConnection>();

        // Create nodes for each layer
        for (int i = 0; i < layerSizes.Length; i++)
        {
            for (int j = 0; j < layerSizes[i]; j++)
            {
                Nodes.Add(new FssNode());
            }
        }

        // Create connections between layers
        int previousLayerStart = 0;
        for (int i = 1; i < layerSizes.Length; i++)
        {
            int currentLayerStart = previousLayerStart + layerSizes[i - 1];

            for (int j = 0; j < layerSizes[i]; j++)
            {
                FssNode currentNode = Nodes[currentLayerStart + j];

                for (int k = 0; k < layerSizes[i - 1]; k++)
                {
                    FssNode previousNode = Nodes[previousLayerStart + k];
                    AddConnection(previousNode, currentNode);
                }
            }

            previousLayerStart += layerSizes[i - 1];
        }
    }

    // Function to add a connection between two nodes
    public void AddConnection(FssNode fromNode, FssNode toNode)
    {
        FssConnection connection = new FssConnection(Nodes.IndexOf(fromNode), Nodes.IndexOf(toNode));
        fromNode.AddConnectionIndex(Connections.Count); // Store the index in the node
        Connections.Add(connection);
    }

    // Function to evolve the network
    public void Evolve(double learningRate)
    {
        // Evolve biases in nodes
        foreach (var node in Nodes)
        {
            node.EvolveBias(learningRate);
        }

        // Evolve all connections in the flat list
        foreach (var connection in Connections)
        {
            connection.EvolveWeight(learningRate);
        }
    }

    // Feedforward function
    public double[] FeedForward(double[] inputs)
    {
        if (inputs.Length != LayerSizes[0])
        {
            throw new ArgumentException("Input size does not match the input layer size.");
        }

        // Copy input data to the first layer
        double[] currentOutputs = inputs;

        // Forward through each layer
        int nodeIndex = LayerSizes[0]; // Skip input layer as we process it directly
        for (int i = 1; i < LayerSizes.Length; i++) // Starting from the first hidden layer
        {
            double[] nextOutputs = new double[LayerSizes[i]];

            for (int j = 0; j < LayerSizes[i]; j++)
            {
                var currentNode = Nodes[nodeIndex];
                nextOutputs[j] = currentNode.Activate(Connections, currentOutputs);
                nodeIndex++;
            }

            currentOutputs = nextOutputs; // Set outputs as inputs for the next layer
        }

        return currentOutputs; // Final layer's outputs
    }

    // Function to retrieve a connection by index
    public FssConnection GetConnection(int index)
    {
        return Connections[index];
    }

    // Function to remove a connection safely by index
    public void RemoveConnection(int index)
    {
        if (index >= 0 && index < Connections.Count)
        {
            Connections.RemoveAt(index);

            // Adjust all connection indices in the nodes
            foreach (var node in Nodes)
            {
                for (int i = 0; i < node.ConnectionIndices.Count; i++)
                {
                    if (node.ConnectionIndices[i] > index)
                    {
                        node.ConnectionIndices[i]--; // Shift down the indices after removal
                    }
                    else if (node.ConnectionIndices[i] == index)
                    {
                        node.ConnectionIndices.RemoveAt(i); // Remove the dead connection reference
                        i--;
                    }
                }
            }
        }
    }

    // Example of connection toggling for "dying off"
    public void DisableConnection(int index)
    {
        if (index >= 0 && index < Connections.Count)
        {
            Connections[index].Disable();
        }
    }

    // Add a new connection and return its index
    public int AddNewConnection(FssNode fromNode, FssNode toNode)
    {
        FssConnection newConnection = new FssConnection(Nodes.IndexOf(fromNode), Nodes.IndexOf(toNode));
        fromNode.AddConnectionIndex(Connections.Count);
        Connections.Add(newConnection);
        return Connections.Count - 1;
    }
}
