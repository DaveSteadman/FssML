using System;
using System.Collections.Generic;

public class FssNode
{
    public double Bias { get; private set; }
    public List<int> ConnectionIndices { get; private set; } // Indices into the flat connection list
    private static Random random = new Random();

    // Constructor
    public FssNode()
    {
        Bias = random.NextDouble() * 2 - 1; // Initialize bias randomly between -1 and 1
        ConnectionIndices = new List<int>();
    }

    // Constructor with predefined bias (useful for deserialization)
    public FssNode(double bias)
    {
        Bias = bias;
        ConnectionIndices = new List<int>();
    }

    // Add an index to the list of connection indices
    public void AddConnectionIndex(int index)
    {
        ConnectionIndices.Add(index);
    }

    // Activation function using flat list of connections
    public double Activate(List<FssConnection> allConnections, double[] inputs)
    {
        double weightedSum = 0.0;
        for (int i = 0; i < ConnectionIndices.Count; i++)
        {
            var connection = allConnections[ConnectionIndices[i]];
            weightedSum += inputs[i] * connection.Weight;
        }
        weightedSum += Bias;
        return Sigmoid(weightedSum);
    }

    // Sigmoid activation function
    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    // Evolve the bias with small random adjustments
    public void EvolveBias(double learningRate)
    {
        Bias += (random.NextDouble() * 2 - 1) * learningRate;
    }
}
