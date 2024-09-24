using System;

public class FssConnection
{
    public int    FromNodeIndex { get; private set; } // Index of the node this connection is from
    public int    ToNodeIndex   { get; private set; } // Index of the node this connection is to
    public double Weight        { get; private set; } // Weight of the connection
    public bool   IsActive      { get; private set; } // Whether the connection is active

    private static Random random = new Random();

    // Constructor
    public FssConnection(int fromNodeIndex, int toNodeIndex)
    {
        FromNodeIndex = fromNodeIndex;
        ToNodeIndex = toNodeIndex;
        Weight = random.NextDouble() * 2 - 1; // Initialize weight randomly between -1 and 1
        IsActive = true;
    }

    // Constructor for deserialization with a predefined weight
    public FssConnection(int fromNodeIndex, int toNodeIndex, double weight)
    {
        FromNodeIndex = fromNodeIndex;
        ToNodeIndex = toNodeIndex;
        Weight = weight;
        IsActive = true;
    }

    // Evolve the weight with small random adjustments
    public void EvolveWeight(double learningRate)
    {
        if (IsActive)
        {
            Weight += (random.NextDouble() * 2 - 1) * learningRate; // Adjust weight randomly
        }
    }

    // Disable the connection
    public void Disable()
    {
        IsActive = false;
    }

    // Enable the connection
    public void Enable()
    {
        IsActive = true;
    }
}
