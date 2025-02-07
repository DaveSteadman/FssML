using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;

public static class BPETokenizer
{
    public static Dictionary<string, int> BuildInitialVocabulary(string input)
    {
        Dictionary<string, int> vocab = new Dictionary<string, int>();
        foreach (char c in input)
        {
            string token = c.ToString();
            if (!vocab.ContainsKey(token))
            {
                vocab[token] = vocab.Count; // assign next available ID
            }
        }
        return vocab;
    }

    // --------------------------------------------------------------------------------------------

    public static string VocabularyToString(Dictionary<string, int> vocab, int count)
    {
        var random = new Random();

        StringBuilder sb = new StringBuilder();
        //foreach (var pair in vocab.Take(count))
        foreach (var pair in vocab.OrderBy(_ => random.Next()).Take(count))
        {
            sb.Append($"[{pair.Key} {pair.Value}] ");
        }
        return sb.ToString();
    }

    // --------------------------------------------------------------------------------------------

    public static List<string> Tokenize(string input)
    {
        List<string> tokens = new List<string>();
        foreach (char c in input)
        {
            tokens.Add(c.ToString());
        }
        return tokens;
    }

    // --------------------------------------------------------------------------------------------

    public static List<int> GetTokenIds(List<string> tokens, Dictionary<string, int> vocab)
    {
        return tokens.Select(token => vocab.ContainsKey(token) ? vocab[token] : -1).ToList();
    }

    // --------------------------------------------------------------------------------------------

    public static void PrintTokensAndIds(List<string> tokens, Dictionary<string, int> vocab)
    {
        Console.WriteLine("Tokens: " + string.Join(" ", tokens));
        Console.WriteLine("Token IDs: " + string.Join(" ", GetTokenIds(tokens, vocab)));
    }

    // --------------------------------------------------------------------------------------------

    public static (List<string> newTokens, Dictionary<string, int> newVocab) ApplyBPEIteration(List<string> tokens, Dictionary<string, int> vocab, int merges = 10)
    {
        // Count frequency of adjacent token pairs.
        Dictionary<(string, string), int> pairFreq = new Dictionary<(string, string), int>();
        for (int i = 0; i < tokens.Count - 1; i++)
        {
            var pair = (tokens[i], tokens[i + 1]);
            if (!pairFreq.ContainsKey(pair))
                pairFreq[pair] = 0;
            pairFreq[pair]++;
        }

        if (pairFreq.Count == 0)
        {
            // No pairs to merge.
            return (tokens, new Dictionary<string, int>(vocab));
        }

        // Select the top 'merges' most frequent pairs.
        var candidatePairs = pairFreq.OrderByDescending(p => p.Value)
                                     .Take(merges)
                                     .Select(p => p.Key)
                                     .ToHashSet();

        // Perform a single pass to merge any occurrence of a candidate pair.
        List<string> newTokens = new List<string>();
        int iIndex = 0;
        while (iIndex < tokens.Count)
        {
            if (iIndex < tokens.Count - 1)
            {
                var currentPair = (tokens[iIndex], tokens[iIndex + 1]);
                if (candidatePairs.Contains(currentPair))
                {
                    // Create a merged token by concatenating the pair.
                    string mergedToken = tokens[iIndex] + tokens[iIndex + 1];
                    // Update the vocabulary if necessary.
                    if (!vocab.ContainsKey(mergedToken))
                    {
                        vocab[mergedToken] = vocab.Count;
                    }
                    newTokens.Add(mergedToken);
                    iIndex += 2; // Skip the next token.
                    continue;
                }
            }
            newTokens.Add(tokens[iIndex]);
            iIndex++;
        }

        // Return the new token list along with a new copy of the vocabulary.
        return (newTokens, new Dictionary<string, int>(vocab));
    }

    // --------------------------------------------------------------------------------------------

    public static List<string> TokenizeUsingVocabulary(string input, Dictionary<string, int> vocab)
    {
        List<string> tokens = new List<string>();
        int pos = 0;
        // Determine the maximum token length in the vocabulary.
        int maxTokenLength = vocab.Keys.Max(token => token.Length);

        while (pos < input.Length)
        {
            bool matched = false;
            // Try to match the longest token first.
            for (int len = Math.Min(maxTokenLength, input.Length - pos); len > 0; len--)
            {
                string substr = input.Substring(pos, len);
                if (vocab.ContainsKey(substr))
                {
                    tokens.Add(substr);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                // If no match is found (should rarely happen), add the single character.
                tokens.Add(input[pos].ToString());
                pos++;
            }
        }
        return tokens;
    }

    // --------------------------------------------------------------------------------------------

    public static List<int> TokenIdsUsingVocabulary(string input, Dictionary<string, int> vocab)
    {
        return GetTokenIds(TokenizeUsingVocabulary(input, vocab), vocab);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Manage
    // --------------------------------------------------------------------------------------------

    public static Dictionary<string, int> LimitVocabSize(Dictionary<string, int> vocab, int maxSize)
    {
        // Sort the vocabulary by frequency.
        var sortedVocab = vocab.OrderBy(pair => pair.Value)
                       .ToDictionary(pair => pair.Key, pair => pair.Value);

        // Take the top 'maxSize' tokens.
        return sortedVocab.Take(maxSize).ToDictionary(pair => pair.Key, pair => pair.Value);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Load Save
    // --------------------------------------------------------------------------------------------
    // New methods to save and load the vocabulary.

    public static void SaveVocabularyToFile(Dictionary<string, int> vocab, string filePath)
    {
        // Create options for human-readable (indented) JSON.
        var options = new JsonSerializerOptions { WriteIndented = true };

        // Serialize the vocabulary to a JSON string.
        string json = JsonSerializer.Serialize(vocab, options);

        // Write the JSON string to the specified file.
        File.WriteAllText(filePath, json);
    }

    // --------------------------------------------------------------------------------------------

    public static Dictionary<string, int> LoadVocabularyFromFile(string filePath)
    {
        // Read the JSON string from the specified file.
        string json = File.ReadAllText(filePath);

        // Deserialize the JSON string back into a dictionary.
        Dictionary<string, int>? newDict = JsonSerializer.Deserialize<Dictionary<string, int>>(json);

        if (newDict == null)
            return new Dictionary<string, int>();

        return newDict;
    }
}

