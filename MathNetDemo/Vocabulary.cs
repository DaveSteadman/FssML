using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

public class Vocabulary
{
    // The internal dictionary mapping tokens to their indices.
    public Dictionary<string, int> VocabDictionary { get; private set; }

    // --------------------------------------------------------------------------------------------
    // MARK: Constructors and init
    // --------------------------------------------------------------------------------------------
    // Constructor: either wrap an existing dictionary or start with an empty one.
    public Vocabulary(Dictionary<string, int> VocabDictionary = null)
    {
        VocabDictionary = VocabDictionary ?? new Dictionary<string, int>();

        // If the dictionary is empty, add some default tokens.
        if (VocabDictionary.Count == 0)
        {
            AddInitialTokens();
        }
    }

    // Private helper: add special tokens and a set of default characters.
    private void AddInitialTokens()
    {
        // Special tokens for padding, unknown, and end-of-sequence.
        VocabDictionary["<PAD>"] = VocabDictionary.Count;
        VocabDictionary["<UNK>"] = VocabDictionary.Count;
        VocabDictionary["<EOS>"] = VocabDictionary.Count;

        // Default characters: letters and digits.
        string defaultChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        foreach (char c in defaultChars)
        {
            string token = c.ToString();
            if (!VocabDictionary.ContainsKey(token))
            {
                VocabDictionary[token] = VocabDictionary.Count;
            }
        }
    }

    // --------------------------------------------------------------------------------------------
    // MARK: New Vocab
    // --------------------------------------------------------------------------------------------

    // Take a string, tokenise it, and BPE the IDs of common pairs into new tokens in the vocab.

    public void ApplyBPEIteration(string inputText, int mergesCount = 10)
    {
        // tokenise the input string
        List<string> tokens = Tokenize(inputText);

        // Count frequency of adjacent token pairs.
        Dictionary<(string, string), int> pairFreq = new Dictionary<(string, string), int>();
        for (int i = 0; i < tokens.Count - 1; i++)
        {
            var pair = (tokens[i], tokens[i + 1]);
            if (!pairFreq.ContainsKey(pair))
                pairFreq[pair] = 0;
            pairFreq[pair]++;
        }

        // No pairs to merge.
        if (pairFreq.Count == 0) return;

        // Select the top 'mergesCount' most frequent pairs.
        var candidatePairs = pairFreq.OrderByDescending(p => p.Value)
                                     .Take(mergesCount)
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
                    if (!VocabDictionary.ContainsKey(mergedToken))
                    {
                        VocabDictionary[mergedToken] = VocabDictionary.Count;
                    }
                    newTokens.Add(mergedToken);
                    iIndex += 2; // Skip the next token.
                    continue;
                }
            }
            newTokens.Add(tokens[iIndex]);
            iIndex++;
        }
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Query
    // --------------------------------------------------------------------------------------------

    // Returns the index for a given token, or throws an exception if not found.
    public int GetTokenIndex(string token)
    {
        if (VocabDictionary.TryGetValue(token, out int index))
            return index;
        throw new ArgumentException($"Token '{token}' not found in vocabulary.");
    }

    // --------------------------------------------------------------------------------------------

    public List<string> Tokenize(string input)
    {
        List<string> tokens = new List<string>();
        int pos = 0;
        // Determine the maximum token length in the vocabulary.
        int maxTokenLength = VocabDictionary.Keys.Max(token => token.Length);

        while (pos < input.Length)
        {
            bool matched = false;
            // Try to match the longest token first.
            for (int len = Math.Min(maxTokenLength, input.Length - pos); len > 0; len--)
            {
                string substr = input.Substring(pos, len);
                if (VocabDictionary.ContainsKey(substr))
                {
                    tokens.Add(substr);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                // If no match is found, add the single character.
                tokens.Add(input[pos].ToString());
                pos++;
            }
        }
        return tokens;
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    // Returns a JSON string representing the vocabulary.
    public string ToJson()
    {
        // Use indented formatting for readability.
        var options = new JsonSerializerOptions { WriteIndented = true };
        return JsonSerializer.Serialize(VocabDictionary, options);
    }

    // Saves the vocabulary to a file at the specified path.
    public void SaveToFile(string filePath)
    {
        File.WriteAllText(filePath, ToJson());
    }

    // Static method to load a Vocabulary from a JSON file.
    public static Vocabulary LoadFromFile(string filePath)
    {
        string json = File.ReadAllText(filePath);
        var dict = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
        if (dict == null)
            throw new Exception("Failed to deserialize vocabulary.");
        return new Vocabulary(dict);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Management
    // --------------------------------------------------------------------------------------------

    // Optionally, limit the vocabulary size (e.g., keep only the most frequent tokens).
    public void LimitSize(int maxSize)
    {
        // Assuming that lower index values imply higher frequency (if built that way).
        var limited = VocabDictionary.OrderBy(pair => pair.Value)
                                  .Take(maxSize)
                                  .ToDictionary(pair => pair.Key, pair => pair.Value);
        VocabDictionary = limited;
    }

    // Perform a save/load round-trip to limit that culls duplicates or issues from control characters.
    // Then limits the size and resaves the file.
    public static void PerformLimitSizePass(Vocabulary vocab, string filePath, int size)
    {
        vocab.SaveToFile(filePath);
        var newVocab = Vocabulary.LoadFromFile(filePath);
        newVocab.LimitSize(size);
        newVocab.SaveToFile(filePath);
    }
}
