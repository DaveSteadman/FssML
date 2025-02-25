using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

public class TokenVocab
{
    // The internal dictionary mapping tokens to their indices.
    public Dictionary<string, int> VocabDictionary { get; private set; } = new Dictionary<string, int>();

    public int Count { get { return VocabDictionary.Count; } }

    // --------------------------------------------------------------------------------------------
    // MARK: Constructors and init
    // --------------------------------------------------------------------------------------------
    // Constructor: either wrap an existing dictionary or start with an empty one.
    public TokenVocab(Dictionary<string, int>? inheritDict = null)
    {
        // Copy across the dictionary entries if provided (at no point do we share the dictionary object across vocabs)
        if (inheritDict != null)
        {
            // loop across and add entries
            foreach (var entry in inheritDict)
            {
                VocabDictionary[entry.Key] = entry.Value;
            }
        }
        // Else, start a new dictionary from scratch
        else
        {
            AddInitialTokens();
        }
    }

    // Private helper: add special tokens and a set of default characters.
    private void AddInitialTokens()
    {
        // Special tokens for padding, unknown, and end-of-sequence.
        VocabDictionary["<UNK>"] = VocabDictionary.Count;
        VocabDictionary["<PAD>"] = VocabDictionary.Count;
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
    // MARK: Deep Copy
    // --------------------------------------------------------------------------------------------

    public TokenVocab DeepCopy()
    {
        return new TokenVocab(VocabDictionary);
    }

    // --------------------------------------------------------------------------------------------
    // MARK: New Vocab
    // --------------------------------------------------------------------------------------------

    // Take a string, tokenise it, and BPE the IDs of common pairs into new tokens in the vocab.

    public void ApplyBPEIteration(string inputText, int mergesCount = 20)
    {
        // tokenise the input to token IDs
        List<int> idList = TokenizeToIds(inputText, true);

        // Create the dictionary for storing the number of occurences of adjacent pairs
        Dictionary<(int, int), int> pairFreq = new Dictionary<(int, int), int>();

        // Loop through the token IDs, counting the frequency of adjacent pairs
        for (int i = 0; i < idList.Count - 1; i++)
        {
            int currTok = idList[i];
            int nextTok = idList[i + 1];

            // filter out any returned <UNK> tokens
            if (currTok == 0 || nextTok == 0)
                continue;

            var pair = (currTok, nextTok);

            if (!pairFreq.ContainsKey(pair))
                pairFreq[pair] = 0;
            pairFreq[pair]++;
        }

        // No pairs to merge (robustness clause).
        if (pairFreq.Count == 0) return;

        // Select the top 'mergesCount' most frequent pairs.
        var candidatePairs = pairFreq.OrderByDescending(p => p.Value)
                                     .Take(mergesCount)
                                     .Select(p => p.Key)
                                     .ToHashSet();

        // Loop through the candidate pairs, merging them into new tokens
        foreach (var pair in candidatePairs)
        {
            // Create a merged token by concatenating the pair.
            string mergedToken = GetTokenString(pair.Item1) + GetTokenString(pair.Item2);

            if (!VocabDictionary.ContainsKey(mergedToken))
                VocabDictionary[mergedToken] = VocabDictionary.Count;
        }
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Query
    // --------------------------------------------------------------------------------------------

    // Returns the index for a given token, or throws an exception if not found.
    public int GetTokenId(string token)
    {
        if (VocabDictionary.TryGetValue(token, out int index))
            return index;

        // If not found, return 0 of the <UNK> token.
        return 0;
    }

    public string GetTokenString(int index)
    {
        foreach (var pair in VocabDictionary)
        {
            if (pair.Value == index)
                return pair.Key;
        }
        return "<UNK>";
    }

    // --------------------------------------------------------------------------------------------

    public List<string> TokenizeToStrings(string input, bool addUnknownCharacters = false)
    {
        // Return a list of token strings.
        List<string> tokens = new List<string>();

        // Determine the maximum token length in the vocabulary.
        // (If the vocab grows dynamically, you might want to recompute this more frequently.)
        int maxTokenLength = VocabDictionary.Keys.Max(token => token.Length);

        int pos = 0;
        while (pos < input.Length)
        {
            bool matched = false;

            // Try to match the longest token in the vocabulary.
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

            // If no match is found...
            if (!matched)
            {
                // In training mode, add the unknown character as a new token.
                if (addUnknownCharacters)
                {
                    string newToken = input[pos].ToString();
                    if (!VocabDictionary.ContainsKey(newToken))
                    {
                        VocabDictionary[newToken] = VocabDictionary.Count;
                    }
                    tokens.Add(newToken);
                }
                else
                {
                    // In inference mode, simply add the <UNK> token.
                    tokens.Add("<UNK>");
                }
                pos++;
            }
        }
        return tokens;
    }

    // Obtain a list of token IDs. TokenizeToStrings does the heavy lifting, then we convert to IDs.
    public List<int> TokenizeToIds(string input, bool addUnknownCharacters = false)
    {
        List<string> strList = TokenizeToStrings(input, addUnknownCharacters);
        List<int>    idList  = new ();

        foreach (string str in strList)
            idList.Add(GetTokenId(str));

        return idList;
    }

    // Output the IDs and strings of the tokens in the input list, largely for debug purposes.
    // Usage: Console.WriteLine(vocab.DebugTokenList(tokenIds));
    public string DebugTokenList(List<int> tokenIds)
    {
        StringBuilder sb = new StringBuilder();
        List<string> tokList = new List<string>();

        foreach(int id in tokenIds)
            tokList.Add(GetTokenString(id));

        if (tokList.Count != tokenIds.Count)
            throw new Exception("Token list and ID list are not the same length.");

        int len = tokList.Count;
        for (int i = 0; i < len; i++)
        {
            sb.Append($"[{tokList[i]}: {tokenIds[i]}] ");
        }
        return sb.ToString();
    }

    // --------------------------------------------------------------------------------------------
    // MARK: Serialization
    // --------------------------------------------------------------------------------------------

    public void SaveToFile(string filepath)
    {
        using (var writer = new StreamWriter(filepath, false, Encoding.UTF8))
        {
            // loop through each value in the dictionary, writing it out.
            foreach (var pair in VocabDictionary)
            {
                // Note that the value may contain whitespace and special characters
                // so we need to escape it.
                writer.WriteLine($"\"{pair.Value}\" \"{pair.Key}\"");
            }
        }
    }

    public static TokenVocab LoadFromFile(string filePath)
    {
        var vocab = new TokenVocab();
        using (var reader = new StreamReader(filePath, Encoding.UTF8))
        {
            while (!reader.EndOfStream)
            {
                string? line = reader.ReadLine();

                if (line == null)
                {
                    Console.WriteLine("Warning: Empty line in vocabulary file.");
                    continue;
                }




                // Step 1: Find the initial quoted number and comma.
                var initialMatch = Regex.Match(line, "^\\s*\"(\\d+)\"\\s*(?:,\\s*|\\s+)");
                if (!initialMatch.Success)
                {
                    throw new Exception($"Invalid line (missing initial quoted number and comma): {line}");
                }
                int tokenId = int.Parse(initialMatch.Groups[1].Value);

                // Remove the matched part from the line.
                string remainder = line.Substring(initialMatch.Length);

                // Step 2: Extract the rest of the line as a quoted string.
                // This regex assumes the rest of the line is just a quoted string with anything inside.
                var textMatch = Regex.Match(remainder, "^\\s*\"(.*)\"\\s*$");
                if (!textMatch.Success)
                {
                    throw new Exception($"Invalid line (missing encapsulated quoted text): {line}");
                }
                string tokenText = textMatch.Groups[1].Value;

                // Now you have tokenId and tokenText regardless of any internal characters.
                //Console.WriteLine($"Token ID: {tokenId}, Token Text: {tokenText}");


                vocab.VocabDictionary[tokenText] = tokenId;




                // //var matches = Regex.Matches(line, "\"([^\"]*)\"");
                // var matches = Regex.Matches(line, "\"((?:[^\"]|\"\")*)\"");
                // if (matches.Count != 2)
                // {
                //     string matchesStr = String.Join("|", matches);
                //     throw new Exception($"Invalid line in vocabulary file: {line} // {matchesStr} // {matches.Count}");
                // }

                // string valueStr = matches[0].Groups[1].Value;
                // string key      = matches[1].Groups[1].Value;

                // if (!String.IsNullOrEmpty(key) && !String.IsNullOrEmpty(valueStr))
                // {
                //     int tokenId = int.Parse(valueStr);
                //     vocab.VocabDictionary[key] = tokenId;
                // }
            }
        }
        return vocab;
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

    // Sweep through the vocabulary and renumber the tokens from 0 to N-1.
    public void RenumberVocabulary()
    {
        // Create a new dictionary to hold the renumbered tokens.
        var newVocab = new Dictionary<string, int>();

        // Order the tokens by their current ID.
        int newIndex = 0;
        foreach (var pair in VocabDictionary.OrderBy(kv => kv.Value))
        {
            newVocab[pair.Key] = newIndex;
            newIndex++;
        }

        // Replace the existing vocabulary with the renumbered version.
        VocabDictionary = newVocab;
    }

    // Perform a save/load round-trip to limit that culls duplicates or issues from control characters.
    // Then limits the size and resaves the file.
    public static void PerformLimitSizePass(string filePath, int size)
    {
        var newVocab = TokenVocab.LoadFromFile(filePath);
        newVocab.LimitSize(size);
        newVocab.RenumberVocabulary();
        newVocab.SaveToFile(filePath);
    }
}
