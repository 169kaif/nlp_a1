import collections

# Tokenizer class 
class Tokenizer:
    def __init__(self):
        # dictionary to store corpus words and their frequencies
        self.word_freq = {}
        # List to store merge rules during vocabulary learning
        self.merge_rules = []
        #set to store all vocabulary (including intermediate ones)
        self.vocab = set()

    # Calculate the count of pairs of symbols in the vocabulary
    def get_stats(self):

        """
        Dictionary to store pair frequencies
        Using the defaultdict from the collections library,
        we don't have to explicitly check if a key exists or not...
        
        if the key doesn't exist, then a key is made and it's value is incremented

        the key used here is a tuple of 2 consecutive characters and 
        the value associated with the key is the frequency of occurence 
        of the sequence (tuple_element 1)(tuple_element 2) 
        which is updated while iterating through the words in the vocabulary
        """

        pairs = collections.defaultdict(int)
        # Iterating over words in the vocabulary
        for word, freq in self.word_freq.items():
            symbols = word.split()
            # Iterating over pairs of symbols in each word
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    # Merging a pair of symbols in the vocabulary
    def merge(self, pair):
        # initalizing an empty dictionary to store new words after merge
        v_out = {}

        bigram = pair[0] + ' ' + pair[1]
        
        # Iterating over words in the current vocabulary
        for word in self.word_freq:
            # Apply the merge rule to each word

            w_out = word.replace(bigram, ''.join(pair))
            v_out[w_out] = self.word_freq[word]
        return v_out

    # Learning the vocabulary from the corpus using BPE
    def learn_vocabulary(self, corpus, num_merges):
        # Creating an empty dictionary to store the vocabulary
        word_freq = {}

        # Convert words in the corpus into appropriate format and count frequencies
        for word, freq in corpus.items():
            word_freq[' '.join(word) + ' $'] = freq

            #create vocabulary
            for char in word:
                self.vocab.add(char)

        # add end character to vocab
        self.vocab.add('$')

        self.word_freq = word_freq

        # Iterating over the number of merges
        for i in range(num_merges):

            # calculate pair frequencies
            pairs = self.get_stats()

            """
            finding most frequent pair

            key used for finding maximum is pairs.get
            which gives us the value associated with a certain dictionary key

            best will hence give us the 2-tuple of characters with
            the highest frequency of occurence together
            """
            best = max(pairs, key=pairs.get)

            # Merge the most frequent pair
            self.word_freq = self.merge(best)

            # Store the merge rule 
            self.merge_rules.append(best)

            # Add new token to the vocabulary
            self.vocab.add(best[0]+best[1])

    # tokenizing a sample using the learned vocabulary
    def tokenize(self, sample_corpus):

        #convert the sample corpus into appropriate format
        for i in range(len(sample_corpus)):

            #sample_corpus[i] refers to an individual sample
            for j in range(len(sample_corpus[i])):

                #sample_corpus[i][j] refers to words in the individual sample

                # check for blank spaces
                sample_corpus[i][j] = sample_corpus[i][j].strip()

                # convert word to appropriate format
                sample_corpus[i][j] = ' '.join(sample_corpus[i][j]) + ' $'

        print("converted the sample corpus into appropriate format")
        print(sample_corpus)

        # applying the learned rules
        for rule in self.merge_rules:

            for i in range(len(sample_corpus)):
                for j in range(len(sample_corpus[i])):
                    
                    """
                    say my rule is a,b

                    ' '.join(rule) will give me "a b" (as in separated by one space)
                    ''.join(rule) wil give me "ab" (as in separated by no spaces)

                    then word.replace() will replace the first argument with the second argument
                    """

                    # #print the rule being applied
                    # print(rule)

                    # #print word before applying rule
                    # print("before applying rule: " + sample_corpus[i][j])

                    sample_corpus[i][j] = sample_corpus[i][j].replace(' '.join(rule), ''.join(rule))

                    # #print word after applying rule
                    # print("after applying rule: " + sample_corpus[i][j])

        # print("applied merge rules")
        # print(sample_corpus)

        #init tokenized sample corpus to store results
        tokenized_sample_corpus = []

        for sample in sample_corpus:

            #init temp token list to store tokens for each sample
            temp_token_list = []

            for word in sample:

                #retrieving tokens from the word
                tokens = word.split()

                for token in tokens:
                    #handling tokens with $ in the end
                    if (token[-1] == '$'):
                        token = (token[:len(token)-1])

                    #handling empty tokens
                    if (len(token) > 0):
                        temp_token_list.append(token)
            
            #append temp_token_list to tokenized_sample_corpus
            tokenized_sample_corpus.append(temp_token_list)

        return tokenized_sample_corpus
    
    
    
    
"""
Open and read the input data from a file and
initialize words and their frequencies in a naive format
"""

#set file read path
file_read_path = "./test_small_corpus.txt"

#initialize corpus dictionary
corpus = collections.defaultdict(int)

with open(file_read_path, 'r') as f:
    temp_word_list = []
    for line in f:
        temp_word_list = line.strip().split()
        for word in temp_word_list:
            corpus[word] += 1

# instance of the Tokenizer class
tokenizer = Tokenizer()
# Learn vocabulary from the corpus
tokenizer.learn_vocabulary(corpus, 100)

# Write the learned vocabulary to a file
with open('tokens.txt', 'w') as f:
    for token in sorted(tokenizer.vocab, key=len):
        f.write(token + "\n")

# Write the learned merge rules to a file
with open('merge_rules.txt', 'w') as f:
    for rule in tokenizer.merge_rules:
        f.write(",".join(map(str, rule)) + "\n")
        
        
        
        
        
        
#EVAL PART 3 (tokenize sample text)

#feed file to tokenize into the tokenizer
#INPUT FORMAT: 1 sample per line
#OUPUT FORMAT: tokenized sample of line x of the input on line x of the output

#set file read path
file_to_tokenize_read_path = "."
file_to_tokenize_corpus = []

with open(file_to_tokenize_read_path, 'r') as f:
    for line in f:
        file_to_tokenize_corpus.append(line.strip().split())

tokenized_sample_corpus = tokenizer.tokenize(file_to_tokenize_corpus)

print(tokenized_sample_corpus)

#write tokenized samples in appropriate format to a file
file_write_path = ""

with open(file_write_path, 'w') as f:

    #write tokens for each sample in one line and then move to the next line for new sample
    for tokenized_sample in tokenized_sample_corpus:
        for index, token in enumerate(tokenized_sample):
            if (index == len(tokenized_sample)-1):
                f.write(token)
            else:
                f.write(token + ",")
        f.write("\n")