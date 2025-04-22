import math, re
import gensim.downloader

from collections import Counter, defaultdict


def tokenize(s):
    '''
    A simple word tokenizer. Applies case folding. Removes leading and trailing non-alphanumeric characters including punctuation for each word (token) in the text. Keeps any other non-alphanumeric characters within the word (token).

    :param s: The input string to tokenize into separate words (tokens).
    :return: Returns a list of all the words (tokens) from the input string. 
    '''
    
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search(r'\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub(r'^\W*', '', t)  # trim leading non-alphanumeric chars
            t = re.sub(r'\W*$', '', t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


def dotprod(vector1, vector2):
    '''
    Calculates the dot product between two vectors (lists).

    :param vector1: The first vector (list format).
    :param vector2: The second vector (list format).
    :return: Return the calculated dot product between the two input vectors (lists).
    '''
    return sum([x * y for x, y in zip(vector1, vector2)])


class Baseline():
    '''
    An extremely basic word similarity model. Always predict the same similarity each time (value of 5 picked here as it is the middle value, possible similarities are 0-10).
    '''
    def __init__(self):
        '''
        Initializes the model, doesn't need to train on anything.
        '''
        self.pred_sim = 5

    def calc_sim(self, word_1, word_2):
        '''
        Since the similarity score will always be the same, doesn't actually need to do anything.

        :param word_1: The first word, isn't used but here for nicer code later.
        :param word_2: The second word, isn't used but here for nicer code later.
        :return: Returns the predicted similarity (in this case always 5).
        '''
        return self.pred_sim


# using the count-vector from a term-document matrix
class Term_document():
    '''
    Utilize a term-document matrix model with the counts of each token per document to calculate cosine similarity between two tokens.
    '''
    
    def __init__(self, corpus):
        '''
        Initilizes the model and trains it given the training texts.

        :param corpus: The training texts file. One document sample per line.
        '''
        self.embeddings: dict[str, list[int]] = {} # The term-document matrix, each column is a different document, rows are tokens
        self.train(corpus)

    
    def train(self, corpus):
        '''
        Calculates the model's term-document matrix on all of the tokens/documents.

        :param corpus: The training texts file. One document sample per line.
        '''

        # Tokenizes the inputs documents
        documents_text = [tokenize(x) for x in open(corpus, encoding='utf8')]
        for i, text in enumerate(documents_text):
            for token in text:
                # If the token has not been seen before,
                # make its vector all zeroes aside from the document it just saw the token in which is now 1.
                # Due to each vector needing to be same size
                if token not in self.embeddings:
                    self.embeddings[token] = [0] * len(documents_text)
                    self.embeddings[token][i] = 1
                # If the token has been seen before, increment the token's count for the current document
                else:
                    self.embeddings[token][i] += 1

    
    def calc_sim(self, word_1, word_2):
        '''
        Calcuates the cosine similarity score between two different tokens based on the training data.

        :param word_1: The first token used in the calculation.
        :param word_2: The second token used in the calculation.
        :return: The similarity score between the two words in the range 0-10 rounded to two decimals. 
        '''
        
        # If either word was not seem in the training data, use a default similairty score.
        if word_1 not in self.embeddings or word_2 not in self.embeddings:
            return 5
        else:
            # Cosine similarity calculation
            dot: float = dotprod(self.embeddings[word_1], self.embeddings[word_2])
            lengthw1: float = math.sqrt(dotprod(self.embeddings[word_1], self.embeddings[word_1]))
            lengthw2: float = math.sqrt(dotprod(self.embeddings[word_2], self.embeddings[word_2]))
            
            pred_sim = dot / (lengthw1 * lengthw2)
            # Clamped result to 0, just in case
            if pred_sim < 0:
                pred_sim = 0
            # Range is initially 0-1, need to scale to range 0-10 and round to 2 decimals
            return round(pred_sim * 10, 2)


class Window():
    '''
    Utilizes a term-term matrix-esque model. Each token stores other tokens that are around it (a 'window' around the token) and how often they appear next to each other.
    The model uses this to calcuate a cosine similarity score between two words.
    '''
    
    def __init__(self, corpus, context_size=1):
        '''
        Initializes the model, including how big of a 'window' to use around tokens, and training the model on input data.

        :param corpus: The training texts file. One document sample per line.
        :param context_size: The size of the 'window' around each token to use to find tokens next to them.
        '''
        self.context_size = context_size
        self.embeddings: dict[str, Counter[str, int]] = defaultdict(Counter)
        self.train(corpus)

    
    def train(self, corpus):
        '''
        Calculates the model's term-term matrix on all of the tokens.

        :param corpus: The training texts file. One document sample per line.
        '''

        documents_text = [tokenize(x) for x in open(corpus, encoding='utf8')]

        for doc in documents_text:
            for i, target_token in enumerate(doc):
                # target_token = doc[i]
                # target_token_vector = self.embeddings.get(target_token, {})
                # target_token_vector = self.embeddings[target_token]
                
                for context in range(-self.context_size, self.context_size + 1):
                    # Makes sure the context window stays within the index of the document
                    if ((i + context) < 0) or ((i + context) > (len(doc) - 1)) or (context == 0):
                        continue
                    else:
                        context_word = doc[i + context]
                        # target_token_vector[context_word] += 1

                        # Increments the count for the context word appear in the window for the main, target word
                        self.embeddings[target_token][context_word] += 1

    
    def calc_sim(self, word_1, word_2):
        '''
        Calculates the cosine similarity score between two tokens based on the training data.

        :param word_1: The first token used in the calculation.
        :param word_2: The second token used in the calculation.
        :return: Returns the cosine similarity score calculated for the two input tokens
        '''

        # If either word was not seem in the training data, use a default similairty score.
        if word_1 not in self.embeddings or word_2 not in self.embeddings:
            return 5
        else:

            # Calculation for the cosine similarity.
            # Vectors are with the words appearing in the context window for the given words
            
            dot = 0
            # Only use context words that appeared with both words of the inputs
            # Otherwise, one of the context words would just have a zero count
            for key in self.embeddings[word_1]:
                if key in self.embeddings[word_2]:
                    dot += self.embeddings[word_1][key] * self.embeddings[word_2][key]

            lengthw1 = math.sqrt(dotprod(self.embeddings[word_1].values(), self.embeddings[word_1].values()))
            lengthw2 = math.sqrt(dotprod(self.embeddings[word_2].values(), self.embeddings[word_2].values()))

            pred_sim = dot / (lengthw1 * lengthw2)

            # Clamps the result to 0 just in case
            if pred_sim < 0:
                pred_sim = 0
            # Range is initially 0-1, need to scale to range 0-10 and round to 2 decimals
            return round(pred_sim * 10, 2)


class Word2Vec():
    '''
    This model just uses the pre-trained word vectorizor model 'word2vec-google-news-300' without more actual training
    '''
    
    def __init__(self):
        '''
        Initializes the model and technically 'trains' the model.
        '''
        self.vectors = None
        self.train()

   
    def train(self):
        '''
        The training here is just downloading the model's vectors without any more fine-tuning (which should maybe be done but isn't needed).
        '''
        self.vectors = gensim.downloader.load("word2vec-google-news-300")

    
    def calc_sim(self, word_1, word_2):
        '''
        Uses the pre-trained model to calculate the similarity between two words and scales the result to the range 0-10 rounded to two decimals.

        :param word_1: The first token to compare.
        :param word_2: The second token to compare against.
        :return: Returns the calculated similarity score between the two input tokens.
        '''
        return round(self.vectors.similarity(word_1, word_2) * 10, 2)