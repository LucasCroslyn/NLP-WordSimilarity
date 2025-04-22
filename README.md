# Natural Language Programming: Word Similarity

This project was to develop a few different models to calculate a (cosine) similarity score (between 0-10) between words. The different models are:
- Baseline: Always use the value of 5.
- Term-Document Matrix: Use the number of times a word shows up in the different training documents to form word vectors.
- Term-Term Window Matrix: Use the number of times words show up around other words to form word vectors.
- Word2Vec: Use a pre-defined and pre-trained model to convert words to vectors.

The different files in this project are:
- The [Data](Data/) folder contains all of the training and testing data.
- The [sim.py]() file contains the functions to make, train, and test the different models.
- The [score.py]() file contains the function to calculate the error in the results from the models.
- The [main.ipynb]() file shows the code being run in a Jupyter Notebook with the results from the different models.
