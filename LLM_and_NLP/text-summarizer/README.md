# Text Summarizer Script

This Python script performs text summarization by extracting the most relevant sentences from a given text file. It leverages natural language processing (NLP) techniques, utilizing the NLTK library for stopwords and sentence similarity, NumPy for numerical operations, and NetworkX for graph theory applications.

## Setup and Dependencies

The script requires Python 3 and the following libraries:
- NLTK (Natural Language Toolkit)
- NumPy
- NetworkX
- Regular expressions (re module, part of the Python Standard Library)

Upon first run, the script will automatically download the necessary NLTK stopwords dataset.

## Key Components

### Initial Setup

Imports necessary libraries and downloads NLTK stopwords quietly at the beginning to avoid cluttering the output.

### `read_article(file_name)`

- **Objective**: Reads an article from a file, splitting it into sentences and words for further processing.
- **Process**: Opens the file, reads its content, splits the text into sentences, cleans each sentence to remove non-alphabetic characters, and finally splits them into words.
- **Error Handling**: Includes a try-except block to catch `FileNotFoundError`.

### `sentence_similarity(sent1, sent2, stopwords=None)`

- **Objective**: Calculates the similarity between two sentences based on their word vectors, excluding stopwords.
- **Process**: Transforms sentences to lowercase, filters out stopwords, then compares vectors for each sentence based on word frequency. Uses cosine distance for similarity measurement.

### `build_similarity_matrix(sentences, stop_words)`

- **Objective**: Creates a similarity matrix for a list of sentences using the stopwords to exclude common words.
- **Process**: Initializes a matrix of zeros and fills it with similarity scores for each pair of sentences.

### `generate_summary(file_name, top_n=5)`

- **Objective**: Generates a summary of an article by selecting the top N sentences based on centrality to the article's content.
- **Process**:
    - Reads the article and splits it into sentences.
    - Builds a similarity matrix.
    - Converts the similarity matrix into a graph and applies the PageRank algorithm.
    - Sorts sentences based on PageRank scores and selects the top N sentences for the summary.

## Summary

This script is an example of combining NLP, numerical analysis, and graph theory to solve a text summarization problem, illustrating an interdisciplinary approach to AI and data science challenges.