import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = dict()

    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name), 'r') as text_file:
            result[file_name]=str(text_file.read())

    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    result = []
    tokenized_words = nltk.tokenize.word_tokenize(document.lower())

    for word_token in tokenized_words:
        if word_token not in nltk.corpus.stopwords.words("english") and word_token not in string.punctuation:
            result.append(word_token)

    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    result={}
    documents_total=len(documents)

    for doc in documents.values():
        for word in doc:
            c=doc.count(word)
            result[word]=math.log(documents_total/c)

    return result


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_scores={}
    results=[]

    for filename, filecontent in files.items():
        tf_idf=0

        for word in query:
            if word in filecontent:
                tf_idf += filecontent.count(word) * idfs[word]

        if tf_idf != 0:
            files_scores[filename] = tf_idf

    for result, result_value in sorted(files_scores.items(), key=lambda x: x[1], reverse=True):
        results.append(result)

    return results[:n]



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    results=[]
    scores_result = {}
    for sentence, sentwords in sentences.items():
        score_res = 0

        for word in query:
            if word in sentwords:
                score_res += idfs[word]

        if score_res != 0:
            sentence_query_result=[]
            for q in query:
                sentence_query_result.append(sentwords.count(q))

            density_result = sum(sentence_query_result) / len(sentwords)
            scores_result[sentence] = (score_res, density_result)

    for result, result_value in sorted(scores_result.items(), key=lambda x:(x[1][0], x[1][1]), reverse=True):
        results.append(result)

    return results[:n]


if __name__ == "__main__":
    main()
