import os
import sys
import pickle as pk
from nltk import word_tokenize
from nltk.corpus import stopwords
from math import log10
import string
from stemming.porter2 import stem

inverse_df = {}
doc_count = 0

def retrieve_files(directory):
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for i in range(len(file_list)):
        file_list[i] = os.path.join(directory, file_list[i])
    return file_list

def initialize_idf(file_list):
    global doc_count
    global inverse_df
    num_docs = len(file_list)
    print("Total documents in directory:", num_docs)
    for file in file_list:
        doc_count += 1
        print("Processing document %d of %d" % (doc_count, num_docs))
        with open(file, 'r') as f:
            doc = f.read()
        doc = word_tokenize(doc)
        stop_words = stopwords.words('english')
        doc = [token for token in doc if token not in stop_words and token not in set(string.punctuation)]
        for i in range(len(doc)):
            doc[i] = stem(doc[i])
        unique_tokens = []
        for token in doc:
            if token not in unique_tokens:
                if token in inverse_df:
                    inverse_df[token] += 1
                else:
                    inverse_df[token] = 1
                unique_tokens.append(token)

def main():
    directory = sys.argv[1]
    idf_output = "idf.out"
    file_list = retrieve_files(directory)
    initialize_idf(file_list)
    for term in inverse_df:
        inverse_df[term] = log10(float(doc_count) / (1.0 + float(inverse_df[term])))

    pk.dump(inverse_df, open(idf_output, "wb+"))

main()
