import pickle as pk
import sys
import clustering as cl
import math as mt

inverse_df = []

def calculate_coverage(S, V, idx_to_token):
    total_coverage = 0
    global inverse_df

    for i in V:
        for s in S:
            total_coverage += cl.calculate_similarity(V[i], s, inverse_df, idx_to_token)

    return total_coverage

def calculate_diversity(S, V, P, idx_to_token):
    total_diversity = 0
    num_docs = len(V)
    global inverse_df

    for k in range(len(P)):
        r = 0
        J = [sent for sent in S if sent in P[k]]
        for j in J:
            for i in V:
                r += cl.calculate_similarity(V[i], j, inverse_df, idx_to_token)
        r /= num_docs
        total_diversity += mt.sqrt(r)

    return total_diversity

def calculate_relevance(sentence_vector, idf, tokenAtIndex):
    relevance = 0
    for key in sentence_vector:
        word = tokenAtIndex[key]
        relevance += sentence_vector[key] * idf[word]
    return relevance

def compute_summary_quality(S, V, new_sent, P, idx_to_token, lambda_val, beta_val):
    S_copy = S.copy()

    if new_sent is not None:
        S_copy.append(new_sent)

    if len(S_copy) == 0:
        relevance_score = 0
    else:
        relevance_score = sum([calculate_relevance(s, inverse_df, idx_to_token) for s in S_copy]) / len(S_copy)


    return (calculate_coverage(S_copy, V, idx_to_token) + 
            lambda_val * calculate_diversity(S_copy, V, P, idx_to_token) +
            beta_val * relevance_score)


# def compute_summary_quality(S, V, new_sent, P, idx_to_token, lambda_val):
#     S_copy = S.copy()

#     if new_sent is not None:
#         S_copy.append(new_sent)

#     return (calculate_coverage(S_copy, V, idx_to_token) + 
#             lambda_val * calculate_diversity(S_copy, V, P, idx_to_token))

def init(idf_file):
    global inverse_df
    inverse_df = pk.load(open(idf_file, "rb"))
