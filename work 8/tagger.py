import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    unique_words = get_unique_words(train_data)
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    for i, word in enumerate(unique_words.keys()):word2idx[word] = i
    for i, tag in enumerate(tags):tag2idx[tag] = i
    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    start_tag_counts = {}
    tag_counts = {}
    transition_counts = {}
    emission_counts = {}
    for line in train_data:
        words = line.words
        pos_tags = line.tags
        first_tag = pos_tags[0]
        if first_tag in start_tag_counts:start_tag_counts[first_tag] += 1
        else:start_tag_counts[first_tag] = 1
        for i in range(len(pos_tags)):
            current_tag = pos_tags[i]
            if current_tag in tag_counts:tag_counts[current_tag] += 1
            else:tag_counts[current_tag] = 1
            if words[i] in word2idx:
                word = words[i]
                key = (current_tag, word)
                if key in emission_counts:emission_counts[key] += 1
                else:emission_counts[key] = 1
            if i < len(pos_tags) - 1:
                next_tag = pos_tags[i+1]
                key = (current_tag, next_tag)
                if key in transition_counts:transition_counts[key] += 1
                else:transition_counts[key] = 1
    for i in range(S):
        tag = tags[i]
        if tag in start_tag_counts:pi[i] = start_tag_counts[tag] / len(train_data)
        else:pi[i] = 0
    for i in range(S):
        current_tag = tags[i]
        if current_tag in tag_counts and tag_counts[current_tag] > 0:
            total = tag_counts[current_tag]
            for j in range(S):
                next_tag = tags[j]
                key = (current_tag, next_tag)
                if key in transition_counts:A[i, j] = transition_counts[key] / total
                else:A[i, j] = 0
        else:A[i, :] = 0
    for i in range(S):
        tag = tags[i]
        if tag in tag_counts and tag_counts[tag] > 0:
            total = tag_counts[tag]
            for word, word_idx in word2idx.items():
                key = (tag, word)
                if key in emission_counts:B[i, word_idx] = emission_counts[key] / total
                else:B[i, word_idx] = 0
        else:B[i, :] = 0

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model

def sentence_tagging(test_data, model, tags):
    tagging = []
    for line in test_data:
        words = line.words
        for word in words:
            if word not in model.obs_dict:
                new_word_idx = len(model.obs_dict)
                model.obs_dict[word] = new_word_idx
                new_B = np.zeros((model.B.shape[0], model.B.shape[1] + 1))
                new_B[:, :-1] = model.B
                new_B[:, -1] = 1e-6
                model.B = new_B
        sentence_tags = model.viterbi(words)
        tagging.append(sentence_tags)
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):
    unique_words = {}
    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq
    return unique_words
