import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv(path = 'co_occurrence.csv'):
    data = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            a = []
            for el in row:
                a.append(float(el))
            a = np.array(a)
            data.append(a)
        data = np.array(data)
        print(np.shape(data))
    return data

def read_txt(path):
    with open(path) as f:
        data = f.read()
        words = data.split("\n")
        print(len(words))
    return words

def plot_evs(D):
    n = np.shape(D)[0]
    x = range(1, n+1)
    fig, ax = plt.subplots()
    ax.plot(x, D)
    ax.set(xlabel = 'Index', ylabel = 'Eigenvalue')
    fig.savefig("ev_plot.png")
    plt.show()

def find_embedding(word, words, E):
    index = 0
    for i in range(len(words)):
        if words[i] == word:
            index = i
            break
    emb = E[index] 
    return emb

def find_most_sim_word(word, words, E):
    emb = find_embedding(word, words, E)
    similarity = {}
    for idx, i in enumerate(E):
        if np.array_equal(i, emb): continue
        i_normalized = i / np.linalg.norm(i) 
        similarity[idx] = np.dot(emb, i_normalized)
    if not similarity: 
        return None
    max_sim_index = max(similarity, key=similarity.get)
    most_sim_word = words[max_sim_index]
    return most_sim_word

def find_info_ev(v, words, k=20):
    v_abs = np.abs(v)
    indices = np.argsort(v_abs)[::-1][:k] 
    info = words[indices]
    return info

def plot_projections(word_seq, proj_seq, filename = 'projections.png'):
    y = np.zeros(len(proj_seq))
    fig, ax = plt.subplots()
    ax.scatter(proj_seq, y)
    plt.rcParams.update({'font.size': 9})
    proj_seq = np.array(proj_seq)
    idx = np.argsort(proj_seq)
    for i in range(len(idx)):
        idx[i] = int(idx[i])
    proj_seq = proj_seq[idx]
    word_seq = word_seq[idx]
    del_y_prev = 0
    for i, label in enumerate(word_seq):
        if i<len(proj_seq)-1:
            if np.abs(proj_seq[i]-proj_seq[i+1])<0.02:
                del_y0 = -0.005-0.0021*len(label)
                if del_y_prev == 0:
                    del_y = del_y0
                else:
                    del_y = 0
                del_y_prev = del_y0
            elif (del_y_prev!=0 and del_y == 0 and np.abs(proj_seq[i]-proj_seq[i-1])<0.02):
                del_y = -0.005-0.0021*len(label)
                del_y_prev = del_y
            else:
                del_y = 0
                del_y_prev = 0
        ax.text(x = proj_seq[i]-0.01, y = y[i]+0.005+del_y, s = label, rotation = 90)
    ax.set_xlim(-0.5,0.5)
    fig.savefig(filename)
    plt.show()

def get_projections(word_seq, words, E, w):
    w = w / np.linalg.norm(w)
    proj_seq = []
    for word in word_seq:
        emb = find_embedding(word, words, E)
        proj = np.dot(emb, w)
        proj_seq.append(proj)
    proj_seq = np.array(proj_seq)
    return proj_seq

def comparative_projections(word_seq, words, E, wd1, wd2):
    w1 = find_embedding(wd1, words, E)
    w2 = find_embedding(wd2, words, E)
    w = w1 - w2
    w = w / np.linalg.norm(w)
    proj_seq = []
    for word in word_seq:
        emb = find_embedding(word, words, E)
        proj = np.dot(emb, w)
        proj_seq.append(proj)
    proj_seq = np.array(proj_seq)
    return proj_seq

def find_most_sim_word_w(w, words, E, ids):
    similarity = []
    valid_indices = []
    for ind, e in enumerate(E):
        if np.array_equal(e, w) or ind in ids: continue
        similarity.append(np.dot(w, e))
        valid_indices.append(ind) 
    if not similarity: return None
    max_sim_index = np.argmax(similarity)
    most_sim_word = words[valid_indices[max_sim_index]]
    return most_sim_word

def find_analog(wd1, wd2, wd3, words, E):
    w1 = find_embedding(wd1, words, E)
    w2 = find_embedding(wd2, words, E)
    w3 = find_embedding(wd3, words, E)
    w = w2 - w1 + w3
    w = w / np.linalg.norm(w)
    ids = []
    for i in range(len(words)):
        if words[i] == wd1 or words[i] == wd2 or words[i] == wd3:ids.append(i)
    wd4 = find_most_sim_word_w(w, words, E, ids)
    return wd4

def check_analogy_task_acc(task_words, words, E):
    acc = 0
    for word_seq in task_words:
        word_seq = word_seq.split()
        wd4_ans = find_analog(word_seq[0], word_seq[1], word_seq[2], words, E)
        if wd4_ans == word_seq[3]:
            acc+=1
    acc = acc/len(task_words)
    return acc

def similarity(wd1, wd2, words, E):
    w1 = find_embedding(wd1, words, E)
    w2 = find_embedding(wd2, words, E)
    similarity = np.dot(w1, w2)
    return similarity

def check_similarity_task_acc(task_words, words, E):
    acc = 0
    for word_seq in task_words:
        word_seq = word_seq.split()
        if similarity(word_seq[0],word_seq[1],words,E) > similarity(word_seq[0],word_seq[2],words,E):acc+=1
    acc = acc/len(task_words)
    return acc