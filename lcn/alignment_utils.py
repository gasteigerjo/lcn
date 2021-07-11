import collections
import gzip
import io
import logging
import os
import pickle

import numpy as np
import torch
from tqdm.autonotebook import tqdm


def load_vectors(fname, maxload=200000, norm=True, center=False):
    logging.debug(f"Loading vectors from '{fname}'")
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(" ")
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    logging.debug(f"{len(words)} word vectors loaded")
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, "w", encoding="utf-8")
    fout.write("%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def save_matrix(fname, x):
    n, d = x.shape
    fout = io.open(fname, "w", encoding="utf-8")
    fout.write("%d %d\n" % (n, d))
    for i in range(n):
        fout.write(" ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def procrustes(X_src, Y_tgt):
    U, s, V = torch.svd(Y_tgt.T @ X_src, some=False)
    return U @ V.T


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = np.zeros([n, d])
    y = np.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y


def load_lexicon(filename, words_src, words_tgt):
    f = io.open(filename, "r", encoding="utf-8")
    lexicon = collections.defaultdict(set)
    idx_src, idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if logging.root.level <= logging.DEBUG:
        coverage = len(lexicon) / float(len(vocab))
        logging.debug(f"Coverage of source vocab: {coverage:.4f}")
    return lexicon, float(len(vocab))


def load_pairs(filename, idx_src, idx_tgt):
    f = io.open(filename, "r", encoding="utf-8")
    pairs = []
    tot = 0
    for line in f:
        a, b = line.rstrip().split(" ")
        tot += 1
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if logging.root.level <= logging.DEBUG:
        coverage = (1.0 * len(pairs)) / tot
        logging.debug(
            f"Found pairs for training: {len(pairs)} "
            f"- Total pairs in file: {tot} "
            f"- Coverage of pairs: {coverage:.4f}"
        )
    return pairs


def compute_nn(x_src, y_tgt, idx_src):
    x_src /= torch.norm(x_src, dim=1, keepdim=True) + 1e-8
    y_tgt /= torch.norm(y_tgt, dim=1, keepdim=True) + 1e-8

    scores = y_tgt @ x_src[idx_src].T
    pred = scores.argmax(dim=0)

    return pred


def compute_csls(x_src, y_tgt, idx_src, k=10, sc_batch_size=1024, sim_batch_size=5000):
    x_src /= torch.norm(x_src, dim=1, keepdim=True) + 1e-8
    y_tgt /= torch.norm(y_tgt, dim=1, keepdim=True) + 1e-8

    sc2 = y_tgt.new_zeros(y_tgt.shape[0])
    for i in range(0, y_tgt.shape[0], sc_batch_size):
        j = min(i + sc_batch_size, y_tgt.shape[0])
        sc_batch = y_tgt[i:j, :] @ x_src.T
        dotprod = torch.topk(sc_batch, k=k, dim=1, sorted=False).values
        sc2[i:j] = torch.mean(dotprod, dim=1)
        del sc_batch

    pred = idx_src.new_zeros(idx_src.shape[0])
    for i in range(0, idx_src.shape[0], sim_batch_size):
        j = min(i + sim_batch_size, idx_src.shape[0])
        similarities = 2 * x_src[idx_src[i:j]] @ y_tgt.T
        similarities -= sc2[None, :]
        pred[i:j] = torch.argmax(similarities, dim=1).to(x_src.device)
        del similarities

    return pred


def compute_accuracy(x_src, y_tgt, lexicon, retrieval_fn, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = torch.tensor(list(lexicon.keys()), dtype=torch.long).to(x_src.device)

    pred = retrieval_fn(x_src, y_tgt, idx_src)

    acc = 0.0
    for idx_pred, idx_lex in enumerate(idx_src):
        if pred[idx_pred].item() in lexicon[idx_lex.item()]:
            acc += 1.0
    return acc / lexicon_size


def refine(x_src, y_tgt, lexicon, lexicon_size=-1, nepochs=5, disable_tqdm=False):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    idx_src = torch.tensor(idx_src, dtype=torch.long).to(x_src.device)
    idx_tgt = [item for sublist in lexicon.values() for item in sublist]
    idx_tgt = torch.tensor(idx_tgt, dtype=torch.long).to(y_tgt.device)

    for i in tqdm(range(nepochs), disable=disable_tqdm):
        pred_src = compute_csls(x_src, y_tgt, idx_src)
        pred_tgt = compute_csls(y_tgt, x_src, idx_tgt)

        idx_mutual_src = torch.cat((idx_src, pred_tgt))
        idx_mutual_tgt = torch.cat((pred_src, idx_tgt))

        R = procrustes(x_src[idx_mutual_src], y_tgt[idx_mutual_tgt])
        y_tgt = y_tgt @ R

    return y_tgt


def pickle_cache(filename, fn, fn_args=[], fn_kwargs={}, compression=False):
    if os.path.isfile(filename):
        if compression:
            with gzip.open(filename, "rb") as f:
                obj = pickle.load(f)
        else:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
    else:
        obj = fn(*fn_args, **fn_kwargs)
        if compression:
            with gzip.open(filename, "wb") as f:
                pickle.dump(obj, f)
        else:
            with open(filename, "wb") as f:
                pickle.dump(obj, f)
    return obj


def sqrt_eig(x):
    U, s, V = torch.svd(x, some=True)
    return U @ torch.diag_embed(torch.sqrt(s)) @ V.T
