import logging
import time

import numpy as np
import seml
import torch
from sacred import Experiment

from lcn.embedding_alignment import (
    convex_init,
    align,
    align_original,
)
from lcn.alignment_utils import (
    compute_accuracy,
    compute_csls,
    compute_nn,
    load_lexicon,
    load_vectors,
    pickle_cache,
    refine,
    save_vectors,
)

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
    language_src,
    language_tgt,
    ntrain=20000,
    lr=500.0,
    niter=300,
    lr_half_niter=100,
    print_niter=50,
    seed=1111,
    ninit=2500,
    sinkhorn_reg=0.05,
    method="full",
    nystrom=None,
    sparse=None,
    test=False,
    data_dir=".",
    output_src=None,
    output_tgt=None,
    device="cuda",
):
    """
    Main function for aligning word embeddings.

    Arguments
    ---------
    language_src:               Source language identifier (en, de, es, fr, ru)
    language_tgt:               Target language identifier (en, de, es, fr, ru)
    ntrain                      Number of word embeddings used for training
    lr                          Learning rate
    niter                       Number of training steps
    lr_half_niter               Number of steps after which lr is halved
    print_niter                 Interval for updating printed training loss
    seed                        Random seed
    ninit                       Number of embeddings used for convex initialization
    sinkhorn_reg                Entropy regularization used for Sinkhorn
    method                      Name of the Sinkhorn approximation method
                                (full, original, nystrom, multiscale, sparse, lcn)
    nystrom                     Dictionary containing Nyström approximation settings
    sparse                      Dictionary containing sparse approximation settings
    test                        Whether to evaluate on the test set
    data_dir                    Directory containing the data set
    output_src                  Whether and where to save the
                                final source language embeddings
    output_tgt                  Whether and where to save the
                                final (rotated) target language embeddings
    device                      Device used by PyTorch (cpu, cuda)
    """
    model_src = f"{data_dir}/wiki.{language_src}.vec"
    model_tgt = f"{data_dir}/wiki.{language_tgt}.vec"
    if test:
        val_numbers = "5000-6500"
    else:
        val_numbers = "0-5000"
    lexicon = f"{data_dir}/{language_src}-{language_tgt}.{val_numbers}.txt"

    sinkhorn_reg = torch.tensor(sinkhorn_reg, device=device)

    logging.info("*** Wasserstein Procrustes ***")

    np.random.seed(seed)
    torch.manual_seed(seed)

    maxload = 200_000
    w_src, x_src = pickle_cache(
        f"{model_src}.pkl",
        load_vectors,
        [model_src, maxload],
        dict(norm=True, center=True),
    )
    w_tgt, x_tgt = pickle_cache(
        f"{model_tgt}.pkl",
        load_vectors,
        [model_tgt, maxload],
        dict(norm=True, center=True),
    )
    src2tgt, _ = load_lexicon(lexicon, w_src, w_tgt)

    x_src = torch.tensor(x_src, dtype=torch.float).to(device)
    x_tgt = torch.tensor(x_tgt, dtype=torch.float).to(device)

    logging.info("Computing initial mapping with convex relaxation...")
    torch.cuda.synchronize()
    t0 = time.time()
    R0 = convex_init(
        x_src[:ninit],
        x_tgt[:ninit],
        sinkhorn_reg=sinkhorn_reg,
        apply_sqrt=True,
        disable_tqdm=True
    )
    torch.cuda.synchronize()
    logging.info(f"Done [{time.time() - t0:.1f} sec]")

    logging.info("Computing mapping with Wasserstein Procrustes...")
    torch.cuda.synchronize()
    t0 = time.time()
    if method == "original":
        R = align_original(x_src, x_tgt, R0.clone(), sinkhorn_reg, disable_tqdm=True)
    else:
        R = align(
            x_src,
            x_tgt,
            R0.clone(),
            sinkhorn_reg=sinkhorn_reg,
            method=method,
            nystrom=nystrom,
            sparse=sparse,
            lr=lr,
            niter=niter,
            lr_half_niter=lr_half_niter,
            ntrain=ntrain,
            print_niter=print_niter,
            disable_tqdm=True,
        )
    torch.cuda.synchronize()
    runtime = time.time() - t0
    logging.info(f"Done [{runtime:.1f} sec]")

    if test:
        logging.info("Evaluation on test set")
    else:
        logging.info("Evaluation on validation set")

    x_tgt_rot = x_tgt @ R.T

    acc_nn = compute_accuracy(x_src, x_tgt_rot, src2tgt, compute_nn)
    logging.info(f"NN precision@1: {100 * acc_nn:.2f}%")

    acc_csls = compute_accuracy(x_src, x_tgt_rot, src2tgt, compute_csls)
    logging.info(f"CSLS precision@1: {100 * acc_csls:.2f}%")

    x_tgt_rot = refine(x_src, x_tgt_rot, src2tgt, disable_tqdm=True)

    acc_refined = compute_accuracy(x_src, x_tgt_rot, src2tgt, compute_csls)
    logging.info(f"Refined CSLS precision@1: {100 * acc_refined:.2f}%")

    if output_src:
        x_src /= torch.norm(x_src, dim=1, keepdim=True) + 1e-8
        save_vectors(output_src, x_src.cpu().numpy(), w_src)
    if output_tgt:
        x_tgt_rot /= torch.norm(x_tgt_rot, dim=1, keepdim=True) + 1e-8
        save_vectors(output_tgt, x_tgt_rot.cpu().numpy(), w_tgt)

    return {
        "acc_nn": acc_nn,
        "acc_csls": acc_csls,
        "acc_refined": acc_refined,
        "runtime": runtime,
    }
