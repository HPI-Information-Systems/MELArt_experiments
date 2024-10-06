# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id , Stats
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from pathlib import Path


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)

def get_ranks(out, labels):
    #The rank of the true label in the output
    ranks = []
    sorted_indices= []
    for i in range(len(out)):
        out_i = out[i]
        label_i = labels[i]
        if label_i == -1:
            ranks.append(-1)
        else:
            rank=sum([1 for j in out_i if j > out_i[label_i]])
            ranks.append(rank)
        # compute the sorted indices
        sorted_indices_i = np.argsort(out_i)
        sorted_indices.append(sorted_indices_i)
        # rank = 1
        # for j in range(len(out_i)):
        #     if out_i[j] > out_i[label_i]:
        #         rank += 1
        # ranks.append(rank)
    return ranks, sorted_indices


def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    all_logits = []
    all_sorted_indices = []
    all_ranks = []
    cnt = 0
    stats=Stats(top_k=30)
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        #clone the labels
        label_input_temp = label_input.clone()
        #assign label 0 to the -1s, it should not affect the scores during inference.
        label_input_temp[label_input_temp==-1]=0

        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input_temp, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)
        ranks,sorted_indices=get_ranks(logits, label_ids)
        all_sorted_indices.extend(sorted_indices)
        for rank in ranks:
            stats.add(rank)
        all_ranks.extend(ranks)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1

    logger.info(stats.output())    

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if zeshel:
        macro = 0.0
        num = 0.0 
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    #results["logits"] = all_logits
    results["ranks"] = all_ranks
    results["sorted_indices"] = all_sorted_indices
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    training_params = params["training_params"]
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    training_params["path_to_model"]=params["path_to_model"]

    # Init model
    reranker = CrossEncoderRanker(training_params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    #load state dict
    #model.module.load_state_dict(torch.load(params["path_to_model"]))

    print("Model loaded from", params["path_to_model"])

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    
    fname = os.path.join(training_params["data_path"], "test.t7")
    test_data = torch.load(fname)
    context_input = test_data["context_vecs"]
    candidate_input = test_data["candidate_vecs"]
    label_input = test_data["labels"]
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    test_tensor_data = TensorDataset(context_input, label_input)
    test_sampler = RandomSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )


    time_start = time.time()

    results = evaluate(
        reranker,
        test_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        zeshel=params["zeshel"],
        silent=params["silent"],
    )

    number_of_samples_per_dataset = {}

    #convert all numpy arrays to lists
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        if isinstance(results[key], list) and isinstance(results[key][0], np.ndarray):
            results[key] = [x.tolist() for x in results[key]]

    utils.write_to_file(
        os.path.join(model_output_path, "eval_results.json"), json.dumps(results, indent=2)
    )

    utils.write_to_file(
        os.path.join(model_output_path, "eval_params.txt"), str(params)
    )

    # logger.info("Starting training")
    # logger.info(
    #     "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    # )

    # optimizer = get_optimizer(model, params)
    # scheduler = get_scheduler(params, optimizer, len(test_tensor_data), logger)

    # model.train()

    # best_epoch_idx = -1
    # best_score = -1

    # num_train_epochs = params["num_train_epochs"]

    # for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
    #     tr_loss = 0
    #     results = None

    #     if params["silent"]:
    #         iter_ = test_dataloader
    #     else:
    #         iter_ = tqdm(test_dataloader, desc="Batch")

    #     part = 0
    #     for step, batch in enumerate(iter_):
    #         batch = tuple(t.to(device) for t in batch)
    #         context_input = batch[0] 
    #         label_input = batch[1]
    #         loss, _ = reranker(context_input, label_input, context_length)

    #         # if n_gpu > 1:
    #         #     loss = loss.mean() # mean() to average on multi-gpu.

    #         if grad_acc_steps > 1:
    #             loss = loss / grad_acc_steps

    #         tr_loss += loss.item()

    #         if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
    #             logger.info(
    #                 "Step {} - epoch {} average loss: {}\n".format(
    #                     step,
    #                     epoch_idx,
    #                     tr_loss / (params["print_interval"] * grad_acc_steps),
    #                 )
    #             )
    #             tr_loss = 0

    #         loss.backward()

    #         if (step + 1) % grad_acc_steps == 0:
    #             torch.nn.utils.clip_grad_norm_(
    #                 model.parameters(), params["max_grad_norm"]
    #             )
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()

    #         if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
    #             logger.info("Evaluation on the development dataset")
    #             evaluate(
    #                 reranker,
    #                 valid_dataloader,
    #                 device=device,
    #                 logger=logger,
    #                 context_length=context_length,
    #                 zeshel=params["zeshel"],
    #                 silent=params["silent"],
    #             )
    #             logger.info("***** Saving fine - tuned model *****")
    #             epoch_output_folder_path = os.path.join(
    #                 model_output_path, "epoch_{}_{}".format(epoch_idx, part)
    #             )
    #             part += 1
    #             utils.save_model(model, tokenizer, epoch_output_folder_path)
    #             model.train()
    #             logger.info("\n")

    #     logger.info("***** Saving fine - tuned model *****")
    #     epoch_output_folder_path = os.path.join(
    #         model_output_path, "epoch_{}".format(epoch_idx)
    #     )
    #     utils.save_model(model, tokenizer, epoch_output_folder_path)
    #     # reranker.save(epoch_output_folder_path)

    #     output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
    #     results = evaluate(
    #         reranker,
    #         valid_dataloader,
    #         device=device,
    #         logger=logger,
    #         context_length=context_length,
    #         zeshel=params["zeshel"],
    #         silent=params["silent"],
    #     )

    #     ls = [best_score, results["normalized_accuracy"]]
    #     li = [best_epoch_idx, epoch_idx]

    #     best_score = ls[np.argmax(ls)]
    #     best_epoch_idx = li[np.argmax(ls)]
    #     logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "testing_time.txt"),
        "The evaluation took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    #parser.add_training_args()
    parser.add_eval_args()
    parser.add_argument(
        "crossencoder_model",
        type=str,
        help="The path to the crossencoder model folder",
    )

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    crossencoder_model_path=Path(args.crossencoder_model)
    training_params={}
    training_params_path = crossencoder_model_path / "training_params.txt"
    #read dict from file
    with open(training_params_path, "r") as file:
        training_params_txt = file.read()
        #as dict
        training_params = dict(eval(training_params_txt))
    
    print(training_params)

    params = args.__dict__
    params["training_params"] = training_params

    best_epoch = 0
    log_txt= crossencoder_model_path / "log.txt"
    with open(log_txt, "r") as file:
        for line in file:
            if "Best performance in epoch" in line:
                # 10/05/2024 19:39:40 - INFO - Blink -   Best performance in epoch: 0
                # extract the epoch number
                best_epoch = int(line.split(":")[-1].strip())

    #set path_to_model
    params["path_to_model"] = crossencoder_model_path / f"epoch_{best_epoch}" / "pytorch_model.bin"

    main(params)
