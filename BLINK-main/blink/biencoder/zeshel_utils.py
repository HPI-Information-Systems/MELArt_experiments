# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Utility code for zeshel dataset
import json
import torch
import numpy as np
# DOC_PATH = "/private/home/ledell/zeshel/data/documents/"
DOC_PATH = "/scratch/user/uqlle6/code/artemo/BLINK-main/data/artel3/documents/"
WORLDS = ["human","location","undefined"]
WORLDS = ["undefined"]
# WORLDS = [
#     'american_football',
#     'doctor_who',
#     'fallout',
#     'final_fantasy',
#     'military',
#     'pro_wrestling',
#     'starwars',
#     'world_of_warcraft',
#     'coronation_street',
#     'muppets',
#     'ice_hockey',
#     'elder_scrolls',
#     'forgotten_realms',
#     'lego',
#     'star_trek',
#     'yugioh'
# ]

world_to_id = {src : k for k, src in enumerate(WORLDS)}


def load_entity_dict_zeshel(logger, params):
    entity_dict = {}
    # different worlds in train/valid/test
    if params["mode"] == "train":
        start_idx = 0
        end_idx = 8
    elif params["mode"] == "valid":
        start_idx = 8
        end_idx = 12
    else:
        start_idx = 12
        end_idx = 16
    # load data
    for i, src in enumerate(WORLDS):#WORLDS[start_idx:end_idx]
        fname = DOC_PATH + src + ".json"
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(fname, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                text = item["title"]+". "+item["text"]
                doc_list.append(text[:256])

                # if params["debug"]:
                #     if len(doc_list) > 200:
                #         break

        logger.info("Load for world %s." % src)
        entity_dict[src_id] = doc_list
    return entity_dict


def calculate_mrr(ranks):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
    ranks (list): A list of ranks of the first relevant document for each query.

    Returns:
    float: The MRR value.
    """

    mrr = np.sum((1. / np.array(ranks)))/(1+len(ranks))
    return mrr


def calculate_mr(ranks):
    """
    Calculate Mean Rank (MR).

    Args:
    ranks (list): A list of ranks of the first relevant document for each query.

    Returns:
    float: The MR value.
    """
    mr = sum(ranks)/(len(ranks)+1)
    return mr
class Stats():
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 3,5,10,15,20]
        self.LEN = len(self.rank) 
        for i in range(self.LEN):
            self.hits.append(0)
        self.results=[]

    def add(self, idx):
        if idx==-1:
            self.results.append(500)
        else:
            self.results.append(idx+1)
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1


    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        output_json += " mr: %.4f, mrr: %.4f" %(calculate_mr(self.results), calculate_mrr(self.results))
        return output_json

