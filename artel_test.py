import blink.main_dense as main_dense
import argparse
import os
import json
def dump_json(obj,save_path):
    with open(save_path, 'w') as outfile:
        json.dump(obj, outfile)
def load_json(file_path):
    if os.path.exists(file_path):
        file=open(file_path,"r")
        return json.load(file)
    return {}
models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": "/scratch/user/uqlle6/code/artemo/BLINK-main/artel2/documents.jsonl",
    "test_mentions": "/scratch/user/uqlle6/code/artemo/BLINK-main/artel2/documents.jsonl",
    "interactive":False,
    "top_k": 20,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": True, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}
args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": 692,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": 1048,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                }
                ]

json_f="/scratch/user/uqlle6/code/artemo/BLINK-main/data/blink_format/test.jsonl"
test_data=[]
with open(json_f) as f:
    test_data = [json.loads(l) for l in f.readlines()]
    for i,record in enumerate(test_data):
        record["context_left"]=record["context_left"].lower()
        record["context_right"] = record["context_right"].lower()
        record["mention"] = record["mention"].lower()
        record["label"]= "unknown"
        record["id"] = i
        test_data[i]=record

# print(test_data)
biencoder_accuracy,recall_at,crossencoder_normalized_accuracy,overall_unormalized_accuracy,sample_len, predictions, scores, = main_dense.run(args, None, *models, test_data=test_data)
dump_json(predictions,"/scratch/user/uqlle6/code/artemo/BLINK-main/data/blink_format/test_preds.json")
print("LEN",len(predictions))
print(predictions[0])
def get_evaluation_score(predictions,groundtruths):
    ranks=[]
    for i,gt in enumerate(groundtruths):
        preds=predictions[i]
        for j,pred in enumerate(preds):
            if pred ==gt:
                ranks.append(j)


    hits20 = (ranks <= 20).mean()
    hits10 = (ranks <= 10).mean()
    hits5 = (ranks <= 5).mean()
    hits3 = (ranks <= 3).mean()
    hits1 = (ranks <= 1).mean()

    print("Test/hits20", hits20)
    print("Test/hits10", hits10)
    print("Test/hits5", hits5)
    print("Test/hits3", hits3)
    print("Test/hits1", hits1)
    print("Test/mr", ranks.mean())
    print("Test/mrr", (1. / ranks).mean())
    return hits1,hits3,hits5,hits10,hits20,ranks.mean(),(1. / ranks).mean()
