import os
import time
import json
import math
import heapq
import numpy as np

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Retrieval data preprocessing.

Reference Data = "coco"
Retrieval Data = "flickr30k", "nocaps"
Retrieval method = Learning Visual Representations via Language-guided Sampling [CVPR 2023] (https://arxiv.org/abs/2302.12248)
"""

def text_to_embedding(text_list, text_model):
    start = time.time()
    text_embedding = text_model.encode(text_list)
    print(f"text embedding time: {(time.time() - start) / 60:.2f} minutes")
    return text_embedding

def top_k_cosine_similarity(tensor1, tensor2, k=3):
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(tensor1, tensor2)

    # Find top k indices
    _, top_k_indices = torch.topk(cos_sim, k)

    return top_k_indices

def check_embedding(data_dir, text_list, text_model):
    #! data_dir must be .npy file
    if not os.path.exists(data_dir):
        text_embedding = text_to_embedding(text_list, text_model)
        np.save(data_dir, text_embedding)
    else:
        text_embedding = np.load(data_dir)

    return text_embedding

def main(ref_dataset, ret_datasets, ref_data_json, ret_data_jsons):

    # Initialize
    reference_text, retrieval_text, augmented_data = [], [], []

    # Load Model
    text_model = SentenceTransformer("sentence-transformers/sentence-t5-xl", device="cuda")

    for ret_dataset, ret_data_json in zip(ret_datasets, ret_data_jsons):
        top_three_all = []

        # check: Reference Dataset Preprocessing
        # COCO dataset preprocessing
        if ref_dataset == "coco":
            with open(ref_data_json, "r") as f:
                reference_data = json.load(f)

                for i in tqdm(range(len(reference_data))):
                    for j in range(len(reference_data[i]["sentences"])):
                        augmented_dict = {}

                        reference_text.append(reference_data[i]["sentences"][j]["raw"])

                        augmented_dict["filename"] = reference_data[i]["filename"]
                        augmented_dict["filepath"] = reference_data[i]["filepath"]
                        augmented_dict["raw"] = reference_data[i]["sentences"][j]["raw"]

                        augmented_data.append(augmented_dict)

            # check the embedding file exists and extract to embedding file
            reference_text_embedding = check_embedding("data/retrieval/coco_reference_t5.npy", reference_text, text_model)

        # check: Retrieval Dataset Preprocessing
        # flickr30k dataset preprocessing
        if ret_dataset == "flickr30k":
            with open(ret_data_json, "r") as f:
                retrieval_data = json.load(f)
                for i in tqdm(range(len(retrieval_data))):
                    retrieval_text.append(retrieval_data[i]["caption"])

            # check the embedding file exists and extract to embedding file
            retrieval_text_embedding = check_embedding("data/retrieval/flickr30k_retrieval_t5.npy", retrieval_text, text_model)

        # nocaps dataset preprocessing
        elif ret_dataset == "nocaps":
            with open(ret_data_json, "r") as f:
                retrieval_data = json.load(f)
                for i in tqdm(range(len(retrieval_data["annotations"]))):
                    retrieval_text.append(retrieval_data["annotations"][i]["caption"])

            # check the embedding file exists and extract to embedding file
            retrieval_text_embedding = check_embedding("data/retrieval/nocaps_retrieval_t5.npy", retrieval_text, text_model)

        # TODO: make redcaps available
        elif ret_dataset == "redcaps":
            with open(ret_data_json, "r") as f:
                retrieval_data = json.load(f)
                for i in tqdm(range(len(retrieval_data))):
                    retrieval_text.append(retrieval_data[i]["caption"])

            # check: try this one first
            # retrieval_text_embedding = check_embedding("data/retrieval/redcaps_retrieval_t5.npy", retrieval_text, text_model)
            retrieval_text_embedding = []

            batch = 121231  # 3273237 / 27
            start = time.time()
            for i in tqdm(range(len(retrieval_text) // batch), desc="Embedding"):
                text_list = retrieval_text[batch * i:batch * (i + 1)]
                text_embedding = text_model.encode(text_list)
                retrieval_text_embedding = np.append(embedding, text_embedding, axis=0)

            print(f"text embedding time: {(time.time() - start) / 60:.2f} minutes")

        # np.ndarray to torch.tensor
        reference_text_embedding = torch.tensor(reference_text_embedding, device=device)
        retrieval_text_embedding = torch.tensor(retrieval_text_embedding, device=device)

        # Top k cosine similarity, return top three indices
        for i in tqdm(range(len(reference_text_embedding)), desc="Retrieval"):
            augmented_dict = {}
            top_three = top_k_cosine_similarity(retrieval_text_embedding, reference_text_embedding[i], k=3)
            top_three_all.extend(top_three)

        # flickr30k to data format
        if ret_dataset == "flickr30k":
            for idx in top_three_all:
                augmented_dict = {}
                image_path = retrieval_data[idx]["image"]
                file_name = os.path.basename(image_path)
                directory_name = os.path.dirname(image_path)

                augmented_dict["filename"] = file_name
                augmented_dict["filepath"] = directory_name
                augmented_dict["raw"] = retrieval_data[idx]["caption"]

                augmented_data.append(augmented_dict)

        elif ret_dataset == "nocaps":
            for idx in top_three_all:
                augmented_dict = {}
                image_path = "nocaps"

                new_idx = idx // 10
                file_name = retrieval_data["images"][new_idx]["file_name"]
                directory_name = "val"

                augmented_dict["filename"] = file_name
                augmented_dict["filepath"] = directory_name


                augmented_dict["raw"] = retrieval_data["annotations"][idx]["caption"]

                augmented_data.append(augmented_dict)

        elif ret_dataset == "redcaps":
            download_json = []
            for idx in top_three_all:
                # for the data download
                download_json.append(retrieval_data[idx])

                # for the retrieval data
                augmented_dict = {}

                file_name = retrieval_data[idx]["image_id"] + ".jpg"
                file_path = "redcaps" + "/" + retrieval_data[idx]["subreddit"]

                augmented_dict["filename"] = file_name
                augmented_dict["filepath"] = file_path
                augmented_dict["raw"] = retrieval_data[idx]["caption"]

                augmented_data.append(augmented_dict)

            filtered_download_json = []
            for d in tqdm(download_json, desc="Removing Duplicates in redcaps"):
                key = (d["image_id"])
                if key not in seen:
                    seen.add(key)
                    filtered_download_json.append(d)
            with open("/data/dataset/redcaps/filtered_redcaps_2020.json", "w") as f:
                json.dump(filtered_download_json, f)


    # Remove Duplicates
    seen = set()
    filtered_augmented_data = []

    for d in tqdm(augmented_data, desc="Removing Duplicates"):
        key = (d["filename"], d["filepath"], d["raw"])
        if key not in seen:
            seen.add(key)
            filtered_augmented_data.append(d)

    # Write json file
    filename = "/home/kylee/projects/mplug/data/retrieval/redcaps_retrieval_t5_2.json"
    with open(filename, "w") as f:
        json.dump(filtered_augmented_data, f)

if __name__ == "__main__":
    # ret_dataset = "nocaps"

    # ret_datasets = ["flickr30k", "nocaps"]

    # ret_data_json = "/data/dataset/nocaps/annotations/nocaps_val_4500_captions.json"
    # ret_data_jsons = ["/data/dataset/lavis/flickr30k/annotations/train.json", "/data/dataset/nocaps/annotations/nocaps_val_4500_captions.json"]

    ref_dataset = "coco"
    ref_data_json = "/home/kylee/projects/mplug/data/coco_object/coco_caption_train_ocr.json"

    # ret_datasets = ["flickr30k", "nocaps"]
    # ret_data_jsons = ["/data/dataset/lavis/flickr30k/annotations/train.json", "/data/dataset/nocaps/annotations/nocaps_val_4500_captions.json"]

    # ret_datasets = ["nocaps"]
    # ret_data_jsons = ["/data/dataset/nocaps/annotations/nocaps_val_4500_captions.json"]

    ret_datasets = ["redcaps"]
    ret_data_jsons = ["data/retrieval/redcaps_2020.json"]

    main(ref_dataset=ref_dataset, ret_datasets=ret_datasets, ref_data_json=ref_data_json, ret_data_jsons=ret_data_jsons)