from random import randint
import numpy as np
import torch
from numpy import save
from tqdm import tqdm
import pandas as pd
import json
from models import InferSent
import argparse

V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()
W2V_PATH = "./fastText/crawl-300d-2M.vec"
model.set_w2v_path(W2V_PATH)
model.build_vocab_k_words(K=100000)


def infersent_embed_doc(rpath, wpath):
    df = pd.read_csv(rpath,chunksize=1000)
    text = []
    count=0
    for chunk in df:
        text = text + chunk['comment'].tolist()
    error_idx = []
    with open(wpath,'w+') as fw:
        for i in range(0,len(text)):
            try:
                embeddings = model.encode([text[i]], bsize=64, tokenize=False, verbose=True).tolist()[0]
                torch.cuda.empty_cache()
            except RuntimeError:
                error_idx.append(i)
            
            fw.write(str(embeddings)+"\n")
            if i%10==0:
                print(i)
            count+=1

def read_data(path):
    df = pd.read_csv(path,header=None)    
    return df[0].tolist(),df[1].tolist()

def inferesent_embedding(path,text,label,batch=64):
    error_idx = []
    count= 0
    with open(path,'w+') as fw:
        for i,j in zip(text,label):
            try:
                print(count)
                embeddings = model.encode([i], bsize=512, tokenize=False, verbose=True).tolist()[0]
                data = {"label":j-1, "embedding": embeddings}
                json.dump(data, fw)
                fw.write("\n")
            except RuntimeError:
                error_idx.append(i)
            count+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdoc", default="/home/amoghmishra23/appledore/MBTI-Personality-Detection/data/mbti_comments/mbti_comments_cleaned.csv")
    parser.add_argument("--wdoc",default="output/embedding_infersent.txt")
    parser.add_argument("--read", default="/home/amoghmishra23/appledore/MBTI-Personality-Detection//data/mbti_comments/extra_clean_seq_60.csv")
    parser.add_argument("--write",default="../output/embedding_extra_clean_seq_60.json")

    args = parser.parse_args()
    infersent_embed_doc(args.rdoc,args.wdoc)
    text, label = read_data(args.read)
    inferesent_embedding(args.write, text, label, 512)
    