"""
Train and eval functions used in main.py
"""

from curses import flash
import numpy as np
from sklearn import cluster
import torch
import faiss
import torch.nn as nn
from datasets.flare import FlareDataset
from models.model import FlareFormer
from utils.losses import Losser
from utils.statistics import Stat
from utils.utils import adjust_learning_rate

from tqdm import tqdm
from argparse import Namespace
from typing import Dict, Tuple, Any
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader


def train_epoch(model: FlareFormer,
                optimizer: Adam,
                train_dl: DataLoader,
                epoch: int,
                lr: float,
                args: Namespace,
                losser: Losser,
                stat: Stat) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""
    model.train()
    losser.clear()
    for _, (x, y, _) in enumerate(tqdm(train_dl)):
        if not args.without_schedule:
            adjust_learning_rate(optimizer, epoch, args.dataset["epochs"], lr, args)
        optimizer.zero_grad()

        imgs, feats = x
        imgs, feats = imgs.cuda().float(), feats.cuda().float()
        output, _ = model(imgs, feats)
        gt = y.cuda().to(torch.float)
        loss = losser(output, gt)
        loss.backward()
        optimizer.step()
        stat.collect(output, y)

    score = stat.aggregate("train")
    return score, losser.get_mean_loss()


def train_pcl_epoch(model: FlareFormer,
                optimizer: Adam,
                train_dl: DataLoader,
                epoch: int,
                lr: float,
                args: Namespace,
                losser: Losser,
                stat: Stat) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""

    cluster_result = None
    # if epoch>=args.warmup_epoch:

    model.eval()
    outputs = []
    print("E-step ...")
    idx_saved = []
    for _, (x, y, idx) in enumerate(tqdm(train_dl)):
        with torch.no_grad():
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output = model(imgs, feats,is_eval=True)
            outputs.append(output)
            idx_saved.append(idx)

    features = torch.cat(outputs,dim=0)
    # placeholder for clustering result
    cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
    cluster_result['im2cluster'].append(torch.zeros(len(train_dl.dataset),dtype=torch.long).cuda())
    cluster_result['centroids'].append(torch.zeros(4,4).cuda())
    cluster_result['density'].append(torch.zeros(4).cuda()) 

    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.detach().cpu().numpy()
    cluster_result = run_kmeans(features)  #run kmeans clustering on master node

    print("M-step ...")
    model.train()
    losser.clear()
    s = 0
    for l, (x, y, idx) in enumerate(tqdm(train_dl)):
        if not args.without_schedule:
            adjust_learning_rate(optimizer, epoch, args.dataset["epochs"], lr, args)
        
        imgs,feats,y,idx = [], [], [], []
        for id in idx_saved[l]:
            _x, _y, _idx = train_dl.dataset[id]
            im, fe = _x
            imgs.append(im.unsqueeze(0))
            feats.append(torch.Tensor(fe).unsqueeze(0))
            y.append(torch.Tensor(_y).unsqueeze(0))
            idx.append(_idx.unsqueeze(0))
            
        imgs = torch.cat(imgs,dim=0)
        feats = torch.cat(feats,dim=0)
        y = torch.cat(y,dim=0)
        idx = torch.cat(idx,dim=0)
        for i in range(4):
            for j in range(i+1,4):
                y_label = torch.argmax(y, dim=1)
                imgs1, feats1 = imgs[y_label == i].cuda().float(), feats[y_label == i].cuda().float()
                imgs2, feats2 = imgs[y_label == j].cuda().float(), feats[y_label == j].cuda().float()
                index = y_label == i
                output, target, output_proto, target_proto = model(imgs1, feats1, imgs2, feats2, cluster_result=cluster_result,index=index + s)
                loss = losser(output, target, output_proto, target_proto)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # stat.collect(output, y)
        s += idx.shape[0]

    return losser.get_mean_loss()


def eval_epoch(model: FlareFormer,
               val_dl: DataLoader,
               losser: Losser,
               args: Namespace,
               stat: Stat) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    model.eval()
    losser.clear()
    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output, _ = model(imgs, feats)
            gt = y.cuda().to(torch.float)
            _ = losser(output, gt)
            stat.collect(output, y)

    score = stat.aggregate("valid")
    return score, losser.get_mean_loss()


def eval_pcl_epoch(model: FlareFormer,
               val_dl: DataLoader,
               losser: Losser,
               args: Namespace,
               stat: Stat) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    model.eval()
    losser.clear()
    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output = model(imgs, feats)
            gt = y.cuda().to(torch.float)
            _ = losser(output, gt)
            stat.collect(output, y)

    score = stat.aggregate("valid")
    return score, losser.get_mean_loss()



def run_kmeans(x):
    """
    Args:
        x: data to be clustered
    """
    
    results = {'im2cluster':[],'centroids':[],'density':[]}
    # intialize faiss clustering parameters
    d = x.shape[1]
    k = 20 # バッチサイズ // 4 より大きくないと負例が取れない可能性があるのでNG
    clus = faiss.Clustering(d, k)
    # clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 42
    clus.max_points_per_centroid = 450
    clus.min_points_per_centroid = 3

    # res = faiss.StandardGpuResources()
    # cfg = faiss.GpuIndexFlatConfig()
    # cfg.useFloat16 = False
    # cfg.device = 0
    # index = faiss.GpuIndexFlatL2(res, d, cfg)
    index = faiss.IndexFlatL2(d)
    
    # print("Clustering ...",flush=True)
    clus.train(x, index)
    
    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]          
    for im,i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = 0.2*density/density.mean()  #scale the mean to temperature 
    
    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)    

    im2cluster = torch.LongTensor(im2cluster).cuda()               
    density = torch.Tensor(density).cuda()
    
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)    
        
    return results
