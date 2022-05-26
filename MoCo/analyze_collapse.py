import torch
from torch.optim import SGD
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

def norm(x):
    return np.linalg.norm(x, ord=2, axis=1, keepdims=False)

#analyze last layer feature
def neural_collapse_embedding(class_num, t1, model, data_loader, device):
    model.eval()
    embedding = []
    label = []
    for iter, pack in enumerate(data_loader):
        # data, target = pack[0].cuda(args.gpu, non_blocking=True), pack.cuda(args.gpu, non_blocking=True)
        data, target = pack[0].to(device), pack[1].to(device)
        embed = model.forward_embedding(data)
        embedding_arr = embed.detach().cpu().numpy()
        embedding.append(embedding_arr)
        label_arr = target.cpu().numpy()
        label.append(label_arr)
    embedding_np = np.concatenate(embedding, 0)
    label_np = np.concatenate(label, 0)
    # print( embedding_np.shape)    
    class_embedding = []
    class_mean = []
    global_mean = np.mean(embedding_np, 0, keepdims=True)
    corr_intra = []
    corr_inter = []
    corr_intra_large = []
    corr_intra_small = []
    corr_inter_large = []
    corr_inter_small = []
    for k in range(class_num):
        tmp_index = [i for i in range(len(label_np)) if int(label_np[i]) ==k]
        class_embedding.append(embedding_np[tmp_index])
        class_mean.append(np.mean(embedding_np[tmp_index], 0))
        corr_intra.append(np.matmul((embedding_np[tmp_index] - np.mean(embedding_np[tmp_index], 0, keepdims=True)).transpose(), (embedding_np[tmp_index] - np.mean(embedding_np[tmp_index], 0, keepdims=True)) ))
        corr_inter.append(np.matmul((np.mean(embedding_np[tmp_index], 0, keepdims=True) - global_mean).transpose(), (np.mean(embedding_np[tmp_index], 0, keepdims=True) - global_mean) ))
    large_mean = np.mean(np.concatenate(class_embedding[:t1],0), 0, keepdims=True)
    small_mean = np.mean(np.concatenate(class_embedding[t1:],0), 0, keepdims=True)
    for k in range(t1):
        corr_intra_large.append(np.matmul((class_embedding[k] - class_mean[k]).transpose(), (class_embedding[k] - class_mean[k])))
        corr_inter_large.append(np.matmul((class_mean[k] - large_mean).transpose(), (class_mean[k] - large_mean)))
    for k in range(t1, class_num):
        corr_intra_small.append(np.matmul((class_embedding[k] - class_mean[k]).transpose(), (class_embedding[k] - class_mean[k])))
        corr_inter_small.append(np.matmul((class_mean[k] - small_mean).transpose(), (class_mean[k] - small_mean)))
    corr_intra = np.mean(np.array(corr_intra), 0)
    corr_intra_large = np.mean(np.array(corr_intra_large), 0)
    corr_intra_small = np.mean(np.array(corr_intra_small), 0)
    corr_inter = np.mean(np.array(corr_inter), 0)
    corr_inter_large = np.mean(np.array(corr_inter_large), 0)
    corr_inter_small = np.mean(np.array(corr_inter_small), 0)
    intra_v = np.matrix.trace(corr_intra * np.linalg.pinv(corr_inter, rcond=1e-6)) / class_num
    intra_v_large = np.matrix.trace(corr_intra_large * np.linalg.pinv(corr_inter_large, rcond=1e-6)) / t1
    intra_v_small = np.matrix.trace(corr_intra_small * np.linalg.pinv(corr_inter_small, rcond=1e-6)) / (class_num - t1)
    equal_norm_activation = np.std(norm(np.array(class_mean) - global_mean)) / np.mean(norm(np.array(class_mean) - global_mean))
    class_mean_matrix = np.array(class_mean) - global_mean
    cosine_sim = cosine_similarity(class_mean_matrix, class_mean_matrix).ravel()
    cosine_sim = np.array([i for i in cosine_sim if i<0.9999])
    equa_ang = np.std(cosine_sim)
    equa_ang_2 = np.mean(np.abs(cosine_sim + 1/(class_num-1)))
    return equal_norm_activation, equa_ang, equa_ang_2, intra_v, intra_v_large, intra_v_small

# def weight_feature(weight, class_num):
#     weight_norm = np.linalg.norm(weight, ord=2, axis=1, keepdims=False).ravel()
#     equinorm_weight = np.std(weight_norm) / np.mean(weight_norm)
#     normalized_weight =  weight / np.linalg.norm(weight, ord='fro')
#     weight_n = weight - np.mean(weight, 0)
#     cosine_sim = cosine_similarity(weight_n, weight_n).ravel()
#     cosine_sim = np.array([i for i in cosine_sim if i<0.9999])
#     equa_ang_w = np.std(cosine_sim)
#     equa_ang_w_2 = np.mean(np.abs(cosine_sim + 1/(class_num - 1)))
#     return equinorm_weight, normalized_weight, equa_ang_w, equa_ang_w_2

#Analyze weights and Minority collapse
'''
input—— 
linear_weights: np.array of shape (class_num, feature_dim)
t1: int, number of majority classes; the first t1 classes are majority classes
'''
def analyze_weight(linear_weights, t1):
    num_classes = len(linear_weights)
    weight_norm = [np.linalg.norm(linear_weights[i]) for i in range(num_classes)]
    cos_matrix = np.zeros((num_classes, num_classes))
    between_class_cos = []
    for i in range(num_classes):
        for j in range(num_classes):
            cos_value = 1 - spatial.distance.cosine(linear_weights[i], linear_weights[j])
            cos_matrix[i, j] = cos_value
            if i != j:
                between_class_cos.append(cos_value)
    weight_norm = np.array(weight_norm)
    avg_square_norm = np.mean(np.square(weight_norm))
    between_class_cos = np.array(between_class_cos)
    equinorm_weight = np.std(weight_norm) / np.mean(weight_norm)
    between_avg = np.mean(between_class_cos)
    equa_ang_w = np.std(between_class_cos)
    equa_ang_w_2 = np.mean(np.abs(between_class_cos + 1 / (num_classes - 1)))
    # compute between-class cosine for small classes
    if t1 != len(linear_weights):
        between_class_cos_small = []
        for i in range(num_classes)[t1:]:
            for j in range(num_classes)[t1:]:
                if i != j:
                    between_class_cos_small.append(cos_matrix[i, j])
        between_class_cos_small = np.array(between_class_cos_small)
        equinorm_weight_small =  np.std(weight_norm[t1:]) / np.mean(weight_norm[t1:])
        between_small_avg = np.mean(between_class_cos_small)
    # compute between-class cosine for large classes
    if t1 != len(linear_weights):
        between_class_cos_large = []
        for i in range(num_classes)[:t1]:
            for j in range(num_classes)[:t1]:
                if i != j:
                    between_class_cos_large.append(cos_matrix[i, j])
        between_class_cos_large = np.array(between_class_cos_large)
        equinorm_weight_large =  np.std(weight_norm[:t1]) / np.mean(weight_norm[:t1])
        between_large_avg = np.mean(between_class_cos_large)
    return equa_ang_w_2, equinorm_weight, equinorm_weight_large, equinorm_weight_small, between_avg, between_large_avg, between_small_avg

def analyze_dual(linear_weights, class_features):
    n_class = len(class_features)
    linear_weights = linear_weights[:n_class]
    linear_weights = linear_weights / np.linalg.norm(linear_weights)
    class_features = class_features / np.linalg.norm(class_features)
    # print('normalized linear weights', linear_weights)
    # print('normalized class features', class_features)
    dual_distance = np.linalg.norm(linear_weights - class_features)
    dual_distance_square = np.square(np.linalg.norm(linear_weights - class_features))
    return dual_distance, dual_distance_square