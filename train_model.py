import argparse
import collections
import json
import random
import dgl
from dgl.data import DGLDataset
import numpy as np
from sklearn.metrics import log_loss
import torch
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sagnn import SAGNN
from focal_loss import FocalLoss2d

def _do_pause():
    raise RuntimeError("!!!!!!!!!!!!!!!!PAUSE")

class MyDataset(DGLDataset):
    
    def __init__(self, cfg_graphs, ast_graphs, graph_idx_list):
        self.cfg_graphs = cfg_graphs  # [  cfg1,           cfg2, ...,              cfgn]
        self.ast_graphs = ast_graphs  # [[ast1_1, ast1_2], [ast2_1, ast2_2],..., [astn_1, astn_2, astn_3]]
        self.graph_idx_list = graph_idx_list
        super().__init__(name='my Dataset')

    def process(self):
        pass

    def __getitem__(self, i):
        return self.cfg_graphs[i], self.ast_graphs[i], self.graph_idx_list[i]

    def __len__(self):
        return len(self.cfg_graphs)


my_batch = collections.namedtuple('my_batch', ['cfg_batch', 'ast_batch', "idx_batch", "node_lables"])
def batcher_v1(device, classify_type):
    def batcher_v1_dev(batch):
        cfgs = []
        asts = []
        gidx = []

        # [cfg_i], [ast_i_1, ast_i_2]
        for cfg_ast in batch:

            _cfg = cfg_ast[0]
            _cfg = dgl.add_self_loop(_cfg) # 在CFG中添加self loop
            cfgs.append(_cfg)

            for idx, ast in enumerate(cfg_ast[1]): 
                ast = dgl.remove_self_loop(ast) # 在AST中删除self loop
                asts.append(ast)
                gidx.append("{}-{}".format(cfg_ast[2], idx))

        cfg_batch = dgl.batch(cfgs) # batch:[1_2n    ,2_3n,     3_2n] 
        ast_batch = dgl.batch(asts) # batch:[[1,2],  [1,2,3],  [1,2]] => [1,2,3,4,5,6,7]

        if classify_type == "multi":
            _node_lables = cfg_batch.ndata['label']

        elif classify_type == "binary":
            _node_lables = cfg_batch.ndata['label_b']

        else:
            raise RuntimeError("错误的分类类型")

        return my_batch(cfg_batch=cfg_batch,
                        ast_batch=ast_batch,
                        idx_batch=gidx,
                        node_lables=_node_lables)
                        

    return batcher_v1_dev


def dgl_bin_process(dataset_name, c_type):

    idx_graphs = [] # 图索引
    cfg_graphs = []
    ast_graphs = []

    no_vul_cnt = 0
    total_cnt = 0
    # 抽取全部数据
    graphs, graphs_infos = dgl.load_graphs(dataset_name)
    for idx in range(0, len(graphs_infos["graph_cnts"])):
        tmp_ast_graphs = []
    
        graph_cnt = graphs_infos["graph_cnts"][idx].int()
        if graph_cnt == 0:
            raise RuntimeError("!!! 出现了没有图的样本")

        for graph_idx in range(0, graph_cnt):

            tmp_graph = graphs.pop(0)
            if graph_idx == 0:   # 第一个是CFG, 其它的是AST
                
                total_cnt += tmp_graph.num_nodes()
                no_vul_cnt += th.sum(tmp_graph.ndata["label"].argmax(1).eq(th.zeros(tmp_graph.num_nodes())))

                tmp_graph.ndata["feature"] = th.zeros(tmp_graph.num_nodes(), 64) # 为CFG开辟一个特征空间,大小为TREE_LSTM的输出
                if c_type == "multi":
                    pass

                elif c_type == "binary":
                    tmp_graph.ndata["label_b"] = tmp_graph.ndata["label"].argmax(1) # 建模为二分类任务, 不使用交叉熵

                cfg_graphs.append(tmp_graph)
            else:
                tmp_ast_graphs.append(tmp_graph)
        
        ast_graphs.append(tmp_ast_graphs)
        idx_graphs.append(idx)

    # 划分train/test/valid数据集
    random.shuffle(idx_graphs)
    train_size = int(0.8*len(idx_graphs))
    valid_size = int(0.1*len(idx_graphs))
    test_size  = len(idx_graphs) - train_size - valid_size

    print("\r\n====================统计信息===========================")
    print("==数据集:{}".format(dataset_name))
    print("==数量信息: total:{} train_size:{}, valid_size:{}, test_size:{}".format(
        len(cfg_graphs), train_size, valid_size, test_size)
    )
    print("==负标签分布: no_vul:{} {}".format(no_vul_cnt, no_vul_cnt/total_cnt))
    print("==正标签分布: vul:{} {}".format(total_cnt - no_vul_cnt, (total_cnt - no_vul_cnt)/total_cnt))

    train_idx_list = idx_graphs[0:train_size]  # 80%
    valid_idx_list = idx_graphs[train_size + 1: train_size + valid_size]  # 10%
    test_idx_list  = idx_graphs[train_size + valid_size + 1:]  # 10%
    
    # 根据划分的结果进行数据集构建
    train_cfg_graphs = []
    train_ast_graphs = []
    for tmp_id in train_idx_list:
        train_cfg_graphs.append(cfg_graphs[tmp_id])
        train_ast_graphs.append(ast_graphs[tmp_id])
    train_dataset = MyDataset(train_cfg_graphs, train_ast_graphs, train_idx_list)
    
    valid_cfg_graphs = []
    valid_ast_graphs = []
    for tmp_id in valid_idx_list:
        valid_cfg_graphs.append(cfg_graphs[tmp_id])
        valid_ast_graphs.append(ast_graphs[tmp_id])
    valid_dataset = MyDataset(valid_cfg_graphs, valid_ast_graphs, valid_idx_list)
    
    test_cfg_graphs = []
    test_ast_graphs = []
    for tmp_id in test_idx_list:
        test_cfg_graphs.append(cfg_graphs[tmp_id])
        test_ast_graphs.append(ast_graphs[tmp_id])
    test_dataset = MyDataset(test_cfg_graphs, test_ast_graphs, test_idx_list)
        
    return train_dataset, valid_dataset, test_dataset


def wrong_sample_log(fn_samples, fp_samples):

    if len(fn_samples) > 0:
        print("\r\n==============fn_sample:start===================")
        for cfgid_astid in fn_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')    
            print("FN: path:{} STMTID:{} type:{}".format(
                DATASET_DB[cfg_id]["path"], 
                ast_id, 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"]))
        print("==============fn_sample:end===================")
    

    if len(fp_samples) > 0:
        print("\r\n==============fp_samples:start===================")
        for cfgid_astid in fp_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')    
            print("FN: path:{} STMTID:{} type:{}".format(
                DATASET_DB[cfg_id]["path"], 
                ast_id, 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"]))
        print("==============fp_samples:end===================")


def calculate_metrics(preds, labels, idxs, prefix, epoch, postive=1):

    save_flag = useless_flag = 0
    TP = FP = TN = FN = 0
    fn_samples = []
    fp_samples = []

    # print("{} preds:{}, laels:{} idx:{}".format(prefix, len(preds), len(labels), len(idxs)))
    
    for pred, label, idx in zip(preds, labels, idxs):
        if pred == postive: # 预测为正样本:postive
            if pred == label: # 预测正确: TRUE
                TP += 1
            else: # 预测错误：FALSE
                FP += 1
                fp_samples.append(idx)
                
        else: # 预测为负样本: negtive
            if pred == label: # 预测正确: TRUE
                TN += 1
            else: # 预测错误：FALSE
                FN += 1
                fn_samples.append(idx)

    total_data_num = TP + TN + FP + FN

    # 计算acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # 计算recall
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        useless_flag = 1
        recall = 9999

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        useless_flag = 1
        precision = 9999

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        useless_flag = 1
        f1 = 9999
    
    # 计算 TPR FNR
    if(TP + FN) != 0:
        TPR = TP / (TP + FN)
        FNR = FN / (TP + FN)
    else:
        TPR = TPR = 9999

    # 计算 FPR TNR
    if(FP + TN) != 0:
        FPR = FP / (FP + TN)
        TNR = TN / (FP + TN)
    else:
        FPR = TNR = 9999
    
    print("{}:{}: EPOCH:{} ACC:{}, RE:{}, P:{},  F1:{}, TOTAL:{}".format(
            prefix, useless_flag, epoch, acc, recall, precision, f1, total_data_num)
        )
    
    if useless_flag == 0 and recall >= 0.90 and precision >= 0.70 and f1 >= 0.80:
        save_flag = 1
        wrong_sample_log(fn_samples, fp_samples)
        print("==TPR:{}, FNR:{}, FPR:{}, TNR:{}".format(TPR, FNR, FPR, TNR))

    return save_flag, acc, recall, precision, f1

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x): # x是step次数
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters # 当前进度 0-1
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-dataset', type=str, default="test")
    args = parser.parse_args()
    return args.dataset

def debug_model_infos():
    print("\r\n=========模型情况打印:===========")
    print("==batch_size:", batch_size)
    print("==epoch:", epoch)
    print("==x_size:", x_size)
    print("==h_size:", h_size)
    print("==attn_drop:", attn_drop)
    print("==feat_drop:", feat_drop)
    print("==warmup:", warmup)
    print("==loss_type:", loss_type)
    print("==classify_type:", classify_type)
    print("==device:", device)

if __name__ == '__main__':

    _dataset = argParse()
    if str(_dataset).endswith(".bin"):
        dataset_name = "dataset//" + _dataset
        dataset_db_file = "dataset//" + str(_dataset).split(".bin")[0] + "_db.json"
    else:
        dataset_name = "dataset//" + "{}.bin".format(_dataset)
        dataset_db_file = "dataset//" + _dataset + "_db.json"
    f = open(dataset_db_file, "r")
    DATASET_DB = json.load(f)

     # 参数列表
    torch.manual_seed(3407)
    batch_size = 1024
    epoch = 256
    x_size = 100 # ast node feature size
    h_size = 64  # tree_lstm learning feature or CFG node feature size
    attn_drop = 0.05
    feat_drop = 0.05
    warmup = 1
    loss_type = "focal_loss" # focal_loss  cross_entropy
    classify_type = "multi" # binary multi 
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        th.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    debug_model_infos()
    
    if loss_type == "focal_loss":
        focal_loss = FocalLoss2d().to(device)

    # 数据
    train_dataset, valid_dataset, test_dataset = dgl_bin_process(dataset_name, classify_type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)

    # 模型
    model = SAGNN(x_size, h_size, attn_drop, feat_drop, classify_type).to(device)
    print("\r\n========模型结构==========")
    print(model)

    # 学习率
    lr_scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if warmup:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # 模型训练
    print("\r\n=============开始训练模型==================")
    for epoch in range(epoch):
        training_loss = 0
        total_nodes = 0

        ######################
        # train the model    #
        ######################
        model.train()
        train_losses = []
        train_idxs = []
        for step, batch in enumerate(train_loader):

            train_idxs += batch.idx_batch
            batch_ast = batch.ast_batch.to(device)
            batch_cfg = batch.cfg_batch.to(device)
            batch_labels = batch.node_lables.to(device) # torch.Size([402, 2])
            n = batch_ast.number_of_nodes()
            h = th.zeros((n, h_size)).to(device)
            c = th.zeros((n, h_size)).to(device)

            logits = model(batch_cfg, batch_ast, h, c)  # torch.Size([402, 2])

            if loss_type == "focal_loss":
                loss = focal_loss(logits, batch_labels.float())
            else:
                loss = F.cross_entropy(logits, batch_labels.float())
            
            training_loss += loss.item() * batch_cfg.num_nodes()
            total_nodes += batch_cfg.num_nodes()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
        
        training_loss /= total_nodes
        print("\r\nEPOCH:{} training_loss:{}".format(epoch, training_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        _preds = []
        _labels = []
        _idxs = []
        for step, batch in enumerate(valid_loader):
            with torch.no_grad():
                
                _idxs += batch.idx_batch
                batch_ast = batch.ast_batch.to(device)
                batch_cfg = batch.cfg_batch.to(device)
                batch_labels = batch.node_lables.to(device)
                n = batch_ast.number_of_nodes()
                h = th.zeros((n, h_size)).to(device)
                c = th.zeros((n, h_size)).to(device)

                logits = model(batch_cfg, batch_ast, h, c)

                # 计算训练中的结果
                _preds += logits.argmax(1)
                _labels += batch_labels.argmax(1)
        
        calculate_metrics(_preds, _labels, _idxs, "VALIDATE", epoch)

        ######################
        # test the model #
        ######################
        model.eval()
        __preds = []
        __labels = []
        __idxs = []
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                
                __idxs += batch.idx_batch
                batch_ast = batch.ast_batch.to(device)
                batch_cfg = batch.cfg_batch.to(device)
                batch_labels = batch.node_lables.to(device)
                n = batch_ast.number_of_nodes()
                h = th.zeros((n, h_size)).to(device)
                c = th.zeros((n, h_size)).to(device)

                logits = model(batch_cfg, batch_ast, h, c)

                # 计算训练中的结果
                __preds += logits.argmax(1)
                __labels += batch_labels.argmax(1)
        
        save_flag, acc, recall, precision, f1 = calculate_metrics(__preds, __labels, __idxs, "TEST", epoch)
        if save_flag == 1:
            _pt_name = "model//{}_{}_{}_{}.pt".format(epoch, recall, precision, f1)
            torch.save(model.state_dict(), _pt_name)
        
    
