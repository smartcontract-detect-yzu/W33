import argparse
import collections
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


class MyDataset(DGLDataset):
    
    def __init__(self, cfg_graphs, ast_graphs):
        self.cfg_graphs = cfg_graphs  # [  cfg1,           cfg2, ...,              cfgn]
        self.ast_graphs = ast_graphs  # [[ast1_1, ast1_2], [ast2_1, ast2_2],..., [astn_1, astn_2, astn_3]]
        super().__init__(name='my Dataset')

    def process(self):
        pass

    def __getitem__(self, i):
        return self.cfg_graphs[i], self.ast_graphs[i]

    def __len__(self):
        return len(self.cfg_graphs)


my_batch = collections.namedtuple('my_batch', ['cfg_batch', 'ast_batch', "node_lables"])
def batcher_v1(device, classify_type):
    def batcher_v1_dev(batch):
        cfgs = []
        asts = []

        # [cfg_i], [ast_i_1, ast_i_2]
        for cfg_ast in batch:

            _cfg = cfg_ast[0]
            _cfg = dgl.add_self_loop(_cfg) # 在CFG中添加self loop
            cfgs.append(_cfg)

            for ast in cfg_ast[1]:
                ast = dgl.remove_self_loop(ast) # 在AST中删除self loop
                asts.append(ast)

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
                        node_lables=_node_lables)
                        

    return batcher_v1_dev


def dgl_bin_process(dataset_name, c_type):

    idx_graphs = [] # 图索引
    cfg_graphs = []
    ast_graphs = []

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

    print("====统计信息: total:{} train_size:{}, valid_size:{}, test_size:{}".format(
        len(cfg_graphs), train_size, valid_size, test_size)
    )

    train_idx_list = idx_graphs[0:train_size]  # 80%
    valid_idx_list = idx_graphs[train_size + 1: train_size + valid_size]  # 10%
    test_idx_list  = idx_graphs[train_size + valid_size + 1:]  # 10%
    
    # 根据划分的结果进行数据集构建
    train_cfg_graphs = []
    train_ast_graphs = []
    for tmp_id in train_idx_list:
        train_cfg_graphs.append(cfg_graphs[tmp_id])
        train_ast_graphs.append(ast_graphs[tmp_id])
    train_dataset = MyDataset(train_cfg_graphs, train_ast_graphs)
    
    valid_cfg_graphs = []
    valid_ast_graphs = []
    for tmp_id in valid_idx_list:
        valid_cfg_graphs.append(cfg_graphs[tmp_id])
        valid_ast_graphs.append(ast_graphs[tmp_id])
    valid_dataset = MyDataset(valid_cfg_graphs, valid_ast_graphs)
    
    test_cfg_graphs = []
    test_ast_graphs = []
    for tmp_id in test_idx_list:
        test_cfg_graphs.append(cfg_graphs[tmp_id])
        test_ast_graphs.append(ast_graphs[tmp_id])
    test_dataset = MyDataset(test_cfg_graphs, test_ast_graphs)

    return train_dataset, valid_dataset, test_dataset


def calculate_metrics(preds, labels, prefix, vul_label_is=1):

    useless_flag = 0
    TP = FP = TN = FN = 0
    vul_cnt = no_vul_cnt = 0

    print("{} preds:{}, laels:{}".format(prefix, len(preds), len(labels)))

    for pred, label in zip(preds, labels):
        if label == vul_label_is:
            vul_cnt += 1
            if pred == label:
                TP += 1
            else:
                FP += 1
        else:
            no_vul_cnt += 1
            if pred == label:
                TN += 1
            else:
                FN += 1

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
    
    # 9999代码表结果不可用
    print("USE:[{}]   ACC:{}, RE:{}, P:{}, F1:{}, TOTAL:{}".format(useless_flag, acc, recall, precision, f1, total_data_num))
    return acc, recall, precision, f1, total_data_num

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


if __name__ == '__main__':

    _dataset = argParse()

    # 参数列表
    torch.manual_seed(3407)
    dataset_name = "{}.bin".format(_dataset)
    batch_size = 512
    epoch = 256
    x_size = 100 # ast node feature size
    h_size = 64  # tree_lstm learning feature or CFG node feature size
    warmup = 1
    loss_type = "focal_loss" # focal_loss  cross_entropy
    classify_type = "multi" # binary multi 
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        th.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(device)
    
    if loss_type == "focal_loss":
        focal_loss = FocalLoss2d().to(device)

    # 数据
    train_dataset, valid_dataset, test_dataset = dgl_bin_process(dataset_name, classify_type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)

    # 模型
    model = SAGNN(x_size, h_size, classify_type).to(device)

    # 学习率
    lr_scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if warmup:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # 模型训练
    for epoch in range(epoch):
        training_loss = 0
        total_nodes = 0

        ######################
        # train the model    #
        ######################
        model.train()
        train_losses = []
        for step, batch in enumerate(train_loader):

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
        
        print("EPOCH:{} training_loss:{}".format(epoch, training_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        _preds = []
        _labels = []
        for step, batch in enumerate(valid_loader):
            with torch.no_grad():

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
        
        calculate_metrics(_preds, _labels, "VALIDATE")

        ######################
        # test the model #
        ######################
        model.eval()
        __preds = []
        __labels = []
        for step, batch in enumerate(test_loader):
            with torch.no_grad():

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
        
        calculate_metrics(__preds, __labels, "TEST")
    
