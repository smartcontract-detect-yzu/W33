import argparse
import collections
import json
import os
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
import logging


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

        # [cfg_i], [ast_i_1, ast_i_2], function_idx
        for cfg_ast in batch:

            _cfg = cfg_ast[0]
            _cfg = dgl.add_self_loop(_cfg) # 在CFG中添加self loop
            cfgs.append(_cfg)

            for idx, ast in enumerate(cfg_ast[1]): 
                ast = dgl.remove_self_loop(ast) # 在AST中删除self loop
                asts.append(ast)
                gidx.append("{}-{}".format(cfg_ast[2], idx)) # 函数ID-当前语句的ASTID
        
        cfg_batch = dgl.batch(cfgs) # batch:[1_2stmts,   2_3stmts,     3_2stmts] 
        ast_batch = dgl.batch(asts) # batch:[[1,2],      [1,2,3],      [1,2]   ] => [1,2,3,4,5,6,7]

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
    cfg_graphs = {} # cfg map
    ast_graphs = {} # ast map

    no_vul_cnt = 0
    total_cnt = 0

    # 抽取全部数据
    graphs, graphs_infos = dgl.load_graphs(dataset_name)
    for idx in range(0, len(graphs_infos["graph_cnts"])):

        tmp_ast_graphs = []
        graph_cnt = graphs_infos["graph_cnts"][idx].int()
        if graph_cnt == 0:
            raise RuntimeError("!!! 出现了没有图的样本")
        
        if str(idx) in BLACK_LIST: # 黑名单过滤
            for graph_idx in range(0, graph_cnt):
                graphs.pop(0) # 丢弃
            
        else:
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

                    # 双向
                    if bidir_cfg == 1:
                        tmp_graph = dgl.add_reverse_edges(tmp_graph)
                    
                    cfg_graphs[idx] = tmp_graph
                else:
                    tmp_ast_graphs.append(tmp_graph)
            
            ast_graphs[idx] = tmp_ast_graphs
            idx_graphs.append(idx)

    # 划分train/test/valid数据集
    random.shuffle(idx_graphs)
    train_size = int(0.8*len(idx_graphs))
    valid_size = int(0.1*len(idx_graphs))
    test_size  = len(idx_graphs) - train_size - valid_size

    logger.debug("\r\n====================统计信息===========================")
    logger.debug("==数据集:{}".format(dataset_name))
    logger.debug("==数量信息: total:{} train_size:{}, valid_size:{}, test_size:{}".format(
        len(cfg_graphs), train_size, valid_size, test_size)
    )
    logger.debug("==负标签分布: no_vul:{} {}".format(no_vul_cnt, no_vul_cnt/total_cnt))
    logger.debug("==正标签分布: vul:{} {}".format(total_cnt - no_vul_cnt, (total_cnt - no_vul_cnt)/total_cnt))

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
        logger.debug("==============FN SAMPLES    [Start...]===================")
        for cfgid_astid in fn_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')
            logger.debug("FN: path:{} STMTID:{} type:{}".format(
                DATASET_DB[cfg_id]["path"], 
                ast_id, 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"]))
        logger.debug("==============FN SAMPLES    [END...]===================")

    if len(fp_samples) > 0:
        logger.debug("==============FP SAMPLES    [Start...]===================")
        for cfgid_astid in fp_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')
            logger.debug("FP: path:{} STMTID:{} type:{}".format(
                DATASET_DB[cfg_id]["path"], 
                ast_id, 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"]))
        logger.debug("==============FP SAMPLES    [END...]===================\r\n")

def _do_countor(cfg_id, ast_id, _type):

    if cfg_id not in SAMPLE_COUNTOR[_type]:
        SAMPLE_COUNTOR[_type][cfg_id] = {"_cnt":1}
    else:
        SAMPLE_COUNTOR[_type][cfg_id]["_cnt"] += 1
    
    if ast_id not in SAMPLE_COUNTOR[_type][cfg_id]:
        SAMPLE_COUNTOR[_type][cfg_id][ast_id] = 1
    else:
        SAMPLE_COUNTOR[_type][cfg_id][ast_id] += 1

def wrong_sample_log_v2(fn_samples, fp_samples):
    
    if len(fn_samples) > 0:
        logger.debug("==============FN SAMPLES:{}    [Start...]===================".format(len(fn_samples)))
        for cfgid_astid in fn_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')
            _do_countor(cfg_id, ast_id, "fn")

            sample_infos = str(DATASET_DB[cfg_id]["path"]).split("//")
            logger.debug("FN: {} {} ASTID:{} TYPE:{}".format(
                sample_infos[2], 
                sample_infos[4], 
                DATASET_DB[cfg_id][ast_id]["ASTID"], 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"])
            )
        logger.debug("==============FN SAMPLES    [END...]===================")

    if len(fp_samples) > 0:
        logger.debug("==============FP SAMPLES:{}    [Start...]===================".format(len(fp_samples)))
        for cfgid_astid in fp_samples:
            cfg_id, ast_id = str(cfgid_astid).split('-')
            _do_countor(cfg_id, ast_id, "fp")

            sample_infos = str(DATASET_DB[cfg_id]["path"]).split("//")
            logger.debug("FP: {} {} ASTID:{} TYPE:{}".format(
                sample_infos[2], 
                sample_infos[4], 
                DATASET_DB[cfg_id][ast_id]["ASTID"], 
                DATASET_DB[cfg_id][ast_id]["vul_tpye"])
            )
        logger.debug("==============FP SAMPLES    [END...]===================\r\n")

def calculate_metrics_v2(y_pred, y_true, sample_idxs, prefix, epoch):
    """
        基于tensor的计算且在GPU内, 速度快于原始版本
    """
    useless_flag = 0 
    y_pred = y_pred.argmax(dim=1)
    y_true = y_true.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    
    fp_samples = (1 - y_true) * y_pred   # fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fp = fp_samples.sum().to(torch.float32)
    if fp.item() > 0:
        fp_idxs = (fp_samples==1).nonzero().squeeze(dim=1).cpu().numpy().tolist()
    else:
        useless_flag = 1
        fp_idxs = None

    fn_samples = y_true * (1 - y_pred)   # fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    fn = fn_samples.sum().to(torch.float32)
    if fn.item() > 0:
        fn_idxs = (fn_samples==1).nonzero().squeeze(dim=1).cpu().numpy().tolist()
    else:
        useless_flag = 1
        fn_idxs = None

    epsilon = 1e-7
    
    a = (tp + tn) / (tp + tn + fp + fn + epsilon)
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2* (p*r) / (p + r + epsilon)

    logger.debug("[epoch:{}] - {}阶段指标: A: {} P: {} R: {} F1: {}".format(epoch, prefix, a.item(), p.item(), r.item(), f1.item()))

    if useless_flag != 1 and f1 >= 0.80:
        fn_samples_idx = [sample_idxs[idx] for idx in fn_idxs]
        fp_smaples_idx = [sample_idxs[idx] for idx in fp_idxs]
        wrong_sample_log_v2(fn_samples_idx, fp_smaples_idx)
        return 1, p.item(), r.item(), f1.item()
    else:
        return 0, p.item(), r.item(), f1.item()
    

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
    
    logger.debug("{}:{}: EPOCH:{} ACC:{}, RE:{}, P:{},  F1:{}, TOTAL:{}".format(
            prefix, useless_flag, epoch, acc, recall, precision, f1, total_data_num)
        )
    
    # if useless_flag == 0 and recall >= 0.90 and precision >= 0.70 and f1 >= 0.80:
    if (epoch > 0 and epoch % 10 ==0) or (useless_flag == 0 and f1 >= 0.75):
        save_flag = 1
        logger.debug("\r\n========================错误日志=====================================")
        logger.debug("{}:{}: EPOCH:{} ACC:{}, RE:{}, P:{},  F1:{}, TOTAL:{}".format(
            prefix, useless_flag, epoch, acc, recall, precision, f1, total_data_num)
        )
        wrong_sample_log(fn_samples, fp_samples)
        logger.debug("==TPR:{}, FNR:{}, FPR:{}, TNR:{}".format(TPR, FNR, FPR, TNR))

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
    logger.debug("\r\n=========模型情况打印:===========")
    logger.debug("==my_seed: %s", my_seed)
    logger.debug("==batch_size: %s", batch_size)
    logger.debug("==epoch: %s", epoch)
    logger.debug("==x_size: %s", x_size)
    logger.debug("==h_size: %s", h_size)
    logger.debug("==attn_drop: %s", attn_drop)
    logger.debug("==feat_drop: %s", feat_drop)
    logger.debug("==warmup: %s", warmup)
    logger.debug("==weight_decay: %s", weight_decay)
    logger.debug("==loss_type: %s", loss_type)
    logger.debug("==classify_type: %s", classify_type)
    logger.debug("==gnn_type: %s", gnn_type)
    logger.debug("==metic_calc: %s", metic_calc)
    logger.debug("==device: %s", device_name)
    logger.debug("==黑名单: %d", len(BLACK_LIST))
    logger.debug("==bidir_cfg: %d", bidir_cfg)

if __name__ == '__main__':

    _dataset = argParse()
    if str(_dataset).endswith(".bin"):
        dataset_name = "dataset//" + _dataset
        dataset_db_file = "dataset//" + str(_dataset).split(".bin")[0] + "_db.json"
        dataset_blacklist = "dataset//" + str(_dataset).split(".bin")[0] + "_black_list.json"
    else:
        dataset_name = "dataset//" + "{}.bin".format(_dataset)
        dataset_db_file = "dataset//" + _dataset + "_db.json"
        dataset_blacklist = "dataset//" + _dataset + "_black_list.json"

    f = open(dataset_db_file, "r")
    DATASET_DB = json.load(f)

   

    # 日志初始化
    _time_stamp = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    log_file_name = "train_log//{}_{}.log".format(_dataset, _time_stamp)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_name, "w+")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)

    SAMPLE_COUNTOR = {}
    SAMPLE_COUNTOR["fn"] = {}
    SAMPLE_COUNTOR["fp"] = {}
    col_json_name = "train_log//{}_{}_collecotor.json".format(_dataset, _time_stamp)

    BLACK_LIST = {}
    if os.path.exists(dataset_blacklist):
        logger.debug(">>>>>>>>>>开启黑名单")
        BLACK_LIST = json.load(open(dataset_blacklist, "r"))

    # 参数列表
    my_seed = 3407
    batch_size = 1024
    epoch = 256
    x_size = 100 # ast node feature size
    h_size = 64  # tree_lstm learning feature or CFG node feature size
    attn_drop = 0.05
    feat_drop = 0.05
    warmup = 1
    weight_decay = 0  # 1e-4
    loss_type = "cross_entropy" # focal_loss  cross_entropy
    classify_type = "multi" # binary multi 
    gnn_type = "tgat" # tgat gcn
    metic_calc = "v2"  # v1 v2
    bidir_cfg = 1    # CFG是否使用双向边


    if torch.cuda.is_available():
        device_name = "cuda:0"
        device = torch.device("cuda:0")
        th.cuda.set_device(device)
    else:
        device_name = "cpu"
        device = torch.device("cpu")
    
    debug_model_infos()
    
    # 模型初始化
    torch.manual_seed(my_seed)
    if loss_type == "focal_loss":
        focal_loss = FocalLoss2d().to(device)

    # 数据
    train_dataset, valid_dataset, test_dataset = dgl_bin_process(dataset_name, classify_type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=batcher_v1(device, classify_type), shuffle=False, num_workers=0)

    # 模型
    model = SAGNN(x_size, h_size, attn_drop, feat_drop, classify_type, gnn_type).to(device)
    logger.debug("\r\n========模型结构==========")
    logger.debug(model) # 需要记录在日志文件中

    # 学习率
    lr_scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    if warmup:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # 模型训练
    logger.debug("\r\n=============开始训练模型==================")
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
        logger.debug("\r\nEPOCH:{} training_loss:{}".format(epoch, training_loss))

        ######################
        # validate the model #
        ######################
        model.eval()

        if metic_calc == "v1":
            _preds = []
            _labels = []

        elif metic_calc == "v2":
            _preds = None
            _labels = None

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
                if metic_calc == "v1":
                    _preds += logits.argmax(1)
                    _labels += batch_labels.argmax(1) 

                elif metic_calc == "v2":
                    if step == 0:
                        _preds = logits
                        _labels = batch_labels
                    else:
                        _preds = torch.cat((_preds, logits), 0)
                        _labels = torch.cat((_labels, batch_labels), 0)
               
        
        if metic_calc == "v1":
            calculate_metrics(_preds, _labels, _idxs, "VALIDATE", epoch)

        elif metic_calc == "v2":
            calculate_metrics_v2(_preds, _labels, _idxs, "VALIDATE", epoch)

        ######################
        # test the model #
        ######################
        model.eval()
        if metic_calc == "v1":
            __preds = []
            __labels = []
            
        elif metic_calc == "v2":
            __preds = None
            __labels = None

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
                if metic_calc == "v1":
                    __preds += logits.argmax(1)
                    __labels += batch_labels.argmax(1) 

                elif metic_calc == "v2":
                    if step == 0:
                        __preds = logits
                        __labels = batch_labels
                    else:
                        __preds = torch.cat((__preds, logits), 0)
                        __labels = torch.cat((__labels, batch_labels), 0)
        
        if metic_calc == "v1":
            calculate_metrics(__preds, __labels, __idxs, "TEST", epoch)

        elif metic_calc == "v2":
            flag, p ,r, f1 = calculate_metrics_v2(__preds, __labels, __idxs, "TEST", epoch)
            if flag == 1:
                _pt_name = "model//{}_{}_{}_{}_{}.pt".format(_dataset, epoch, p, r, f1)
                torch.save(model.state_dict(), _pt_name)
                
        # save_flag, acc, recall, precision, f1 = calculate_metrics(__preds, __labels, __idxs, "TEST", epoch)
        # if save_flag == 1:
        #     _pt_name = "model//{}_{}_{}_{}.pt".format(epoch, recall, precision, f1)
        #     torch.save(model.state_dict(), _pt_name)
        
    # 保存计数器结果
    with open(col_json_name, "w+") as f:
        f.write(json.dumps(SAMPLE_COUNTOR, indent=4,  separators=(",", ":")))
