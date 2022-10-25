"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import collections
import random
import dgl
from dgl.data import DGLDataset
import numpy as np
import torch
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.nn.pytorch.conv.graphconv import GraphConv
from dgl.nn.pytorch.conv import GATConv
from tgat import TGATConv

class ChildSumTreeLSTMCell(nn.Module):

    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f_t = self.U_f(nodes.mailbox['h'])
        f = th.sigmoid(f_t)
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class SAGnn(nn.Module):
    
    def __init__(self,
                 x_size,
                 h_size,
                 gnn_type="gcn"):

        super(SAGnn, self).__init__()
        self.type = gnn_type
        self.x_size = x_size
        self.tree_lstm = ChildSumTreeLSTMCell(x_size, h_size)
        self.gcn = GraphConv(in_feats=h_size, out_feats=32, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.tgat = TGATConv(in_feats=h_size, out_feats=32, num_heads=3, attn_drop=0.05, feat_drop=0.05, activation=nn.ReLU())

        self.linner1 = nn.Linear(32, 16)
        self.linner2 = nn.Linear(16, 2)

    def forward(self, cfg, ast, h, c):

        # feed embedding
        ast.ndata['iou'] = self.tree_lstm.W_iou(ast.ndata['x'])
        ast.ndata['h'] = h
        ast.ndata['c'] = c

        # propagate: do tree_lstm
        dgl.prop_nodes_topo(ast, self.tree_lstm.message_func, self.tree_lstm.reduce_func, apply_node_func=self.tree_lstm.apply_node_func)
        
        # copy the learned feature to cfg
        for idx, stmt_ast in enumerate(dgl.unbatch(ast)):
            cfg.ndata["feature"][idx].copy_(stmt_ast.ndata['h'][0]) # NOTE: 每个ast.nodes[0]就是语句的根节点 --> 由dataset_construct.py保证
        
        
        if self.type =="gcn": # GNN
            res = self.gcn(cfg, cfg.ndata['feature'])
            res = self.linner1(res)
            res = self.linner2(res)
        
        elif self.type =="tgat": # TGAT
            res = self.tgat(cfg, cfg.ndata['type'])
            res = self.linner1(res)
            res = self.linner2(res)
        
        return res

def dgl_bin_process():
    
    graph_idx = [] # 图索引
        graphs, graphs_infos = dgl.load_graphs("test_1.bin")
        for idx in range(0, len(graphs_infos["graph_cnts"])):
            tmp_ast_graphs = []
        
            graph_cnt = graphs_infos["graph_cnts"][idx].int()
            for graph_idx in range(0, graph_cnt):

                tmp_graph = graphs.pop(0)
                if graph_idx == 0:   # 第一个是CFG, 其它的是AST
                    tmp_graph.ndata["feature"] = th.zeros(tmp_graph.num_nodes(), 64) # 为CFG开辟一个特征空间,大小为TREE_LSTM的输出
                    self.cfg_graphs.append(tmp_graph)
                else:
                    tmp_ast_graphs.append(tmp_graph)
            
            self.ast_graphs.append(tmp_ast_graphs)

            graph_idx.append(idx)
        
        # 划分train/test/valid数据集
        random.shuffle(graph_idx)
        train_size = int(0.8*len(graph_idx))
        valid_size = int(0.1*len(graph_idx))

        train_idx_list = graph_idx[0:train_size]
        valid_idx_list = graph_idx[train_size + 1: train_size + valid_size]
        test_idx_list  = graph_idx[train_size + valid_size + 1:]

        train_cfg_graphs = []
        train_ast_graphs = []
        for tmp_id in train_idx_list:
            train_cfg_graphs.append(self.cfg_graphs[tmp_id])
            train_ast_graphs.append(self.ast_graphs[tmp_id])
        
        valid_cfg_graphs = []
        valid_ast_graphs = []
        for tmp_id in valid_idx_list:
            valid_cfg_graphs.append(self.cfg_graphs[tmp_id])
            valid_ast_graphs.append(self.ast_graphs[tmp_id])
        
        test_cfg_graphs = []
        test_ast_graphs = []
        for tmp_id in test_idx_list:
            test_cfg_graphs.append(self.cfg_graphs[tmp_id])
            test_ast_graphs.append(self.ast_graphs[tmp_id])


class CfgDataset(DGLDataset):
    def __init__(self):
        self.cfg_graphs = []  # [  cfg1,           cfg2, ...,              cfgn]
        self.ast_graphs = []  # [[ast1_1, ast1_2], [ast2_1, ast2_2],..., [astn_1, astn_2, astn_3]]
        super().__init__(name='Cfg_Dataset')

    def process(self):
        """
            [cfg1, ast1_1, ast1_2, cfg2, ast2_1, cfg3, ast3_1, ast3_2, ...]
        """
        graph_idx = [] # 图索引
        graphs, graphs_infos = dgl.load_graphs("test_1.bin")
        for idx in range(0, len(graphs_infos["graph_cnts"])):
            tmp_ast_graphs = []
        
            graph_cnt = graphs_infos["graph_cnts"][idx].int()
            for graph_idx in range(0, graph_cnt):

                tmp_graph = graphs.pop(0)
                if graph_idx == 0:   # 第一个是CFG, 其它的是AST
                    tmp_graph.ndata["feature"] = th.zeros(tmp_graph.num_nodes(), 64) # 为CFG开辟一个特征空间,大小为TREE_LSTM的输出
                    self.cfg_graphs.append(tmp_graph)
                else:
                    tmp_ast_graphs.append(tmp_graph)
            
            self.ast_graphs.append(tmp_ast_graphs)

            graph_idx.append(idx)
        
        # 划分train/test/valid数据集
        random.shuffle(graph_idx)
        train_size = int(0.8*len(graph_idx))
        valid_size = int(0.1*len(graph_idx))

        train_idx_list = graph_idx[0:train_size]
        valid_idx_list = graph_idx[train_size + 1: train_size + valid_size]
        test_idx_list  = graph_idx[train_size + valid_size + 1:]

        train_cfg_graphs = []
        train_ast_graphs = []
        for tmp_id in train_idx_list:
            train_cfg_graphs.append(self.cfg_graphs[tmp_id])
            train_ast_graphs.append(self.ast_graphs[tmp_id])
        
        valid_cfg_graphs = []
        valid_ast_graphs = []
        for tmp_id in valid_idx_list:
            valid_cfg_graphs.append(self.cfg_graphs[tmp_id])
            valid_ast_graphs.append(self.ast_graphs[tmp_id])
        
        test_cfg_graphs = []
        test_ast_graphs = []
        for tmp_id in test_idx_list:
            test_cfg_graphs.append(self.cfg_graphs[tmp_id])
            test_ast_graphs.append(self.ast_graphs[tmp_id])
            

    def __getitem__(self, i):
        return self.cfg_graphs[i], self.ast_graphs[i]

    def __len__(self):
        return len(self.cfg_graphs)

class StmtDataset(DGLDataset):
    
    def __init__(self):
        super().__init__(name='statement_dataset')

    def process(self):
        self.graphs = []
        self.labels = []
        
        graphs, label_dict = dgl.load_graphs("test.bin")
        self.graphs += graphs
        self.labels += label_dict["glabel"]

        random.shuffle(self.graphs)
        
        print("TOTAL: {}".format(len(self.graphs)))

        train_size = int(0.8*len(self.graphs))
        train_dataset = self.graphs[0:train_size]

        valid_size = int(0.1*len(self.graphs))
        valid_dataset = self.graphs[train_size + 1: train_size + valid_size]

        test_dataset = self.graphs[train_size + valid_size + 1: ]
        
        return train_dataset, valid_dataset, test_dataset
        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


StmtAstBatch = collections.namedtuple('StmtAstBatch', ['graph', 'feature', 'label', "size"])
def batcher(device):
    def batcher_dev(batch):
        labels = torch.zeros(len(batch), 2, dtype=torch.float32)
        for idx, graph in enumerate(batch):
            labels[idx] = graph.ndata["label"][0]
        batch_trees = dgl.batch(batch)
        return StmtAstBatch(graph=batch_trees, feature=batch_trees.ndata['feature'].to(device), label=labels.to(device), 
                            size=batch_trees.batch_size)   
    return batcher_dev


my_batch = collections.namedtuple('my_batch', ['cfg_batch', 'ast_batch', "node_lables"])
def batcher_v1(device):
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

        cfg_batch = dgl.batch(cfgs)
        ast_batch = dgl.batch(asts) # batch:[[],[],[]] => []

        return my_batch(cfg_batch=cfg_batch,
                        ast_batch=ast_batch,
                        node_lables=cfg_batch.ndata['label'].to(device))
                        

    return batcher_v1_dev

def calculate_metrics(preds, labels, vul_label_is=1):
    """
    
    """
    TP = FP = TN = FN = 0
    vul_cnt = no_vul_cnt = 0

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
        recall = 9999

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 9999

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 9999
    
    # 9999代码表结果不可用
    print("ACC:{}, RE:{}, P:{}, F1:{},TOTAL:{}".format(acc, recall, precision, f1, total_data_num))
    return acc, recall, precision, f1, total_data_num


def do_test_model(model, dataset_loader, device):

    # 模型测试
    with torch.no_grad():
        model.eval()
        predicts = []
        labels = []
        for _, batch in enumerate(dataset_loader):  
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, 150)).to(device)  # [number_of_nodes * h_size(is 150)]
            c = th.zeros((n, 150)).to(device)

            logits = model(g, h, c)
            # logits = F.log_softmax(logits, 1)
            
            predicts += logits.argmax(dim=1)
            labels += batch.label.argmax(dim=1)

        acc, recall, precision, f1, total_data_num = calculate_metrics(predicts, labels)
        return acc, recall, precision, f1, total_data_num

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x): # x是step次数
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters # 当前进度 0-1
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

warmup = 1
if __name__ == '__main__':

    torch.manual.seed(3407)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        th.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    print(device)

    train_dataset = CfgDataset()

    batch_size = 256
    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        collate_fn=batcher_v1(device),
                        shuffle=True,
                        num_workers=0)
    
    
   
    epoch = 256
    x_size = 100
    h_size = 64
    model = SAGnn(x_size, h_size).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = None
    if warmup:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    # 模型训练
    for epoch in range(epoch):
        model.train()

        training_loss = 0
        total_nodes = 0
        
        for step, batch in enumerate(train_loader):

            batch_ast = batch.ast_batch.to(device)
            batch_cfg = batch.cfg_batch.to(device)
            batch_labels = batch.node_lables.to(device)
            n = batch_ast.number_of_nodes()
            h = th.zeros((n, h_size)).to(device)
            c = th.zeros((n, h_size)).to(device)

            logits = model(batch_cfg, batch_ast, h, c)
            # 计算训练中的结果
            # preds = logits.argmax(1)
            # labels = batch_labels.argmax(1)
            # calculate_metrics(preds, labels)

            # logits = F.log_softmax(logits, 1)  # log_softmax(logits, 1)
            loss = F.cross_entropy(logits, batch_labels.long())
            # loss = F.nll_loss(logp, batch.label, reduction='sum')
            training_loss += loss.item() * batch_cfg.num_nodes()
            total_nodes += batch_cfg.num_nodes()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
        training_loss /= total_nodes
        print("EPOCH:{} training_loss:{}".format(epoch, training_loss))

                    