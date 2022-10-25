"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils.internal import expand_as_pair
from dgl.nn.pytorch.conv.graphconv import GraphConv


class TGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(TGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.type_liner = nn.Linear(100, 64, bias=False)  # type = 100, feature = 64
        if isinstance(in_feats, tuple):

            print("WARINING !!!! 无用分支1")

            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.out_liner = nn.Linear(64, 32, bias=False)  # type = 100, feature = 64
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph.')

            if isinstance(feat, tuple):

                print("WARINING !!!! 无用分支2")

                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                feat = self.type_liner(feat) # 100压缩到64
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)

                if graph.is_block: # falise
                    print("WARINING !!!! 无用分支3")
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing: 节点本身的信息依赖cfg的自旋 ==> add self loop for cfg
            graph.update_all(fn.u_mul_e('feature', 'a', 'm'),  # 'm'= 'ft'*'a'
                             fn.sum('m', 'feature'))   # 'ft'= 'm1' + ...

            # rst = graph.dstdata['feature']
            rst = self.out_liner(graph.dstdata['feature'])

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']

            else:
                return rst

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


class SAGNN(nn.Module):
    
    def __init__(self,
                 x_size,
                 h_size,
                 attn_drop,
                 feat_drop,
                 classify_type,
                 gnn_type="gcn"):

        super(SAGNN, self).__init__()
        self.classify_type = classify_type
        self.type = gnn_type
        self.x_size = x_size
        self.tree_lstm = ChildSumTreeLSTMCell(x_size, h_size)
        self.gcn = GraphConv(in_feats=h_size, out_feats=32, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.tgat = TGATConv(in_feats=h_size, out_feats=32, num_heads=4, attn_drop=attn_drop, feat_drop=feat_drop, activation=nn.Softmax())
        
        self.linner1 = nn.Linear(32, 16)
        if self.classify_type == "binary":
            self.linner2 = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linner2 = nn.Linear(16, 2)

    def forward(self, cfg, ast, h, c):

        # feed embedding
        ast.ndata['iou'] = self.tree_lstm.W_iou(ast.ndata['x'])
        ast.ndata['h'] = h
        ast.ndata['c'] = c

        # propagate: do tree_lstm
        dgl.prop_nodes_topo(ast, self.tree_lstm.message_func, self.tree_lstm.reduce_func, apply_node_func=self.tree_lstm.apply_node_func)
        
        # copy the learned feature to cfg => cfg和ast的节点id一一对应，保持一致
        for idx, stmt_ast in enumerate(dgl.unbatch(ast)):
            cfg.ndata["feature"][idx].copy_(stmt_ast.ndata['h'][0]) # NOTE: 每个ast.nodes[0]就是语句的根节点 --> 由dataset_construct.py保证
        
        if self.type =="gcn": # GNN
            res = self.gcn(cfg, cfg.ndata['feature'])
            res = self.linner1(res)
            res = self.linner2(res)
        
        elif self.type =="tgat": # TGAT
            res = self.tgat(cfg, cfg.ndata['type'])
            res = self.linner1(res) # 32 16
            res = self.linner2(res) # 16 2
        
        if self.classify_type == "binary":
            res = self.sigmoid(res)

        return res