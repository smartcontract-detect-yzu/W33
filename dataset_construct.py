import json
import dgl 
import torch
import os
from infercode.client.infercode_client import InferCodeClient
from tqdm import tqdm

INFERCODE_FEATURE_SIZE = 100

def infercode_init():

    # Change from -1 to 0 to enable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    infercode = InferCodeClient(language="solidity")
    infercode.init_from_config()
    return infercode

def _map_ast_node_id(stmt_ast_nodes_maps, stmt_ast_label, infercode):
    """
    Nodes in the graph have consecutive IDs starting from 0.
    """
    dgl_node_id = 0
    dgl_nodes_content = torch.zeros(len(stmt_ast_nodes_maps), 100)
    dgl_nodes_label = torch.zeros(len(stmt_ast_nodes_maps), 2) # [no_vul, vul]
    dgl_nodes_map = {}
    
    for ast_node_id in stmt_ast_nodes_maps:
        if ast_node_id not in dgl_nodes_map:
            dgl_nodes_map[int(ast_node_id)] = dgl_node_id

            content = stmt_ast_nodes_maps[ast_node_id]["content"]
            v = infercode.encode([content])
            dgl_nodes_content[dgl_node_id] = torch.from_numpy(v[0])

            if stmt_ast_label == 1:
                dgl_nodes_label[dgl_node_id] = torch.tensor([0,1], dtype=torch.float32)
            else:
                dgl_nodes_label[dgl_node_id] = torch.tensor([1,0], dtype=torch.float32)

            dgl_node_id += 1

    return dgl_nodes_map, dgl_nodes_content, dgl_nodes_label


def _construct_dgl_graph(ast_json_file, infercode):

    dgl_graphs = []
    graphs_labels = []

    with open(ast_json_file) as f:
        ast_json = json.load(f)
        for stmt in ast_json:
            stmt_ast_json = ast_json[stmt]
            stmt_ast_nodes_maps = stmt_ast_json["nodes"]
            stmt_ast_edges = stmt_ast_json["edges"]
            stmt_ast_label = int(stmt_ast_json["vul"])

            if len(stmt_ast_edges) == 0:
                continue  # statement with only one node as return

            dgl_nodes_map, dgl_nodes_content, dgl_nodes_label = _map_ast_node_id(stmt_ast_nodes_maps, stmt_ast_label, infercode)

            src = []
            dst = []
            for edge in stmt_ast_edges:
                
                # NOTE: 边: 叶子节点->根节点（与原始AST边的方向相反）
                to_id = edge["to"]
                src.append(dgl_nodes_map[to_id])

                from_id = edge["from"]
                dst.append(dgl_nodes_map[from_id])
            
            # save the graph nodes and edges
            u, v = torch.tensor(src), torch.tensor(dst)
            g = dgl.graph((u, v))
            g.ndata['feature'] = dgl_nodes_content
            g.ndata['label'] = dgl_nodes_label  # 由于DGL不支持图级别特征，规避
            dgl_graphs.append(g)

            # save the graph lable
            graphs_labels.append(int(stmt_ast_json["vul"]))

        return dgl_graphs, graphs_labels


def _construct_dgl_graph_v2(ast_json_file, infercode):

    """
        对单独的function构建dgl graph数据
        [cfg, stmt1_ast, stmt2_ast,..., stmtN_ast]
        CFG节点数目与AST数目相等
          -- AST 的根节点必须在第一个
    """

    graphs_for_function = []
    cfg_node_id = 0
    cfg_stmt_id_map = {} # from stmt ast id to cfg dgl id
    stmt_ast_id_to_dgl_idx = {}
   
    with open(ast_json_file) as f:
        ast_json = json.load(f)

        cfg_src = []
        cfg_dst = []

        # 首先根据CFG的顺序为AST分配ID
        cfg_edges = ast_json["cfg_edges"]
        for cfg_edge_info in cfg_edges:
            from_id = int(cfg_edge_info["from"])
            to_id = int(cfg_edge_info["to"])

            if from_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[from_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = from_id
                cfg_node_id += 1

            if to_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[to_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = to_id
                cfg_node_id += 1
            
            cfg_src.append(cfg_stmt_id_map[from_id])
            cfg_dst.append(cfg_stmt_id_map[to_id])

        u, v = torch.tensor(cfg_src), torch.tensor(cfg_dst)    
        cfg = dgl.graph((u, v))
        graphs_for_function.append(cfg) # 第一个永远是CFG
        cfg_dgl_nodes_label = torch.zeros(len(cfg_stmt_id_map), 2)    
        
        
        # 根据CFG的顺序创建各 语句粒度 AST
        # NOTE: range 不包含cfg_node_id
        for idx in range(0, cfg_node_id):
            
            self_loop_flag = 0

            # NOTE: cfg_id = 0 ==> cfg的根节点
            stmt_ast_id = str(stmt_ast_id_to_dgl_idx[idx]) # 语句AST的根节点
            stmt_ast_json = ast_json[stmt_ast_id]
            stmt_ast_nodes_maps = stmt_ast_json["nodes"]
            stmt_ast_edges = stmt_ast_json["edges"]
            stmt_ast_label = int(stmt_ast_json["vul"])

            # 构建 STMT AST  DGL GRAPH
            # NOTE: 语句的AST的根节点ID必须是0
            stmt_ast_node_id_map = {}
            stmt_ast_node_id_map[int(stmt_ast_id)] = 0    
            
            # 特征向量初始化, 大小为 INFERCODE_FEATURE_SIZE = 100
            dgl_nodes_content = torch.zeros(len(stmt_ast_nodes_maps), INFERCODE_FEATURE_SIZE)

            # NOTE: 其它节点从1开始
            dgl_node_id = 1 
            for ast_node_id in stmt_ast_nodes_maps:
                
                content = stmt_ast_nodes_maps[ast_node_id]["content"]
                if len(content) == 0:
                    content = stmt_ast_nodes_maps[ast_node_id]["ast_type"]
                v = infercode.encode([content])

                if int(ast_node_id) not in stmt_ast_node_id_map:
                   stmt_ast_node_id_map[int(ast_node_id)] = dgl_node_id
                   dgl_nodes_content[dgl_node_id] = torch.from_numpy(v[0])
                   dgl_node_id += 1
                
                elif stmt_ast_node_id_map[int(ast_node_id)] == 0: 
                    dgl_nodes_content[0] = torch.from_numpy(v[0])
                
                else:
                    print("ERROR !!!!!")
            
            # print(stmt_ast_node_id_map)

            # add self loop ==> 在进行LSTM训练前通过DGL接口删除
            if len(stmt_ast_edges) == 0:
                self_loop_flag = 1
                stmt_ast_edges.append({"from":stmt_ast_id, "to":stmt_ast_id})

            src = []
            dst = []
            for edge in stmt_ast_edges:

                # NOTE: 边: 叶子节点->根节点（与原始AST边的方向相反）
                to_id = int(edge["to"])
                src.append(stmt_ast_node_id_map[to_id])

                from_id = int(edge["from"])
                dst.append(stmt_ast_node_id_map[from_id])

            # save the graph nodes and edges
            u, v = torch.tensor(src), torch.tensor(dst)
            stmt_g = dgl.graph((u, v))
            stmt_g.ndata['x'] = dgl_nodes_content

            # NOTE: 0节点是否为根节点的检测
            if not (stmt_g.out_degrees(torch.tensor([0])) == 0):
                if not self_loop_flag:
                    print("ERROR:!!!! ast的第一个节点必须是根节点")
            
            # 根据顺序压入stmt ast
            graphs_for_function.append(stmt_g)

            if stmt_ast_label == 1:
                cfg_dgl_nodes_label[idx] = torch.tensor([0,1], dtype=torch.float32) 
            else:
                cfg_dgl_nodes_label[idx] = torch.tensor([1,0], dtype=torch.float32) 

        cfg.ndata["label"] = cfg_dgl_nodes_label

        return graphs_for_function


def construct_dgl_graphs_for_sample(contract_sample_dir, infercode):

    function_cnt = 0
    sample_graphs = []
    sample_graphs_cnt = []

    all_samples = os.listdir(contract_sample_dir)
    for sample in all_samples:
        sample_ast_json = contract_sample_dir + sample + "//statement_ast_infos.json"
        
        if not os.path.exists(sample_ast_json):
            pass
        else:
            try:
                function_graphs = _construct_dgl_graph_v2(sample_ast_json, infercode)
                sample_graphs += function_graphs
                sample_graphs_cnt.append(len(function_graphs))
                function_cnt += 1
            except:
                pass
                continue
    
    return sample_graphs, sample_graphs_cnt, function_cnt


def construct_dgl_graphs_for_dataset(dataset_dir, infercode):

    graphs = []
    graphs_cnts = []
    total_function = 0

    all_contracts = os.listdir(dataset_dir)
    with tqdm(total=len(all_contracts)) as pbar:
        for contract in all_contracts:
            contract_sample_dir = dataset_dir + contract + "//sample//"
            
            # construct dgl graph and lables
            sample_graphs, sample_graphs_cnt, function_cnt = construct_dgl_graphs_for_sample(contract_sample_dir, infercode)

            # add to the list
            graphs += sample_graphs
            graphs_cnts += sample_graphs_cnt

            total_function += function_cnt
            pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
            pbar.update(1)
          
            if total_function > 10240:
                print("!!!Already collect max function samples")
                break
                
    # construct the dgl dataset bin file
    infos = {"graph_cnts": torch.tensor(graphs_cnts)}
    bin_file_name = "{}.bin".format(dataset_dir.split("//")[-2])
    print("!! Save the dataset into {}".format(bin_file_name))
    dgl.save_graphs(bin_file_name, graphs, infos)

if __name__ == '__main__':

    infercode = infercode_init()

    # ast_json_file = "dataset//resumable_loop//0x77c42a88194f81a17876fecce71199f48f0163c4//sample//Bitcoinrama-swapBack-4777//statement_ast_infos.json"
    # _construct_dgl_graph_v2(ast_json_file, infercode)

    # sample_dir = "dataset//reentrancy//0xffa3a0ff18078c0654b174cf6cb4c27699a4369e//sample//"
    # sample_graphs, sample_graph_lables = construct_dgl_graphs_for_sample(sample_dir, infercode)

    dataset_dir = "dataset//resumable_loop//"
    construct_dgl_graphs_for_dataset(dataset_dir, infercode)

    
    