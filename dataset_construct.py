import argparse
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

    """
        对单独的function构建dgl graph数据
        [cfg, stmt1_ast, stmt2_ast,..., stmtN_ast]
        CFG节点数目与AST数目相等
          -- AST 的根节点必须在第一个
    """

    graphs_for_function = []
    infos_for_function = {"path":ast_json_file}

    cfg_node_id = 0
    cfg_stmt_id_map = {} # from stmt ast id to cfg dgl id
    stmt_ast_id_to_dgl_idx = {}
    
    with open(ast_json_file) as f:
        ast_json = json.load(f)

        cfg_src = []
        cfg_dst = []

        # 为CFG各节点分配ID
        cfg_edges = ast_json["cfg_edges"]
        if len(cfg_edges) == 0:
            return graphs_for_function, infos_for_function  # NOTE: 返回空, 需要在外部跳过, 不能保存

        for cfg_edge_info in cfg_edges:
            from_id = str(cfg_edge_info["from"])
            to_id = str(cfg_edge_info["to"])

            if from_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[from_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = from_id
                infos_for_function[cfg_node_id] = {"ASTID":from_id}
                cfg_node_id += 1
            
            if to_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[to_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = to_id
                infos_for_function[cfg_node_id] = {"ASTID":to_id}
                cfg_node_id += 1
            
            cfg_src.append(cfg_stmt_id_map[from_id])
            cfg_dst.append(cfg_stmt_id_map[to_id])

        u, v = torch.tensor(cfg_src), torch.tensor(cfg_dst)    
        cfg = dgl.graph((u, v))
        graphs_for_function.append(cfg) # 第一个永远是CFG
        cfg_dgl_nodes_label = torch.zeros(len(cfg_stmt_id_map), 2, dtype=torch.float) 
        cfg_dgl_nodes_types = torch.zeros(len(cfg_stmt_id_map), INFERCODE_FEATURE_SIZE)   
        
        
        # 保序: 根据CFG的顺序创建各 语句粒度 AST ==> CFG与AST保序压入数组
        for idx in range(0, cfg_node_id):
            
            self_loop_flag = 0

            # NOTE: cfg_id = 0 ==> cfg的根节点
            stmt_ast_id = str(stmt_ast_id_to_dgl_idx[idx]) # 语句AST的根节点
            stmt_ast_json = ast_json[stmt_ast_id]
            stmt_ast_nodes_maps = stmt_ast_json["nodes"]
            stmt_ast_edges = stmt_ast_json["edges"]
            stmt_ast_label = int(stmt_ast_json["vul"])
            stmt_vul_type = stmt_ast_json["vul_type"]

            infos_for_function[idx]["vul_tpye"] = stmt_vul_type
            infos_for_function[idx]["label"] = stmt_ast_label

            if "stmt_type" in stmt_ast_json:
                stmt_cfg_type  = stmt_ast_json["stmt_type"]

            elif stmt_ast_id == "exit":
                stmt_cfg_type  = "EXIT_POINT"

            else:
                raise RuntimeError("没有stmt_type, 并且不是EXIT")

            # 构建 STMT AST  DGL GRAPH
            stmt_ast_node_id_map = {}
            stmt_ast_node_id_map[stmt_ast_id] = 0  # NOTE: 语句的AST的根节点ID必须是0
            
            # 特征向量初始化, 大小为 INFERCODE_FEATURE_SIZE = 100
            dgl_nodes_content = torch.zeros(len(stmt_ast_nodes_maps), INFERCODE_FEATURE_SIZE)

            # NOTE: 其它节点从1开始
            dgl_node_id = 1
            for ast_node_id in stmt_ast_nodes_maps:
                
                content = stmt_ast_nodes_maps[ast_node_id]["content"]
                if len(content) == 0:
                    content = stmt_ast_nodes_maps[ast_node_id]["ast_type"]
                v = infercode.encode([content])

                if ast_node_id not in stmt_ast_node_id_map:
                   stmt_ast_node_id_map[ast_node_id] = dgl_node_id
                   dgl_nodes_content[dgl_node_id] = torch.from_numpy(v[0])
                   dgl_node_id += 1
                
                elif stmt_ast_node_id_map[ast_node_id] == 0: 
                    dgl_nodes_content[0] = torch.from_numpy(v[0])
                
                else:
                    print("ERROR !!!!!")
            
            # add self loop ==> 在进行LSTM训练前通过DGL接口删除
            if len(stmt_ast_edges) == 0:
                self_loop_flag = 1
                stmt_ast_edges.append({"from":stmt_ast_id, "to":stmt_ast_id})

            src = []
            dst = []
            for edge in stmt_ast_edges:

                # NOTE: 边: 叶子节点->根节点（与原始AST边的方向相反）
                src.append(stmt_ast_node_id_map[str(edge["to"])])
                dst.append(stmt_ast_node_id_map[str(edge["from"])])

            # save the graph nodes and edges
            u, v = torch.tensor(src), torch.tensor(dst)
            stmt_g = dgl.graph((u, v))
            stmt_g.ndata['x'] = dgl_nodes_content

            # NOTE: 0节点是否为根节点的检测, 反向AST, 计算out_degree
            if not (stmt_g.out_degrees(torch.tensor([0])) == 0):
                if not self_loop_flag:
                    print("ERROR:!!!! ast的第一个节点必须是根节点")
                    raise RuntimeError("ERROR:!!!! ast的第一个节点必须是根节点")
            
            # 根据顺序压入stmt ast
            graphs_for_function.append(stmt_g)

            if stmt_ast_label == 1:
                cfg_dgl_nodes_label[idx] = torch.tensor([0,1], dtype=torch.float) 
            else:
                cfg_dgl_nodes_label[idx] = torch.tensor([1,0], dtype=torch.float) 
            
            _type_v = infercode.encode([stmt_cfg_type])
            cfg_dgl_nodes_types[idx] = torch.from_numpy(_type_v[0])

        cfg.ndata["label"] = cfg_dgl_nodes_label
        cfg.ndata["type"] = cfg_dgl_nodes_types

        return graphs_for_function, infos_for_function


def construct_dgl_graphs_for_sample(contract_sample_dir, infercode, check):

    function_cnt = 0
    sample_graphs = []
    sample_graphs_cnt = []
    smaple_infos = []

    all_samples = os.listdir(contract_sample_dir)
    for sample in all_samples:
        sample_ast_json = contract_sample_dir + sample + "//statement_ast_infos.json"
        sample_dgl_done_flag = contract_sample_dir + sample + "//dgl_done_flag.flag"

        if not os.path.exists(sample_ast_json):
            pass

        elif check == 1 and os.path.exists(sample_dgl_done_flag):
            function_cnt += 1
            pass
        
        else:
            try:
                function_graphs, function_infos = _construct_dgl_graph(sample_ast_json, infercode)
                # print(json.dumps(function_infos, indent=4,  separators=(",", ":")))
                if len(function_graphs) != 0:
                    sample_graphs += function_graphs
                    sample_graphs_cnt.append(len(function_graphs))  # 1 + number_of_AST
                    smaple_infos.append(function_infos)
                    function_cnt += 1
                with open(sample_dgl_done_flag, "w+") as f:
                    f.write("dgl_done")
            except:
                print("ERROR: !!!!!!!:", sample_ast_json)
                if check != 1:  # 检测模式下不抛出异常
                    raise RuntimeError("!!!!!!!!!!!!!!!!!")
                    
        
    return sample_graphs, sample_graphs_cnt, smaple_infos, function_cnt


def construct_dgl_graphs_for_dataset(dataset_dir, infercode, check):

    graphs = []
    graphs_cnts = []
    graph_infos = []
    total_function = 0

    print(">>>>>>>>>>>>>>开始构建数据集:{}<<<<<<<<<<<<<<<".format(dataset_dir))

    all_contracts = os.listdir(dataset_dir)
    with tqdm(total=len(all_contracts)) as pbar:
        for contract in all_contracts:
            
            # 必须是文件夹, 且文件夹存在construct_done.flag标志
            if str(contract).endswith(".json") or not os.path.exists(dataset_dir + contract + "//construct_done.flag"): 
                pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
                pbar.update(1)
                continue

            # construct dgl graph and lables
            contract_sample_dir = dataset_dir + contract + "//sample//"
            sample_graphs, sample_graphs_cnt, smaple_infos, function_cnt = construct_dgl_graphs_for_sample(contract_sample_dir, infercode, check)

            # add to the list
            graphs += sample_graphs
            graphs_cnts += sample_graphs_cnt
            graph_infos += smaple_infos

            total_function += function_cnt
            pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
            pbar.update(1)

            if total_function > 10240:
                print("!!!Already collect max function samples")
                break

    if check == 1:
        print("==total_function is :{}".format(total_function))
        print("==total ast and cfg is:{}".format(len(graphs)))

    else:
        if check == 2:
            content_file = dataset_dir + "table_of_contents.json"
            with open(content_file, "w+") as f:
                table_of_contents = {}
                for idx, sample_name in enumerate(graph_infos):
                    table_of_contents[str(idx)] = sample_name
                f.write(json.dumps(table_of_contents, indent=4,  separators=(",", ":")))

        # construct the dgl dataset bin file
        infos = {"graph_cnts": torch.tensor(graphs_cnts)}
        bin_file_name = "{}_{}_{}.bin".format(dataset_dir.split("//")[-2], total_function, len(graphs))
        print("!! Save the dataset into {}, 图的总体数量为:{}".format(bin_file_name, total_function))
        dgl.save_graphs(bin_file_name, graphs, infos)

def _static_dgl_graph(ast_json_file):

    """
        对单独的function构建dgl graph数据
        [cfg, stmt1_ast, stmt2_ast,..., stmtN_ast]
        CFG节点数目与AST数目相等
          -- AST 的根节点必须在第一个
    """
    infos_for_function = {"path":ast_json_file}
    cfg_node_id = 0
    cfg_stmt_id_map = {} # from stmt ast id to cfg dgl id
    stmt_ast_id_to_dgl_idx = {}
    
    with open(ast_json_file) as f:
        ast_json = json.load(f)

        # 为CFG各节点分配ID
        cfg_edges = ast_json["cfg_edges"]
        if len(cfg_edges) == 0:
            return False, infos_for_function

        for cfg_edge_info in cfg_edges:
            from_id = str(cfg_edge_info["from"])
            to_id = str(cfg_edge_info["to"])

            if from_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[from_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = from_id
                infos_for_function[cfg_node_id] = {"ASTID":from_id}
                cfg_node_id += 1
            
            if to_id not in cfg_stmt_id_map:
                cfg_stmt_id_map[to_id] = cfg_node_id
                stmt_ast_id_to_dgl_idx[cfg_node_id] = to_id
                infos_for_function[cfg_node_id] = {"ASTID":to_id}
                cfg_node_id += 1
        
        for idx in range(0, cfg_node_id):

            stmt_ast_id = str(stmt_ast_id_to_dgl_idx[idx]) # 语句AST的根节点
            stmt_ast_json = ast_json[stmt_ast_id]
            stmt_ast_label = int(stmt_ast_json["vul"])
            stmt_vul_type = stmt_ast_json["vul_type"]

            infos_for_function[idx]["vul_tpye"] = stmt_vul_type
            infos_for_function[idx]["label"] = stmt_ast_label
        
        return True, infos_for_function


def static_dgl_graphs_for_sample(contract_sample_dir):

    function_cnt = 0
    smaple_infos = []

    all_samples = os.listdir(contract_sample_dir)
    for sample in all_samples:
        sample_ast_json = contract_sample_dir + sample + "//statement_ast_infos.json"

        if not os.path.exists(sample_ast_json):
            pass
        else:
            flag, function_infos = _static_dgl_graph(sample_ast_json)
            if flag == True:
                smaple_infos.append(function_infos)
                function_cnt += 1
    
    return smaple_infos, function_cnt

def static_dgl_graphs_for_dataset(dataset_dir):
    graph_infos = []
    total_function = 0

    print(">>>>>>>>>>>>>>构建数据集database文件:{}<<<<<<<<<<<<<<<".format(dataset_dir))
    all_contracts = os.listdir(dataset_dir)
    with tqdm(total=len(all_contracts)) as pbar:
        for contract in all_contracts:
            
            # 必须是文件夹, 且文件夹存在construct_done.flag标志
            if str(contract).endswith(".json") or not os.path.exists(dataset_dir + contract + "//construct_done.flag"): 
                pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
                pbar.update(1)
                continue

            contract_sample_dir = dataset_dir + contract + "//sample//"
            
            # construct dgl graph and lables
            smaple_infos, function_cnt = static_dgl_graphs_for_sample(contract_sample_dir)

            # add to the list
            graph_infos += smaple_infos

            total_function += function_cnt
            pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
            pbar.update(1)
        
        content_file = dataset_dir + "{}_db.json".format(dataset_dir.split("//")[-2])
        with open(content_file, "w+") as f:
            table_of_contents = {}
            for idx, sample_name in enumerate(graph_infos):
                table_of_contents[str(idx)] = sample_name
            f.write(json.dumps(table_of_contents, indent=4,  separators=(",", ":")))


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-check', type=int, default=0)
    parser.add_argument('-static', type=int, default=0)

    args = parser.parse_args()
    return args.dataset, args.check, args.static

if __name__ == '__main__':

    data_set, check, static = argParse()

    # ast_json_file = "dataset//resumable_loop//0x77c42a88194f81a17876fecce71199f48f0163c4//sample//Bitcoinrama-swapBack-4777//statement_ast_infos.json"
    # _construct_dgl_graph_v2(ast_json_file, infercode)

    # sample_dir = "dataset//reentrancy//0xffa3a0ff18078c0654b174cf6cb4c27699a4369e//sample//"
    # sample_graphs, sample_graph_lables = construct_dgl_graphs_for_sample(sample_dir, infercode)

    dataset_dir = "dataset//{}//".format(data_set)
    if static != 0:
        static_dgl_graphs_for_dataset(dataset_dir)

    else:
        infercode = infercode_init()
        construct_dgl_graphs_for_dataset(dataset_dir, infercode, check)