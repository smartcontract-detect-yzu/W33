import argparse
import json
import random
import re
import shutil
import dgl 
import torch
import os
from tqdm import tqdm

from baseline.baseline_constructor import Baseline_Constructor

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
                if len(content) == 0 or str(content).isspace():
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


def construct_dgl_graphs_for_sample(contract_sample_dir, infercode, pass_flag):

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

        elif pass_flag != 0 and os.path.exists(sample_dgl_done_flag):
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
                if pass_flag != 0:  # PASS模式下不抛出异常
                    raise RuntimeError("!!!!!!!!!!!!!!!!!")
                    
        
    return sample_graphs, sample_graphs_cnt, smaple_infos, function_cnt


def construct_dgl_graphs_for_dataset(dataset_dir, infercode, pass_flag, db):

    graphs = []
    graphs_cnts = []
    graph_infos = []
    total_function = 0
    
    print(">>>>>>>>>>>>>>开始构建数据集:{}<<<<<<<<<<<<<<<".format(dataset_dir))

    all_contracts = os.listdir(dataset_dir)
    with tqdm(total=len(all_contracts)) as pbar:
        for contract in all_contracts:
            
            if "sbp_dataset" in dataset_dir:
                pass
                
            # 非sbp_dataset数据集必须是文件夹, 且文件夹存在construct_done.flag标志
            elif str(contract).endswith(".json") or not os.path.exists(dataset_dir + contract + "//construct_done.flag"): 
                pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
                pbar.update(1)
                continue

            # construct dgl graph and lables
            contract_sample_dir = dataset_dir + contract + "//sample//"
            sample_graphs, sample_graphs_cnt, smaple_infos, function_cnt = construct_dgl_graphs_for_sample(contract_sample_dir, infercode, pass_flag)

            # add to the list
            graphs += sample_graphs
            graphs_cnts += sample_graphs_cnt
            graph_infos += smaple_infos

            total_function += function_cnt
            pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
            pbar.update(1)

            if total_function > 9999999999:
                print("!!!Already collect max function samples")
                break
    
    print("==total_function is :{}".format(total_function))
    print("==total ast and cfg is:{}".format(len(graphs)))
    
    if db != 0: # 创建数据库
        # content_file = dataset_dir + "table_of_contents.json"
        content_file = "{}//{}_{}_{}_db.json".format(DATASET_BIN_DIR, dataset_dir.split("//")[-2], total_function, len(graphs))
        with open(content_file, "w+") as f:
            table_of_contents = {}
            for idx, sample_name in enumerate(graph_infos):
                table_of_contents[str(idx)] = sample_name
            f.write(json.dumps(table_of_contents, indent=4,  separators=(",", ":")))

    # construct the dgl dataset bin file
    infos = {"graph_cnts": torch.tensor(graphs_cnts)}
    bin_file_name = "{}//{}_{}_{}.bin".format(DATASET_BIN_DIR, dataset_dir.split("//")[-2], total_function, len(graphs))
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

            if "sbp_dataset" in dataset_dir:
                pass
            
            elif str(contract).endswith(".json") or not os.path.exists(dataset_dir + contract + "//construct_done.flag"): 
                # 非sbp_dataset数据集必须是文件夹, 且文件夹存在construct_done.flag标志
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


VUL_TYPE_LIST = ["SafeMath", "low-level call", "safe cast", "transaction order dependency", "nonReentrant", "onlyOwner", "resumable_loop"]
PHASE_ONE_VUL = ["safe cast", "transaction order dependency", "resumable_loop"]
def vul_type_based_dataset(phase, new_dataset):

    root_dir = "dataset//{}//".format(new_dataset)
    backup_datasets = ["dataset//dataset_reloop//", "dataset//reentrancy//"]

    if phase == 2:
        target_number = random.randint(1500,2000)
        _cnt = 0
        print("目标添加样本: ", target_number)

    elif phase == 3:
        target_number = random.randint(1000,1500)
        _cnt = 0
        already_cnt = 0
        print("目标添加样本: ", target_number)

    for _tmp_dataset in backup_datasets:
        sample_files = os.listdir(_tmp_dataset)
        random.shuffle(sample_files)
        for address in sample_files:
            path_sample = "{}//{}//".format(_tmp_dataset, address)
            if os.path.exists(path_sample + "construct_done.flag"):
                sample_dir_path = path_sample + "sample//"
                c_f_samples = os.listdir(sample_dir_path)
                for c_f in c_f_samples:
                    cfid = str(c_f).split("-")
                    c_name = cfid[0]
                    f_name = cfid[1]
                    c_f_sample_dir_path = sample_dir_path + c_f + "//"
                    if os.path.exists(c_f_sample_dir_path + "statement_ast_infos.json"):
                        
                        with open(c_f_sample_dir_path + "statement_ast_infos.json", "r") as f:
                            stmts_infos = json.load(f)
                            for stmt_id in stmts_infos:
                                
                                if stmt_id == "cfg_edges":
                                    continue

                                # 开始统计
                                stmt_info = stmts_infos[stmt_id]
                                if stmt_info["vul"] != 0:
                                    vul_type_infos = stmt_info["vul_type"]
                                    for ast_id in vul_type_infos:

                                        # 阶段1: 聚合 resumable_loop; safe cast 和 transaction order dependency 三个最少类型样本
                                        if phase == 1:

                                            if vul_type_infos[ast_id] in PHASE_ONE_VUL:
                                                if not os.path.exists(root_dir + address):
                                                    # dst = root_dir + address
                                                    # src = path_sample
                                                    # shutil.copytree(src, dst)
                                                    # shutil.rmtree(root_dir + address + "//sample")
                                                    os.makedirs(root_dir + address + "//sample")

                                                src = c_f_sample_dir_path
                                                dst = root_dir + address + "//sample//" + c_f
                                                if not os.path.exists(dst):
                                                    shutil.copytree(src, dst) # 拷贝到目标目录
                                                break    

                                        # 阶段2: 补全 low-level call 增加 2000~3000的随机数
                                        elif phase == 2:
                                            if c_name in ["ERC20", "ERC720"]:
                                                pass

                                            elif vul_type_infos[ast_id] == "low-level call":
                                                
                                                if not os.path.exists(root_dir + address):
                                                    # dst = root_dir + address
                                                    # src = path_sample
                                                    # shutil.copytree(src, dst)
                                                    # shutil.rmtree(root_dir + address + "//sample")
                                                    os.makedirs(root_dir + address + "//sample")

                                                src = c_f_sample_dir_path
                                                dst = root_dir + address + "//sample//" + c_f
                                                if not os.path.exists(dst):
                                                    shutil.copytree(src, dst) # 拷贝到目标目录
                                                    _cnt += 1
                                                
                                                if _cnt >= target_number:
                                                    return # 达到目标
                                                else:
                                                    break  # 下一个sample    
                                        
                                        # 阶段3: 补全 nonReentrant 增加 2000~4000的随机数
                                        elif phase == 3:
                                            if c_name in ["ERC20", "ERC720"]:
                                                pass

                                            elif vul_type_infos[ast_id] == "nonReentrant":
                                                already_cnt += 1
                                                if not os.path.exists(root_dir + address):
                                                    # dst = root_dir + address
                                                    # src = path_sample
                                                    # shutil.copytree(src, dst)
                                                    # shutil.rmtree(root_dir + address + "//sample")
                                                    os.makedirs(root_dir + address + "//sample")

                                                src = c_f_sample_dir_path
                                                dst = root_dir + address + "//sample//" + c_f
                                                if not os.path.exists(dst):
                                                    shutil.copytree(src, dst) # 拷贝到目标目录
                                                    _cnt += 1

                                                if _cnt >= target_number:
                                                    return # 达到目标

                                                else:
                                                    break  # 下一个sample  

def _select_solc_version(version_info):
    versions = ['0', '0.1.7', '0.2.2', '0.3.6', '0.4.26', '0.5.17', '0.6.12', '0.7.6', '0.8.17']

    start = 0

    for i, char in enumerate(version_info):
        if char == '0' and start == 0:
            start = 1
            op_info = version_info[0:i]

            space_cnt = 0
            for c in op_info:
                if c == '^' or c == '>':
                    return versions[int(version_info[i + 2])]

                if c == '=':
                    last_char = version_info[i + 5]
                    if '0' <= last_char <= '9':
                        return version_info[i:i + 6]
                    else:
                        return version_info[i:i + 5]

                if c == ' ':
                    space_cnt += 1

            if space_cnt == len(op_info):
                last_char = version_info[i + 5]

                if '0' <= last_char <= '9':
                    return version_info[i:i + 6]
                else:
                    return version_info[i:i + 5]

    return "auto"

def _parse_solc_version(file_name):

        version_resault = None

        with open(file_name, 'r', encoding='utf-8') as contract_code:

            mini = 100
            for line in contract_code:
                target_id = line.find("pragma solidity")
                if target_id != -1:
                    new_line = line[target_id:]
                    version_info = new_line.split("pragma solidity")[1]
                    v = _select_solc_version(version_info)

                    if v[-3] == '.':
                        last_version = int(v[-2:])
                    else:
                        last_version = int(v[-1:])

                    if mini > last_version:
                        mini = last_version
                        version_resault = v
                    
                    return version_resault

            if version_resault is None:
                version_resault = "0.4.26"

            return version_resault


def _smartbugs_get_function(sol_file):
    
    function_vul = {}
    _function_name = None

    lines = open(sol_file, "r").readlines()
    for line in lines:
        text = line.strip() 
        result = re.search(r'(function)\s(\w+)\([a-zA-Z0-9_:\[\]=, ]*\)',text) # function名匹配
        if result != None:
            _function_name = result.group(2)
        
        if "<yes> <report>" in text:
            function_vul[_function_name] = text
    
    return function_vul
            
        
      


def smartbugs_dataset(smartbugs_dir):

    name_map = {
        "access_control": "ac",
        "arithmetic": "math",
        # "bad_randomness": "random",
        "denial_of_service": "dos",
        "front_running": "tod",
        "reentrancy": "re",
        # "short_addresses": "short",
        # "time_manipulation": "time",
        "unchecked_low_level_calls": "llc"
    }

    target_dir = "dataset//verified_smartbugs_new//"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    vul_type_dirs = os.listdir(smartbugs_dir)
    with tqdm(total=len(vul_type_dirs)) as pbar:
        for vul_type in vul_type_dirs:

            vul_dir_path = smartbugs_dir + vul_type
            if not os.path.isdir(vul_dir_path) or vul_type not in name_map:
                pass

            else:
                sol_files = os.listdir(vul_dir_path)
                for sol_file in sol_files:
                    if str(sol_file).endswith(".sol"):

                        address = str(sol_file).strip(".sol")
                        _new_dir_name = f"{address}-{name_map[vul_type]}"
                        
                        if os.path.exists(target_dir + _new_dir_name):
                            shutil.rmtree(target_dir + _new_dir_name)
                        os.mkdir(target_dir + _new_dir_name)

                        _src = f"{vul_dir_path}//{sol_file}"
                        _dst = target_dir + _new_dir_name
                        shutil.copy(_src, _dst)

                        # 得到存在漏洞的函数名称
                        function_vul = _smartbugs_get_function(_src)

                        # 解析版本号和目标sol文件类型
                        _info_file = f"{target_dir}{_new_dir_name}//download_done.txt"
                        sol_version = _parse_solc_version(_src)
                        file_info = {
                            "name": sol_file,
                            "ver": sol_version,
                            "label": function_vul,
                            "compile":"ok"
                        }
                        with open(_info_file, "w+") as f:
                            f.write(json.dumps(file_info, indent=4, separators=(",", ":")))

                       
                        
        pbar.update(1)               

def create_dataset_list(dataset_dir, d_name):

    print(">>>>>>>>>>>>>>扫描数据集:{}<<<<<<<<<<<<<<<".format(dataset_dir))
    
    all_contract_dirs = os.listdir(dataset_dir)
    datset_list = {}

    with tqdm(total=len(all_contract_dirs)) as pbar:
        for contract_dir in all_contract_dirs:

            _contract_sample_dir = dataset_dir + "{}//sample//".format(contract_dir)
            address = contract_dir
            if not os.path.isdir(_contract_sample_dir):
                pbar.update(1)
                continue
            
            c_f_samples = os.listdir(_contract_sample_dir)
            for _sampe in c_f_samples:
                _info = {
                    "address":address,
                    "sample": _sampe
                }
                key = _contract_sample_dir + _sampe
                datset_list[key] = _info
            
            pbar.update(1)
            
    _list_name = "dataset//" + d_name + "_list.json"
    with open(_list_name, "w+") as f:
        f.write(json.dumps(datset_list, indent=4,  separators=(",", ":")))

def do_create_src_dataset(dataset_name):

    backup_datasets = ["dataset//dataset_reloop//", "dataset//reentrancy//"]

    dataset_src_name = dataset_name + "_src"
    dataset_src_name_path = "dataset//" + dataset_src_name + "//"
    if not os.path.exists(dataset_src_name_path):
        os.mkdir(dataset_src_name_path)

    # 首先读取dataset_list
    list_file = f"{dataset_name}_list.json"
    _list_file = f"dataset//{list_file}"
    if not os.path.exists(list_file):
        print("ERROR: 请先创建LIST JOSN文件")
        return
    
    address_list = []
    dataset_list_json = json.load(open(_list_file, "r"))
    for path_key in dataset_list_json:
        address = dataset_list_json[path_key]["address"]
        address_list.append(address)
        

    for target_address in address_list:

        print(dataset_src_name_path + target_address)

        if not os.path.exists(dataset_src_name_path + target_address):
            os.mkdir(dataset_src_name_path + target_address)

        for backup_dir in backup_datasets:
            _target_dir = backup_dir + target_address
            if os.path.exists(_target_dir):
                _target_files = os.listdir(_target_dir)
                for _target_file in _target_files:
                    
                    if str(_target_file) == "sbp_json":
                        target_sbp_infos = {}
                        sbp_dir = _target_dir + "//" + _target_file + "//"
                        sbp_files = os.listdir(sbp_dir)
                        for _sbp_file_name in sbp_files:
                            with open(sbp_dir + _sbp_file_name) as f:
                                sbp_info = json.load(f)
                                _sbp_key = str(_sbp_file_name).strip(".json")
                                target_sbp_infos[_sbp_key] = sbp_info
                        contract_level_sbp_summary = dataset_src_name_path + target_address + "//total_sbp.json"
                        with open(contract_level_sbp_summary, "w+") as f:
                            f.write(json.dumps(target_sbp_infos, indent=4, separators=(",", ":")))

                    if str(_target_file).endswith(".sol") or str(_target_file) == "download_done.txt":
                        src = _target_dir + "//" + _target_file
                        dst = dataset_src_name_path + target_address
                        shutil.copy(src, dst)
                        

def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-baseline', type=str, default=None)
    parser.add_argument('-phase', type=int, default=0)
    parser.add_argument('-pass_flag', type=int, default=0)
    parser.add_argument('-db', type=int, default=0)
    parser.add_argument('-static', type=int, default=0)
    parser.add_argument('-create_list', type=int, default=0)
    parser.add_argument('-create_src_dataset', type=str, default=None)

    args = parser.parse_args()
    return args.dataset, args.baseline, args.phase, args.pass_flag, args.db, args.static, args.create_list, args.create_src_dataset

def do_test_by_trained_model():
    pass

if __name__ == '__main__':

    DATASET_BIN_DIR = "dataset_bin"
    data_set, baseline, phase, pass_flag, db, static, create_list, create_src_dataset = argParse()

    # ast_json_file = "dataset//resumable_loop//0x77c42a88194f81a17876fecce71199f48f0163c4//sample//Bitcoinrama-swapBack-4777//statement_ast_infos.json"
    # _construct_dgl_graph_v2(ast_json_file, infercode)

    # sample_dir = "dataset//reentrancy//0xffa3a0ff18078c0654b174cf6cb4c27699a4369e//sample//"
    # sample_graphs, sample_graph_lables = construct_dgl_graphs_for_sample(sample_dir, infercode)

    if create_src_dataset is not None:
        print("创建源代码数据集")
        do_create_src_dataset(create_src_dataset)
		
    if data_set is None and baseline is None:
        
        new_dataset = "sbp_dataset_var"
        if phase == 1:
            print("开始手动构建数据集 -- 阶段1: 收集resumable_loop safe_cast 和 transaction_order_dependency 三个最少类型样本")
            vul_type_based_dataset(phase, new_dataset)

        elif phase == 2:
            print("开始手动构建数据集 -- 阶段2: 补充 1500 ~ 2000 个low-level call 样本")
            vul_type_based_dataset(phase, new_dataset)

        elif phase == 3:
             print("开始手动构建数据集 -- 阶段3: 补充 1500 ~ 2000 个 nonReentrant 样本")
             vul_type_based_dataset(phase, new_dataset)

        elif phase == 4:
            print("开始手动构建数据集 -- 将smartbugs构建为一般表示")
            smartbugs_dir = "dataset//smartbugs//"
            smartbugs_dataset(smartbugs_dir)
        
        else:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    elif baseline != None:
        
        print("创建baseline数据集")
        dataset = "dataset//" + baseline
        bc = Baseline_Constructor(dataset, "re")
        bc.get_target_samples()
        bc.construct_normalized_dataset()
        
        
        if 0:  # tmp dataset
            bc.TMP_create_feature_for_smaples()
            bc.TMP_create_train_valid_dataset()
        
        if 0: # peculiar dataset
            bc.Peculiar_create_feature_for_dataset()
            bc.Peculiar_create_train_valid_dataset()
        
        if 0: # CBGRU dataset
            bc.CBGRU_create_feature_for_sample()
        
    
    else:
        dataset_dir = "dataset//{}//".format(data_set)
        if static != 0:
            static_dgl_graphs_for_dataset(dataset_dir)

        if create_list:
            create_dataset_list(dataset_dir, data_set)
        
        else:
            from infercode.client.infercode_client import InferCodeClient
            infercode = infercode_init()
            construct_dgl_graphs_for_dataset(dataset_dir, infercode, pass_flag, db)