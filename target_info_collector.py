import json
import os
import shutil
import json
import platform
import shutil
import subprocess
from slither.slither import Slither
from slither.core.declarations import Function as SFunction
from slither.core.declarations.structure_contract import StructureContract
import logging
import networkx as nx

def is_in_black_list(c_name, f_name):

    key = "{}-{}".format(c_name, f_name)
    if key in ["ERC721-safeTransferFrom"]:
        print("黑名单成员:", key)
        return True

    return False

def compile_sol_file(sol_file, sol_ver):
    """
        compile the sol file based on differen OS(windows\linux)
    """

    # For different OS, with different solc select method
    if platform.system() == "Windows":
        solc_path = "{}{}{}".format("D:\\solc_compiler\\", sol_ver, "\\solc-windows.exe")
        slither = Slither(sol_file, solc=solc_path)
    else:
        subprocess.check_call(["solc-select", "use", sol_ver])
        slither = Slither(sol_file)
    return slither

class TrgetInfoCollector:

    def __init__(self, target_dir, log_level=logging.WARN) -> None:
        self.target_dir = target_dir
        self.logger = None
        self.target_filter = {}
        self.modifier_filter = {}
        self.polymorphism_filter = {}
        self.slither_error = 0

        # 合约语义相关
        self.structs = {} # 合约定义的结构体信息

        # slither cfg去重操作
        self.simple_key_dup = {}
        self.duplicate_simple_key = {}
        self.duplicate_slither_infos = {}
        self.duplicate_cfg_infos = {}

        # function info saved as slither
        self.function_slither_infos = {}
        self.modifier_slither_infos = {}
        
        # cfg info saved as json
        self.modifier_cfg_infos = {}
        self.function_cfg_infos = {}

        # modifier的语句信息保存此处，方便函数的展开
        self.modifier_stmt_infos = {}

        # cfg info saved as graph
        self.modifier_cfg_graphs = {}
        self.function_cfg_graphs = {}

        self.logger_init(log_level)
        self.get_polymorphism_filter()           # 函数多态过滤: function polymorphism filter
        self.environment_prepare_for_function()  # prepare for functions with SBP
        self.collect_cfg_infos_for_target()      # CONSTRUCT ALL CFG FOR FUNCTION AND modifier
        self.environment_prepare_for_modifier()  # prepare for modifier with or without SBP

    def logger_init(self, log_level):
        logger = logging.getLogger('TC')
        logger.setLevel(log_level)
        self.logger = logger
    
    def get_target_filter(self):
        self.logger.debug("\r\n")
        self.logger.debug(json.dumps(self.target_filter, indent=4, separators=(",", ":")))
        self.logger.debug("\r\n")
        return self.target_filter

    def get_modifier_filter(self):
        return self.modifier_filter

    def get_modifier_filte_by_key(self, c_name, m_name):
        return self.modifier_filter[c_name][m_name]

    def get_cfg_by_key(self, key, is_modifier):

        if not is_modifier:
            if key in self.function_cfg_infos:
                return self.function_cfg_infos[key]
        else:
            if key in self.modifier_cfg_infos:
                return self.modifier_cfg_infos[key]
        
        raise RuntimeError("!!!! No cfg for key ", key) # 没有CFG信息
    
    
    def get_slither_cfg_info_before_align(self, key, simple_key, cnt_key, is_modifier, target):
        """
            在对齐之前根据不同情况进行查询
            1.c_name-f_name
            2.入参个数
        """
        if target not in ["slither", "cfg_info"]:
            raise RuntimeError("get_slither_cfg_info: 错误的get请求 {}".format(target))

        if not is_modifier:

            if target == "slither":
                target_info = self.function_slither_infos
                dup_target_info = self.duplicate_slither_infos
            else:
                target_info = self.function_cfg_infos
                dup_target_info = self.duplicate_cfg_infos

            if self.duplicate_simple_key[simple_key] == 1:
                return target_info[key]
            else:
                return dup_target_info[simple_key][cnt_key]
        
        else:
            if target == "slither":
                return self.modifier_slither_infos[key]
            else:
                 return self.modifier_cfg_infos[key]

    def get_polymorphism_filter(self):
        duplicate_map = {}

        ast_dir =  self.target_dir + "ast_json//"
        for _, _, file_list in os.walk(ast_dir):
            for ast_json_file in file_list:
                
                if not str(ast_json_file).endswith(".json") or "--" in ast_json_file:
                    continue

                smaple_infos = str(ast_json_file).split(".json")[0].split("-")
                c_name = smaple_infos[0]
                f_name = smaple_infos[1]
                ast_id = smaple_infos[2]

                if ast_id == "MOD": continue
                
                dup_key = "{}-{}".format(c_name, f_name)
                if dup_key not in duplicate_map:
                    duplicate_map[dup_key] = [int(ast_id)]
                else:
                    duplicate_map[dup_key].append(int(ast_id))

        # 记录最大的
        for key in duplicate_map:
            if len(duplicate_map[key]) > 1:
                sample_ast_id = max(duplicate_map[key])
                self.polymorphism_filter[key] = sample_ast_id


    def get_modifier_info_by_key(self, c_name, m_name):
        return self.modifier_filter[c_name][m_name]


    def set_cfg_graph_by_key(self, graph:nx.DiGraph, up:nx.DiGraph, buttom:nx.DiGraph, key, is_modifier):

        if not is_modifier:
            self.function_cfg_graphs[key] = graph
        else:
            self.modifier_cfg_graphs[key] = {"cfg":graph, "up":up, "buttom":buttom}
    
    def get_cfg_graph_by_key(self, key, is_modifier):

        if not is_modifier:
            return self.function_cfg_graphs[key]
        
        else:
            if key not in self.modifier_cfg_graphs:
                return None
            
            return self.modifier_cfg_graphs[key]
  
    def set_modifier_stmts(self, ast_id, info):
        self.modifier_stmt_infos[str(ast_id)] = info

    def get_modifier_stmts(self, ast_id):

        # 格式为: {"vul":0, "vul_type":0, "stmt_type": "EXIT_POINT", "nodes":nodes, "edges":[]}
        if str(ast_id) not in self.modifier_stmt_infos:
            # print("======{}=======".format(str(ast_id)))
            # print(json.dumps(self.modifier_stmt_infos, indent=4, separators=(",", ":")))
            return None
        
        return self.modifier_stmt_infos[str(ast_id)]

    def environment_prepare_for_modifier(self):
        
        if self.slither_error:
            return
            
        ast_json_dir = self.target_dir  + "ast_json//"
        all_ast_jsons = os.listdir(ast_json_dir)
        
        for modifier_key in self.modifier_cfg_infos:
            
            modifier_infos = str(modifier_key).split("-")
            c_name = modifier_infos[0]
            f_name = modifier_infos[1]

            if f_name in ["onlyOwner", "nonReentrant"]:
                continue

            modifier_ast_file_profix = "{}-{}-{}".format(c_name, f_name, "MOD")
            self.logger.debug("==MODIFIER IS :{}".format(modifier_ast_file_profix))
            for ast_json_file in all_ast_jsons:
                if modifier_ast_file_profix in str(ast_json_file):
                    
                    modifier_sample_dir = self.target_dir  + "modifier//" + modifier_ast_file_profix + "//"
                    if os.path.exists(modifier_sample_dir):
                        shutil.rmtree(modifier_sample_dir)
                    os.mkdir(modifier_sample_dir)     
                    
                    shutil.copy(self.target_dir + "ast_json//" + str(ast_json_file), modifier_sample_dir)
                    if os.path.exists(self.target_dir + "sbp_json//" + str(ast_json_file)):
                        shutil.copy(self.target_dir + "sbp_json//" + str(ast_json_file), "{}{}".format(modifier_sample_dir, "sbp_info.json"))
                    
                    # check the temp dir for dot/png
                    for temp_dir in ["ast_dot//", "ast_png//", "cfg_dot//", "cfg_png//"]:
                        if not os.path.exists(modifier_sample_dir + temp_dir):
                            os.mkdir(modifier_sample_dir + temp_dir)

                    if c_name not in self.modifier_filter:
                        self.modifier_filter[c_name] = {}

                    if f_name not in self.modifier_filter[c_name]:
                        self.modifier_filter[c_name][f_name] = {
                            "ast_id":"mod",
                            "cname":c_name,
                            "fname":f_name,
                            "dir": modifier_sample_dir,
                            "file_name": ast_json_file
                        }

                    break
    
    def environment_prepare_for_function(self):

        # Record all target sample need to analyze
        target_dir = self.target_dir
        print("start ==> {}".format(target_dir))

        for temp_dir in ["sample//", "modifier//"]:
            if os.path.exists(target_dir + temp_dir):
                shutil.rmtree(target_dir + temp_dir)
            os.mkdir(target_dir + temp_dir)
        
        for _, _, file_list in os.walk(target_dir + "sbp_json//"):
            for sbp_file in file_list:
            
                # only support json format of AST, and pass the construct function
                if not str(sbp_file).endswith(".json") or "--" in sbp_file:
                    continue

                smaple_infos = str(sbp_file).split(".json")[0].split("-")
                c_name = smaple_infos[0]
                f_name = smaple_infos[1]
                ast_id = smaple_infos[2]

                # 多态检测
                sample_key = "{}-{}".format(c_name, f_name)
                if sample_key in self.polymorphism_filter:
                    if int(ast_id) != int(self.polymorphism_filter[sample_key]):
                        print("warning: 非最大多态样本:{}-{}".format(sample_key,  int(ast_id)))
                        continue

                # create the contract-function dir inside the sample dir
                smaple_dir = target_dir + "sample//" + "{}-{}-{}//".format(c_name, f_name, ast_id)
                os.mkdir(smaple_dir)
                
                # check the temp dir for dot/png
                for temp_dir in ["ast_dot//", "ast_png//", "cfg_dot//", "cfg_png//"]:
                    if not os.path.exists(smaple_dir + temp_dir):
                        os.mkdir(smaple_dir + temp_dir)

                # copy the json file to the example dir
                shutil.copy(target_dir + "ast_json//" + sbp_file, smaple_dir)
                shutil.copy(target_dir + "sbp_json//" + sbp_file, smaple_dir + "sbp_info.json")

                if c_name not in self.target_filter:
                    self.target_filter[c_name] = {}

                if f_name in self.target_filter[c_name]:
                    # 多态: slither目前不支持多态的CFG生成 https://github.com/smartcontract-detect-yzu/W33/issues/2
                    self.target_filter[c_name].pop(f_name)

                else: #if f_name not in self.target_filter[c_name]:
                    if not is_in_black_list(c_name, f_name):
                        self.target_filter[c_name][f_name] = {
                            "ast_id":ast_id,
                            "cname":c_name,
                            "fname":f_name,
                            "dir": smaple_dir,
                            "file_name": sbp_file
                        }
        
        # print(json.dumps(self.target_filter, indent=4, separators=(",", ":")))

    def _construct_cfg_for_all_modifiers(self, slither:Slither):
        """
            Get all modifiers' cfg in the target
        """
        for contract in slither.contracts:
            for _modifier in contract.modifiers:
                cfg_edges_list = []
                cfg_nodes_list = []
                node_duplicate = {}
                for stmt in _modifier.nodes:

                    if stmt.node_ast_id not in node_duplicate:
                        node_duplicate[stmt.node_ast_id] = 1
                        cfg_nodes_list.append(
                            {
                                "cfg_id":stmt.node_id,
                                "ast_id":stmt.node_ast_id, 
                                "node_type":stmt._node_type.__str__(), 
                                "node_expr":str(stmt)
                            }
                        )

                    for successor_stmt in stmt.sons:
                        cfg_edges_list.append({"from": stmt.node_ast_id, "to": successor_stmt.node_ast_id})
                
                key = "{}-{}-modifier".format(contract.name, _modifier.name)
                function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list, "offset":0}

                self.modifier_cfg_infos[key] = function_cfg_info
                self.modifier_slither_infos[key] = _modifier

    def _collect_structs_for_all_contract(self, slither:Slither):
        """
            收集合约中的结构体信息
        """
        for contract in slither.contracts:

            for structure in contract.structures:
                _struct:StructureContract = structure

                if _struct.canonical_name not in self.structs:
                    self.structs[_struct.canonical_name] = {}
                    # print("\r\n结构体名称:", _struct.canonical_name, _struct.name)
                    
                    for elem in _struct.elems:
                        self.structs[_struct.canonical_name][elem] = str(_struct.elems[elem].type)
                        # print("===>", elem, "  ", _struct.elems[elem].type)


    def _construct_cfg_for_all_target_functions(self, slither:Slither):
        """
            Construct cfg info for all target functions
        """
        for contract in slither.contracts:

            if contract.is_top_level:
                continue

            if contract.name not in self.target_filter:
                continue
                
            function_filter = self.target_filter[contract.name]
            for _function_slither in contract.functions + contract.modifiers:
                if _function_slither.name in function_filter:

                    # save the json file
                    cfg_edges_list = []
                    cfg_nodes_list = []
                    node_duplicate = {}

                    for stmt in _function_slither.nodes:
                        if stmt.node_ast_id not in node_duplicate:
                            node_duplicate[stmt.node_ast_id] = 1
                            cfg_nodes_list.append(
                                {
                                    "cfg_id":stmt.node_id, 
                                    "ast_id":stmt.node_ast_id,
                                    "node_type":stmt._node_type.__str__(),
                                    "node_expr": str(stmt), 
                                    "full_name": str(_function_slither.full_name),
                                    "input_params":len(_function_slither.parameters)
                                }
                            )
                        
                        for successor_stmt in stmt.sons:
                            cfg_edges_list.append({"from": stmt.node_ast_id, "to": successor_stmt.node_ast_id})

                    smaple_name = "{}-{}-{}".format(contract.name, _function_slither.name, function_filter[_function_slither.name]["ast_id"])
                    function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list, "offset":0}

                    self.function_cfg_infos[smaple_name] = function_cfg_info
                    self.function_slither_infos[smaple_name] = _function_slither

                    # slither对于同一个函数会生成多个具体的实现
                    simple_key = "{}-{}".format(contract.name, _function_slither.name)
                    if simple_key not in self.duplicate_simple_key:
                        self.duplicate_simple_key[simple_key] = 1
                    else: 
                        self.duplicate_simple_key[simple_key] += 1

                    # 出现了(1)多态; (2)重复实现: slither的bug
                    # 基于入参个数的目标函数选择
                    if simple_key not in self.duplicate_slither_infos:
                        self.duplicate_slither_infos[simple_key] = {}
                        self.duplicate_cfg_infos[simple_key] = {}

                    # 原则:只使用最后一个 --> 如果存在一个fun(param1, param2)的两个slither实现, 选最后一个
                    _in_param_cnt = len(_function_slither.parameters)
                    self.duplicate_cfg_infos[simple_key][_in_param_cnt] = function_cfg_info
                    self.duplicate_slither_infos[simple_key][_in_param_cnt] = _function_slither

                    self.logger.debug("slither入参类型:{}".format([str(in_param.type) for in_param in _function_slither.parameters]))
        
        # print("!!!! slither 多函数实现:{}".format(self.duplicate_simple_key))
        # print("!!!! 多态的实现:{}".format(self.duplicate_cfg_infos))
        # print("===目标函数的CFG:{}".format([key for key in self.function_cfg_infos]))

    def _get_function_cfg(self, function:SFunction, cname, ast_id):
        """
            暂时不用
            -- 利用slither的dot文件生成cfg
        """

        cfg_dot_file = "{}_cfg.dot".format(self.function.name)
        function.cfg_to_dot(cfg_dot_file)

        cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
        os.remove(cfg_dot_file)

        cfg.graph["name"] = function.name
        cfg.graph["contract_name"] = cname
        cfg.graph["ast_id"] = ast_id

        if len(cfg.nodes) == 0:
            return None

        for node in function.nodes:
            cfg_node = cfg.nodes[str(node.node_id)]
            new_label = "ID:{} {}".format(str(node.node_id), cfg_node["label"])
            cfg_node["label"] = new_label
            cfg_node["expression"] = node.expression.__str__()
            if cfg_node["expression"] is None:
                cfg_node["expression"] = cfg_node["label"]
            cfg_node["type"] = node.type.__str__()
            cfg_node["fid"] = function.id
            cfg_node["node_id"] = node.node_id
            cfg_node["ast_id"] = node.node_ast_id

        return cfg

    
    def collect_cfg_infos_for_target(self):
        """
            Get the cfg info into the modifier_cfg_infos and function_cfg_infos
        """

        pwd = os.getcwd()
        os.chdir(self.target_dir)  

        # compile the sol file by slither
        f = open("download_done.txt")
        compile_info = json.load(f)
        try:
            # 为了节约内存slither信息并不保存
            slither = compile_sol_file(compile_info["name"], compile_info["ver"])
        except:
            self.logger.error("silther error for {} {}".format(compile_info["name"], compile_info["ver"]))
            self.slither_error = 1
            os.chdir(pwd)
            return

        self._collect_structs_for_all_contract(slither)

        self._construct_cfg_for_all_modifiers(slither)

        self._construct_cfg_for_all_target_functions(slither)

        os.chdir(pwd)
                       