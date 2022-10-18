import json
import os
import shutil
import json
import platform
import shutil
import subprocess
from slither.slither import Slither
from slither.core.declarations import Function as SFunction
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
        self.slither_error = 0

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
        self.environment_prepare_for_function()  # prepare for functions with SBP
        self.collect_cfg_infos_for_target()  # CONSTRUCT ALL CFG FOR FUNCTION AND modifier
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

        # 没有CFG信息
        raise RuntimeError("!!!! No cfg for key ", key)
    
    def get_slither_by_key(self, key, is_modifier):

        if not is_modifier:
            if key in self.function_slither_infos:
                return self.function_slither_infos[key]
        else:
            if key in self.modifier_slither_infos:
                return self.modifier_slither_infos[key]
        
        # 没有CFG信息
        raise RuntimeError("!!!! No Slither for key ", key)

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
        # print(json.dumps(self.modifier_stmt_infos, indent=4, separators=(",", ":")))
        if str(ast_id) not in self.modifier_stmt_infos:
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

                if f_name not in self.target_filter[c_name]:
                    if not is_in_black_list(c_name, f_name):
                        self.target_filter[c_name][f_name] = {
                            "ast_id":ast_id,
                            "cname":c_name,
                            "fname":f_name,
                            "dir": smaple_dir,
                            "file_name": sbp_file
                        }

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
                function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list}

                self.modifier_cfg_infos[key] = function_cfg_info
                self.modifier_slither_infos[key] = _modifier

    
    def _construct_cfg_for_all_target_functions(self, slither:Slither):
        """
            Construct cfg info for all target functions
        """
        for contract in slither.contracts:
            if contract.name not in self.target_filter:
                continue
                
            function_filter = self.target_filter[contract.name]
            for _function in contract.functions_and_modifiers:
                if _function.name in function_filter:
                    
                     # save the json file
                    cfg_edges_list = []
                    cfg_nodes_list = []
                    node_duplicate = {}
                    for stmt in _function.nodes:

                        if stmt.node_ast_id not in node_duplicate:
                            node_duplicate[stmt.node_ast_id] = 1
                            cfg_nodes_list.append(
                                {
                                    "cfg_id":stmt.node_id, 
                                    "ast_id":stmt.node_ast_id,
                                    "node_type":stmt._node_type.__str__(),
                                    "node_expr": str(stmt) 
                                }
                            )
                        
                        for successor_stmt in stmt.sons:
                            cfg_edges_list.append({"from": stmt.node_ast_id, "to": successor_stmt.node_ast_id})

                    smaple_name = "{}-{}-{}".format(contract.name, _function.name, function_filter[_function.name]["ast_id"])
                    function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list}

                    self.function_cfg_infos[smaple_name] = function_cfg_info
                    self.function_slither_infos[smaple_name] = _function


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

        self._construct_cfg_for_all_modifiers(slither)

        self._construct_cfg_for_all_target_functions(slither)

        os.chdir(pwd)
                       