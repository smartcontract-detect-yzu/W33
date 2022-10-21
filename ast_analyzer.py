import itertools
import json
import os
import shutil
import subprocess
import networkx as nx
import logging
from target_info_collector import TrgetInfoCollector
from slither.core.declarations import Function as SFunction
from slither.core.declarations import SolidityFunction as SolidityFunction
from slither.core.declarations import Contract as SContract
from slither.core.declarations import SolidityVariable as SolidityVariable
from slither.core.variables.variable import Variable as SVariable

from slither.core.cfg.node import Node as SNode
import networkx.drawing.nx_pydot as nx_dot

def _get_function_call_info(node:SNode, debug=None):

    external_call_flag = node.can_reenter()
    state_assign_flag = "False"
    if len(node.state_variables_written) > 0:
        state_assign_flag = "True"

    if node.node_ast_id == debug:

        print("\r\n===================variables:START=======================")
        print("state_variables_written:{} {}".format(len(node.state_variables_written), 
            [wsv.name for wsv in node.state_variables_written]))
        print("variables_written:{} {}".format(len(node.variables_written), 
            [ws.name for ws in node.variables_written]))
        print("===================variables:END=======================")

        print("\r\n===============CALL_INFO:START===============")
        print("---can_reenter:{} can_send_eth:{}".format(node.can_reenter, node.can_send_eth))
        for in_call in node.internal_calls:
            if isinstance(in_call, SFunction):
                print("----【 internal_calls:Function】", in_call.name, in_call.id)

            if isinstance(in_call, SolidityFunction):
                print("----【 internal_calls:SolidityFunction】", in_call.name, in_call.name)

        for in_call in node.library_calls:
            if isinstance(in_call, SContract):
                print("----【 library_calls:SContract", in_call.name, in_call.id)

            if isinstance(in_call, SFunction):
                print("----【 library_calls:SFunction", in_call.name, in_call.id)

        for in_call in node.solidity_calls:
            print("----【 solidity_calls:】", in_call.name, in_call.name)

        for in_call in node.high_level_calls:
            if isinstance(in_call, SContract):
                print("----【 high_level_calls:SContract】", in_call.name, in_call.id)

            if isinstance(in_call, SFunction):
                print("----【 high_level_calls:SFunction", in_call.name, in_call.id)

            if isinstance(in_call, SVariable):
                print("----【 high_level_calls:SContract】", in_call.name, in_call.name)
        
        for in_call in node.low_level_calls:      
            if isinstance(in_call, SVariable):
                print("----【 low_level_calls:SVariable", in_call.name, in_call.id)

            if isinstance(in_call, SolidityVariable):
                print("----【 low_level_calls:SolidityVariable", in_call.name, in_call.name)

            if isinstance(in_call, str):
                print("----【 low_level_calls:SVariable", in_call)
        
        for ext_call in node.external_calls_as_expressions:
            print("^^^^【ext_call】", ext_call)
        
        print("===============CALL_INFO:END=================\r\n")

    return external_call_flag, state_assign_flag


def _get_all_leaf_nodes(graph:nx.DiGraph):

    leaf_nodes = []
    for graph_node in graph:
        if graph.out_degree(graph_node) == 0:
            leaf_nodes.append(graph_node)
     
    return leaf_nodes

def _remove_useless_leaf(graph:nx.DiGraph):
    nodes_to_remove = []

    for graph_node in graph:

        # 叶子节点
        if graph.out_degree(graph_node) == 0:  
            if graph.nodes[graph_node]["expr"] in ["Block"]:
                nodes_to_remove.append(graph_node)

    graph.remove_nodes_from(nodes_to_remove)

    return graph

def _do_take_second(elem):
    return elem[1]



def _do_remove_node(g: nx.DiGraph, node):
    """
        删除节点的同时继承边:
            A -e1-> B(to_remove) -e2-> C
        删除后
            the rule is that it will inherit the e1 edge
            A -e1-> C
    """

    sources = []
    targets = []
    
    for source, _ in g.in_edges(node):
        sources.append(source)

    for _, target in g.out_edges(node):
        targets.append(target)

    new_edges = itertools.product(sources, targets)
    new_edges_with_data = []
    for new_from, new_to in new_edges:
        new_edges_with_data.append((new_from, new_to))
    
    g.add_edges_from(new_edges_with_data)
    g.remove_node(node)

    return g

def _do_save_entry_point_to_json(entry_ast_id, entry_ast_info, stmt_ast_json):

    nodes = {}
    edges = []

    nodes[int(entry_ast_id)] = {
            "id": entry_ast_id,
            "label": 0,
            "content": "ENTRY_POINT",
            "ast_type": "ENTRY_POINT",
            "pid": 0
    }

    if "vul" in entry_ast_info:
        entry_info = {"vul":entry_ast_info["vul"], "stmt_type":"ENTRY_POINT", "vul_type":entry_ast_info["vul_type"], "nodes":nodes, "edges":edges}
    else:
        entry_info = {"vul":0, "vul_type":0, "stmt_type":"ENTRY_POINT", "nodes":nodes, "edges":edges}

    stmt_ast_json[str(entry_ast_id)] = entry_info

    return entry_info

def _do_save_stmt_ast_to_json(stmt_ast_graph:nx.DiGraph, stmt_ast_json, stmt_ast_root, vul_label, vul_type, stmt_type):
    """
    transfer a ast graph of a statement into a json format
    """
    nodes = {}
    edges = []

    for node_id in stmt_ast_graph.nodes:

        nodes[node_id] = {
            "id": node_id,
            "label": stmt_ast_graph.nodes[node_id]["label"],
            "content": stmt_ast_graph.nodes[node_id]["expr"],
            "ast_type": stmt_ast_graph.nodes[node_id]["ast_type"],
            "pid": stmt_ast_graph.nodes[node_id]["pid"]
        }

    for edge in stmt_ast_graph.edges:
        edges.append({"from":edge[0], "to":edge[1]})

    stmt_ast_info = {"vul":vul_label, "vul_type":vul_type, "stmt_type":stmt_type, "nodes":nodes, "edges":edges}
    stmt_ast_json[str(stmt_ast_root)] = stmt_ast_info

    return stmt_ast_info

def _do_create_exit_point_to_json(stmt_ast_json):
    nodes = {}
    nodes["exit"] = {"id": "exit","label": "EXIT_POINT","content": "EXIT_POINT","ast_type": "EXIT_POINT","pid": None}
    stmt_ast_info = {"vul":0, "vul_type":0, "nodes":nodes, "edges":[]}
    stmt_ast_json["exit"] = stmt_ast_info
    return stmt_ast_info

def _get_stmt_label_info(lable_infos):
    """
    为了适配SBP INFO的版本更新
    """
    if isinstance(lable_infos, list):
        return lable_infos[0]
    else:
        return lable_infos


def _split_function_stmts(function_ast_graph: nx.DiGraph):
    """
        根据第一个block节点, 将一个function分解成多个语句
        function
            -- modifier
            -- params
            -- block
                -- [stmt...]
    """
    function_root = function_ast_graph.graph["root"]

    all_sub_nods = []
    function_stmts = []

    for block_node in nx.neighbors(function_ast_graph, function_root):

        # find the first Block node as Root (1)params (2) Block
        if function_ast_graph.nodes[block_node]['expr'] == "Block":

            # record the top-level "Block" node
            function_ast_graph.graph["top_block"] = block_node

            # statement save
            for stmt_node in nx.neighbors(function_ast_graph, block_node):
                sub_nodes = nx.dfs_tree(function_ast_graph, stmt_node)
                all_sub_nods += sub_nodes

                stmt_ast = nx.DiGraph(nx.subgraph(function_ast_graph, [node for node in sub_nodes]))
                stmt_ast.graph["name"] = "stmt_at_{}".format(stmt_node)
                stmt_ast.graph["root"] = stmt_node
                function_stmts.append(nx.DiGraph(stmt_ast))

            function_ast_graph.remove_nodes_from(all_sub_nods)
            function_ast_graph.graph["top_stmts"] = []
            return function_ast_graph, function_stmts

class SbpNormalizer:
    """
        security best practice normalize 
    """
    def __init__(self) -> None:

        self.SAFE_BEST_PRACTICES_LIBS = {
            "safeMath": 1,
            "SafeERC20": 1
        }

        self.safeMathMap = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/"
        }

        self.SafeLowLevelCallMap = {
            "safeTransfer": "transfer",
            "safeTransferFrom": "transferFrom",
            "sendValue": "1",
            "functionCall": "1",
            "functionCallWithValue": "1",
            "functionStaticCall": "1"
        }

        self.TodCallMap = {
            "approve": "1",
            "safeApprove": "1",
            "safeIncreaseAllowance": "approve",
            "safeDecreaseAllowance": "approve",
        }
        
    def _normalize_safemath(self, ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
        """
            简化调用SafeMath接口的函数
        """
        subnodes = [subnode for subnode in ast.successors(sbp_node)]

        # when call safemath, the first left node of functionCall is the api
        left_child = subnodes[0]
        if ast.nodes[left_child]["expr"] in self.safeMathMap:

            # normalize the functionCall node with its original operation
            ast.nodes[sbp_node]["expr"] = self.safeMathMap[ast.nodes[left_child]["expr"]]
            ast.nodes[sbp_node]["label"] = "{}  @ID:{}".format(ast.nodes[sbp_node]["expr"], sbp_node)

            # remove the called safeMath API
            node_to_remove.append(left_child)

            if len(subnodes) == 3:  # v1.add(v2, note) 有接口提供了3个入参，最后一个入参是提示信息
                # 提示信息的内容全部删除，可以直接删除
                nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, subnodes[2]))


    def _normalize_llc_v1(self, ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
        """
            简化调用安全 low-level-call接口的函数
            只保留call, 其他的全部删除
            expressionStatement --> functionCall --> api --> address(对象)
                                                 --> params...
        """
        subnodes = [subnode for subnode in ast.successors(sbp_node)]

        # when call safe llc, the first left node of functionCall is the api
        left_child = subnodes[0]
        if ast.nodes[left_child]["expr"] in self.SafeLowLevelCallMap:

            # normalize the functionCall node with its original operation: call
            ast.nodes[sbp_node]["expr"] = "call"
            ast.nodes[sbp_node]["label"] = "{}  @ID:{}".format(ast.nodes[sbp_node]["expr"], sbp_node)

            # remove the called llc api node
            node_to_remove.append(left_child)

            # For: call(param1,param2,....).value(paramn)
            for irrelevant_node in subnodes[1:]:
                # 其它参数对low-level call语义没有帮助, 可以直接删除
                nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, irrelevant_node))
    
    def _normalize_llc(self, ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
        """
            简化调用安全 low-level-call接口的函数
            只保留call, 其他的全部删除
            expressionStatement --> functionCall --> api(替换掉) --> address(对象)
                                                 --> params...
        """
        subnodes = [subnode for subnode in ast.successors(sbp_node)]

        # when call safe llc, the first left node of functionCall is the api
        left_child = subnodes[0]
        if ast.nodes[left_child]["expr"] in self.SafeLowLevelCallMap:

            # 替换api为low-level call
            ast.nodes[left_child]["expr"] = "call"
            ast.nodes[left_child]["label"] = "{}  @ID:{}".format(ast.nodes[left_child]["expr"], left_child)

            # For: call(param1,param2,....).value(paramn)
            for irrelevant_node in subnodes[1:]:
                # 其它参数对low-level call语义没有帮助, 可以直接删除
                nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, irrelevant_node))
    
    def _normalize_safecast(self, ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
        """
            Safe down cast normalize
        """
        subnodes = [subnode for subnode in ast.successors(sbp_node)]
        cast_api_node = subnodes[0]
        cast_api = str(ast.nodes[cast_api_node]["expr"])
        cast_type = cast_api.split("to")[1].lower()

        cast_api_sub_nodes = nx.dfs_tree(ast, cast_api_node)
        cast_api_sub_ast = nx.DiGraph(nx.subgraph(ast, [node for node in cast_api_sub_nodes]))
        ast.remove_nodes_from(cast_api_sub_nodes) 

        # 1. functionCall ==> typeConversion
        ast.nodes[sbp_node]["expr"] = "typeConversion"
        ast.nodes[sbp_node]["label"] = "typeConversion  @ID:{}".format(cast_api_node)
       
        # 2. delete the cast_api_node
        node_to_remove.append(cast_api_node)

        # 3. add two new subnodes to cast_api_node
        new_edges = []

        # 3.1 fist new node
        new_pid_1 = sbp_node
        new_cid_1 = "{}@{}".format(sbp_node, 1)
        new_content_1 = "type({})".format(cast_type)
        new_type_1 = "ElementaryTypeNameExpression"
        new_lable_1 = "{}  {}".format(new_content_1, new_cid_1)
        ast.add_node(new_cid_1, label=new_lable_1, expr=new_content_1, ast_type=new_type_1, pid=new_pid_1)
        new_edges.append((new_pid_1, new_cid_1))

        # 3.2 second new node
        new_pid_2 = new_cid_1
        new_cid_2 = "{}@{}".format(sbp_node, 2)
        new_content_2 = "{}".format(cast_type)
        new_type_2 = "ElementaryTypeName"
        new_lable_2 = "{}  {}".format(new_content_2, new_cid_2)
        ast.add_node(new_cid_2, label=new_lable_2, expr=new_content_2, ast_type=new_type_2, pid=new_pid_2)
        new_edges.append((new_pid_2, new_cid_2))

        # 3.2 add new edges
        ast.add_edges_from(new_edges)

        for cast_node in cast_api_sub_ast.nodes:
            cid = cast_node
            label = cast_api_sub_ast.nodes[cast_node]["label"]
            expr = cast_api_sub_ast.nodes[cast_node]["expr"]
            ast_type = cast_api_sub_ast.nodes[cast_node]["ast_type"]
            pid = cast_api_sub_ast.nodes[cast_node]["pid"]
            ast.add_node(cid, label=label, expr=expr, ast_type=ast_type, pid=pid)
        
        # ast.add_nodes_from(cast_api_sub_ast.nodes)
        ast.add_edges_from(cast_api_sub_ast.edges)    
        ast.add_edge(sbp_node, cast_api_node)


    def _normalize_transaction_orde_dependency(self, ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
        """
            TOD漏洞适用的最佳实践:
                token.approve(spender, value)  ==> 常见
                approve(token, spender, value)
        """
        subnodes = [subnode for subnode in ast.successors(sbp_node)]
        approve_api_node = subnodes[0]
        if ast.nodes[approve_api_node]["expr"] in ["safeIncreaseAllowance", "safeDecreaseAllowance"]:
            
            # change the api to approve
            ast.nodes[approve_api_node]["expr"] = "approve"

        # token.approve(spender, value? or 0) --> 需要将整个expression删除:最后删除,保持CFG的不改变
        elif ast.nodes[approve_api_node]["expr"] in ["approve", "safeApprove"]:
            value_node = subnodes[2]

             # if there is token.approve(spender, 0). Its a SBP and we have to delete it
            if ast.nodes[value_node]["expr"] == "0":
                nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, sbp_node))
                 
            else:
                # normalize the api to approve
                ast.nodes[approve_api_node]["expr"] = "approve"    
    
    def _normalize_modifier(self, ast: nx.DiGraph, sbp_node, node_to_remove, directly_remove_nodes):
        """
            When a function call the SBP modifier: nonReentrant/ownlyOwner
                -- Remove the hole modifier 
        """
        directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, sbp_node))


    def _normalize_resumable_loop(self, ast: nx.DiGraph, sbp_node, gaslef_infos, node_to_remove, directly_remove_nodes):
        
        # [{"gasleft_id":xx, "closest_stmt_id":xxx}]
        for gasleft_info in gaslef_infos:
            stmt_id = gasleft_info["closest_stmt_id"]
            gasleft_expr_id = gasleft_info["gasleft_id"]

            # if the gasleft inside the for(condition), just remove the subnode of the condition
            if ast.nodes[stmt_id]["expr"] in ["ForStatement", "WhileStatement", "DoWhileStatement"]:

                # the conditions
                for sub_node in ast.neighbors(stmt_id):
                    if nx.has_path(ast, sub_node, gasleft_expr_id):
                        directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, sub_node))
                        break
            else:
                directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, stmt_id))

    def normalize_sbp(self, expr_info, ast, expr_id, nodes_to_remove, nodes_to_remove_directly):
        """
            Normalize the AST node based on the sbp type

        """
        # normalize for safemath
        if expr_info["label"] == "SafeMath":
            self._normalize_safemath(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # normalize for low-level call
        if expr_info["label"] == "low-level call":
            self._normalize_llc(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # normalize for safe downcast
        if expr_info["label"] == "safe cast":
            self._normalize_safecast(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # normalize for transaction order dependency
        if expr_info["label"] == "transaction order dependency":
            self._normalize_transaction_orde_dependency(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # normalize for sbp modifier    
        if expr_info["label"] in ["nonReentrant", "onlyOwner"]:
            self._normalize_modifier(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # normalize for resumable_loop
        if expr_info["label"] == "resumable_loop" and "gasleft" in expr_info:
            self._normalize_resumable_loop(ast, expr_id, expr_info["gasleft"], nodes_to_remove, nodes_to_remove_directly)

class FunctionAstAnalyzer:
    
    def __init__(self, target_info,target_infos_collector:TrgetInfoCollector, log_level=logging.WARNING, is_modifier=False, save_png=1) -> None:
        
        self.target_infos_collector = target_infos_collector
        self.sbp_normalizer = SbpNormalizer()
        self.logger = self.logger_init(log_level)

        self.save_png = save_png
        self.is_modifier = is_modifier
        self.ast_json_file_name = target_info["file_name"]
        self.sample_dir_with_path = target_info["dir"]
        
        self.ast_file = self.sample_dir_with_path + self.ast_json_file_name
        self.sbp_file = self.sample_dir_with_path + "sbp_info.json"

        # 语法相关
        self.in_param_cnt = None
        self.ast_root  = None
        self.ast_graph = None
        self.normalized_ast_graph = None
        
         # 语义相关
        self.cfg_slither:SFunction = None
        self.cfg_info = None
        self.cfg:nx.DiGraph = None
        self.normalized_cfg:nx.DiGraph = None
        self.final_cfg:nx.DiGraph = None

        # 解决AST与CFG之间无法对齐的问题
        self.cfg_astid_offset = 0 

        # 函数本身特征
        self.simple_key = None
        self.cfg_key = None
        self.c_name = None
        self.f_name = None
        self.ast_id = None
        
        self.stmts_type_map = {}
        self.cfg_id_to_ast_id = {}
        self.ast_id_to_cfg_id = {}
        self.cfg_nodes_map = {}  # 当前函数的CFG节点信息
        self.modifier_infos = {}

        self.entry_ast_id = None
        self.entry_ast_info = {}
        self.statements_ast = None

        self.get_cfg_key()
    
    def logger_init(self, log_level):
        logger = logging.getLogger('FA')
        if log_level == logging.DEBUG:
            logger.addHandler(logging.FileHandler("log.log", mode='a'))

        logger.setLevel(log_level)
        return logger

    def get_cfg_key(self):

        smaple_infos = str(self.ast_json_file_name).split(".json")[0].split("-")
        self.c_name = smaple_infos[0]
        self.f_name = smaple_infos[1]
        self.ast_id = smaple_infos[2]

        self.simple_key = "{}-{}".format(self.c_name, self.f_name)

        if self.is_modifier:
            self.cfg_key = "{}-{}-modifier".format(self.c_name, self.f_name)
        else:
            self.cfg_key = "{}-{}-{}".format(self.c_name, self.f_name, self.ast_id)

        self.logger.debug("\r\n================================\r\n")
        self.logger.debug("\r\nInit for :{}".format(self.cfg_key))
        
        return self.cfg_key
    
    def get_function_entry_info(self):
        self.entry_ast_info =  self.normalized_ast_graph.nodes[int(self.entry_ast_id)]
        self.logger.debug("\r\nentry_ast_info :{}".format(self.entry_ast_info))
    
    def set_stmts_types_in_cfg(self):
        for cfg_node_id in self.normalized_cfg.nodes:
            old_lable = self.normalized_cfg.nodes[cfg_node_id]["label"]
            ast_id = self.normalized_cfg.nodes[cfg_node_id]["ASTID"]
            if str(ast_id) in self.stmts_type_map:
                stmt_type = self.stmts_type_map[str(ast_id)]
                self.normalized_cfg.nodes[cfg_node_id]["label"] = old_lable + " @{}".format(stmt_type)


    def get_stmts_types(self):
        
        tmp_cfg = self.normalized_cfg
        tmp_ast = self.normalized_ast_graph

        # 解合AST属性和CFG属性得到自定义的语句属性
        for cfg_node_id in tmp_cfg.nodes:

            if int(tmp_cfg.nodes[cfg_node_id]["ASTID"]) not in tmp_ast.nodes:
                continue # 如果cfg中存在但是ast中不存在: 情况是end_if  e.g. 0xd78C4b1c1ff1d2Fbb069e29379540FA59d5a11C1
            
            cfg_info = tmp_cfg.nodes[cfg_node_id]
            ast_node_id = int(cfg_info["ASTID"])
            ast_info = tmp_ast.nodes[ast_node_id]

            # 基于规则的语句属性构建
            if cfg_info["stmt_type"] == "ENTRY_POINT":
                final_stmt_type = "ENTRY_POINT"
            
            elif cfg_info["stmt_type"] == "RETURN":
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "IF":
                final_stmt_type = ast_info["ast_type"]

            elif cfg_info["stmt_type"] == "NEW VARIABLE":
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "INLINE ASM":
                final_stmt_type = ast_info["ast_type"]

            elif cfg_info["stmt_type"] == "IF_LOOP":  # 循环条件
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "THROW":
                final_stmt_type = ast_info["ast_type"]

            elif cfg_info["stmt_type"] == "BREAK":
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "CONTINUE":
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "_":
                final_stmt_type = ast_info["ast_type"]
                
            elif cfg_info["stmt_type"] == "TRY":
                final_stmt_type = ast_info["ast_type"]

            elif cfg_info["stmt_type"] == "CATCH":
                final_stmt_type = ast_info["ast_type"]

            elif cfg_info["stmt_type"] == "BEGIN_LOOP":  # 循环开始: ForStatement; WhileStatement
                final_stmt_type = ast_info["ast_type"]
                self.logger.debug("BEGIN_LOOP---->{}".format(final_stmt_type))

            elif cfg_info["stmt_type"] == "EXPRESSION":
                if ast_info["ast_type"] == "EmitStatement":
                    final_stmt_type = "EmitStatement"
                else:
                    expr_sub_nodes = [subnode for subnode in tmp_ast.successors(ast_node_id)]
                    if len(expr_sub_nodes) == 1:
                        
                        sub_stmt_ast_type = tmp_ast.nodes[expr_sub_nodes[0]]["ast_type"]
                        self.logger.debug("[{}] --> EXPRESSION---->{}".format(expr_sub_nodes[0], sub_stmt_ast_type))

                        if sub_stmt_ast_type == "Assignment":
                            if cfg_info["state_assign"] == "True":
                                final_stmt_type = "Storage Assignment"
                            else:
                                final_stmt_type = "Memory Assignment"
                        
                        elif sub_stmt_ast_type == "FunctionCall":
                            if cfg_info["ext_call"] == "True":
                                final_stmt_type = "External FunctionCall"
                            else:
                                final_stmt_type = "Internal FunctionCall"
                        
                        else:
                            final_stmt_type = sub_stmt_ast_type

                    elif len(expr_sub_nodes) > 1:

                        # VariableDeclaration: 存在多个子节点
                        #  (bool ret, ) = address(TOKEN).call("tag").value(amount);
                        #  VariableDeclarationStatement  --> VariableDeclaration
                        #                                --> functionCall
                        self.logger.debug("[{}] [{}-{}] --> EXPRESSION---->{} {}".format(
                            str(ast_node_id), expr_sub_nodes[0], expr_sub_nodes[1], 
                            tmp_ast.nodes[expr_sub_nodes[0]]["ast_type"], 
                            tmp_ast.nodes[expr_sub_nodes[1]]["ast_type"]
                        ))
                        
                        if tmp_ast.nodes[expr_sub_nodes[0]]["ast_type"] == "VariableDeclaration":
                            final_stmt_type = tmp_ast.nodes[expr_sub_nodes[1]]["ast_type"]
                            if final_stmt_type == "FunctionCall":
                                if cfg_info["ext_call"] == "True":
                                    final_stmt_type = "External FunctionCall"
                                else:
                                    final_stmt_type = "Internal FunctionCall"
                        
                        else:
                            self.logger.warning("CFG EXPRESSION节点【{}】存在多个子节点".format(ast_node_id))
                        
                    else:
                        # ERC20(token).approve(Alice, 0)  --> EXPRESSION1:删除了子树而没有删除根节点导致的
                        # ERC20(token).approve(Alice, value) --> EXPRESSION2
                        self.logger.error("发现了无子节点的EXPRESSION: {}".format(ast_node_id))
            
            elif cfg_info["stmt_type"] in ["END_IF", "END_LOOP", "OTHER_ENTRYPOINT"]:
                # 不分析: 原因是END_IF和condition是同一个AST节点, 避免覆盖
                final_stmt_type = "pass_tag"
           
            
            # 非pass的，需要记录类型
            if final_stmt_type != "pass_tag":
                self.stmts_type_map[str(ast_node_id)] = final_stmt_type

    def _save_stmt_ast_to_json(self, stmt_ast_graph:nx.DiGraph, stmt_ast_json, stmt_ast_root, vul_label, vul_type):
        """
        transfer a ast graph of a statement into a json format
        """
        nodes = {}
        edges = []

        for node_id in stmt_ast_graph.nodes:

            nodes[node_id] = {
                "id": node_id,
                "label": stmt_ast_graph.nodes[node_id]["label"],
                "content": stmt_ast_graph.nodes[node_id]["expr"],
                "ast_type": stmt_ast_graph.nodes[node_id]["ast_type"],
                "pid": stmt_ast_graph.nodes[node_id]["pid"]
            }

        for edge in stmt_ast_graph.edges:
            edges.append({"from":edge[0], "to":edge[1]})
        
        stmt_ast_json[stmt_ast_root] = {"vul":vul_label, "vul_type":vul_type, "nodes":nodes, "edges":edges}

    
    def _record_function_modifier(self, ast_node):

        contract_name = ast_node["info"]["contract_name"]
        modifier_name = ast_node["info"]["modifier_name"]

        sbp_modifier_tag = 0
        if modifier_name in ["nonReentrant", "onlyOwner"]:
            sbp_modifier_tag = 1
        
        key = "{}-{}".format(contract_name, modifier_name)
        self.modifier_infos[key] = {
            "c_name": contract_name,
            "m_name": modifier_name,
            "sbp_tag": sbp_modifier_tag
        }
        
        self.logger.debug("==== modifier: {}".format(key))
    
    def _construct_ast_from_json_file(self):
        """
            Construct the AST from the json file
        """
        if os.stat(self.ast_file).st_size == 0:
            return None

        with open(self.ast_file, "r") as f:

            ast = nx.DiGraph()
            ast_infos = json.load(f)
            
            edges = []
            for ast_node in ast_infos:
                if "content" not in ast_node:
                    print("type:{} ID:{} file:{}".format(ast_node["type"], ast_node["cid"], self.ast_file))
                    raise RuntimeError("error!!!")

                content = ast_node["content"]
                ast_type = ast_node["type"]
                cid = ast_node["cid"]
                pid = ast_node["pid"]

                # get the modifier AST file name as: <Gauge-updateReward-MOD-1394.json>
                if ast_type == "ModifierInvocation" and "info" in ast_node:
                   self._record_function_modifier(ast_node)

                label_content = "{}  @ID:{}".format(content, cid)
                ast.add_node(cid, label=label_content, expr=content, ast_type=ast_type, pid=pid)
                
                if ast_type not in ["FunctionDefinition", "ModifierDefinition"]:
                    edges.append((pid, cid))
                else:
                    root = cid  # the def node is the root node of current AST

            ast.add_edges_from(edges)
            self.ast_root = root
            self.ast_graph = ast
        
        return ast

    def _adjunction_ast_infos(self):
        """
            补充AST信息
        """
        top_block_node = 0
        top_stmt_nodes = []

        self.ast_graph.graph["cfg_supplement_stmts"] = [] # CFG中存在但是AST—top_stmts中不存在的语句
        self.ast_graph.graph["root"] = self.ast_root
        self.ast_graph.graph["name"] = self.ast_json_file_name

        # 根节点的一节邻居
        for block_node in nx.neighbors(self.ast_graph, self.ast_graph.graph["root"]):
            if self.ast_graph.nodes[block_node]['expr'] == "Block":
                top_block_node = block_node
                for stmt_node in nx.neighbors(self.ast_graph, block_node):
                    top_stmt_nodes.append(stmt_node)
        
        self.ast_graph.graph["top_block"] = top_block_node  # ast的top_block就是cfg的entry_point
        self.ast_graph.graph["top_stmts"] = top_stmt_nodes
    
    def _input_params_infos(self):

        input_param_root = [subnode for subnode in self.ast_graph.successors(self.ast_root)][0]
        if self.ast_graph.nodes[input_param_root]["ast_type"] != "ParameterList":
            self.in_param_cnt = 0
        else:
            self.in_param_cnt = len([subnode for subnode in self.ast_graph.successors(input_param_root)])
        
        self.logger.debug("入参数量:{}".format(self.in_param_cnt))
    
    def normalize_sbp_in_ast(self):
        """
            Normalize the AST based on its sbp file
        """
        
        _to_normal_ast_graph = nx.DiGraph(self.ast_graph)
        sbp_file = self.sbp_file
        
        if not os.path.exists(self.sbp_file):
            if self.is_modifier:
                self.logger.warn("NO SBP FOR modifier? {}".format(self.is_modifier))
                self.normalized_ast_graph = _to_normal_ast_graph
                return
            else:
                raise RuntimeError("No SBP info for the target function")

        f = open(sbp_file, "r")
        function_sbp_infos = json.load(f)["function_sbp_infos"]

        nodes_to_remove = []  # 需要删除的节点再中间，删除节点后需要重新添加边
        nodes_to_remove_directly = []  # 删除节点后不需要重新添加边：即当前节点和其字节的全部删除

        for sbp_info in function_sbp_infos:

            expr_id = sbp_info["expr_id"]
            expr_lable_info = _get_stmt_label_info(sbp_info["lable_infos"])  # 为了适配老版本 -- sbp_info["lable_infos"]可能是数组也可能是map

            # 打标签: modifier类的标签打在entry上
            if expr_lable_info["label"] in ["nonReentrant", "onlyOwner"]:
                tag_node_id = self.ast_graph.graph["top_block"]
            else:
                tag_node_id = expr_id

            # 节点打标签: save the sbp info into the AST node
            _to_normal_ast_graph.nodes[tag_node_id]["vul_type"] = {tag_node_id: expr_lable_info["label"]}
            _to_normal_ast_graph.nodes[tag_node_id]["color"] = "red"  # change the node color to red
            _to_normal_ast_graph.nodes[tag_node_id]["vul"] = 1

            self.ast_graph.nodes[tag_node_id]["vul_type"] = {tag_node_id: expr_lable_info["label"]}
            self.ast_graph.nodes[tag_node_id]["color"] = "red"
            self.ast_graph.nodes[tag_node_id]["vul"] = 1
            
            # normalize the sbp expression
            self.sbp_normalizer.normalize_sbp(expr_lable_info, _to_normal_ast_graph, expr_id, nodes_to_remove, nodes_to_remove_directly)

        # 1.可以直接删除的节点：
        _to_normal_ast_graph.remove_nodes_from(nodes_to_remove_directly)

        # 2.间接删除，需要重新连接边
        for node in nodes_to_remove:
            if _to_normal_ast_graph.has_node(node):
                _to_normal_ast_graph = _do_remove_node(_to_normal_ast_graph, node)

        # 3.图的记录特征删除
        for stmt_id in _to_normal_ast_graph.graph["top_stmts"]:
            if stmt_id in nodes_to_remove_directly or stmt_id in nodes_to_remove:
                _to_normal_ast_graph.graph["top_stmts"].remove(stmt_id)
        
        # 记录
        self.normalized_ast_graph = _to_normal_ast_graph


    def normalize_sbp_in_cfg(self):
        """
            1.AST中删除的语句级别节点, 在CFG中也要相应的删除;
            2.modifier也要从cfg中删除;
        """
        normalized_cfg = nx.DiGraph(self.cfg)

        for cfg_node_ast_id in self.cfg_nodes_map:
            if self.cfg_nodes_map[cfg_node_ast_id]["tag"] in ["normalized_remove", "modifier_remove"]:
                    
                cfg_id = str(self.cfg_nodes_map[cfg_node_ast_id]["cfg_id"])
                if normalized_cfg.has_node(cfg_id):
                    normalized_cfg = _do_remove_node(normalized_cfg, cfg_id)
        
        self.normalized_cfg = normalized_cfg 
        self.normalized_cfg.graph["leaves"] = _get_all_leaf_nodes(self.normalized_cfg)


    def cfg_supplement_stmts_for_ast(self):

        cfg_supplement_stmts = []
        self.cfg_nodes_map.clear()
        for cfg_node in self.cfg_info["nodes"]:
            self.cfg_nodes_map[cfg_node["ast_id"]] = {"content":cfg_node["node_type"], "cfg_id":cfg_node["cfg_id"] ,"tag":""}
        
        ast_top_stmts = {}
        for ast_stmt in self.normalized_ast_graph.graph["top_stmts"]:
            ast_top_stmts[ast_stmt] = 1
        
        for cfg_node_ast_id in self.cfg_nodes_map:

            # CFG中存在非ast_top_stmt的节点: 例如 if(cond) {[blocks]}
            if cfg_node_ast_id not in ast_top_stmts:

                # 非 AST节点：已被删除
                if cfg_node_ast_id not in self.normalized_ast_graph.nodes:
                    self.cfg_nodes_map[cfg_node_ast_id]["tag"] = "normalized_remove"
                    pass

                # 入口节点：pass
                elif self.cfg_nodes_map[cfg_node_ast_id]["content"] == "ENTRY_POINT":
                    self.cfg_nodes_map[cfg_node_ast_id]["tag"] = "entry_remove"
                    pass

                # modifer节点：pass
                elif self.normalized_ast_graph.nodes[cfg_node_ast_id]["expr"] == 'ModifierInvocation':
                    self.cfg_nodes_map[cfg_node_ast_id]["tag"] = "modifier_remove"
                    pass 

                # 常规CFG节点
                else:
                    self.cfg_nodes_map[cfg_node_ast_id]["tag"] = "recored" 
                    cfg_supplement_stmts.append(cfg_node_ast_id)

        # save the cfg_supplement_stmts
        self.normalized_ast_graph.graph["cfg_supplement_stmts"] = cfg_supplement_stmts
        self.normalized_ast_graph.graph["cfg_entry_ast_id"] = self.entry_ast_id

    def construct_ast_for_function_sample(self):
        self.logger.debug("construct ast for :{}".format(self.cfg_key))
        self._construct_ast_from_json_file()
        self._adjunction_ast_infos()
        self._input_params_infos()
    
    def construct_up_buttom_cfg_for_modifier(self):

        duplicat = {}
        subg_nbatch = []

        # 复制CFG 作为分割的基础    
        modifier_cfg = nx.DiGraph(self.cfg)
        place_holder = modifier_cfg.graph["place_holder"]   # modifier的 _ id

        # 上半段
        paths = nx.all_simple_paths(modifier_cfg, "0", place_holder)
        for path in paths:
            for node in list(path):
                if node not in duplicat:
                    duplicat[node] = 1
                    subg_nbatch.append(node)

        upper_half = nx.subgraph(modifier_cfg, subg_nbatch)
        upper_half.graph["name"] = modifier_cfg.graph["name"]
        upper_half.graph["place_holder"] = place_holder

        # 下半段准备工作
        duplicat.clear()
        subg_nbatch.clear()
        
        # NOTE:添加虚拟退出节点 => 下半段需要保存exit节点
        leafs = _get_all_leaf_nodes(modifier_cfg) # 寻找叶子节点
        modifier_cfg.add_node("exit", label="EXIT_POINT", ASTID="exit", mk=modifier_cfg.nodes["0"]["mk"])
        new_edges = [(leaf, "exit") for leaf in leafs]
        modifier_cfg.add_edges_from(new_edges)
        
        # 下半段
        paths = nx.all_simple_paths(modifier_cfg, place_holder,  "exit")
        for path in paths:
            for node in path:
                if node not in duplicat:
                    duplicat[node] = 1
                    subg_nbatch.append(node)
        bottom_half:nx.DiGraph = nx.subgraph(modifier_cfg, subg_nbatch)
        bottom_half.graph["name"] = modifier_cfg.graph["name"]
        bottom_half.graph["place_holder"] = place_holder
        
        if self.save_png:
            self.save_cfg_as_png(postfix="upper", _graph=upper_half)
            self.save_cfg_as_png(postfix="bottom", _graph=bottom_half)

        return upper_half, bottom_half

    def construct_cfg_for_function_sample(self, is_modifier=False, postfix="cfg", ):
        
        place_holder = None

        name_prefix = "{}-{}-{}".format(self.c_name, self.f_name, self.ast_id)
        if self.is_modifier is True:
            modifier_key = "{}-{}-{}".format(self.c_name, self.f_name, "modifier")
        else:
            modifier_key = "not-modifier"

        dot_name = self.sample_dir_with_path + "cfg_dot//" + "{}-{}.dot".format(name_prefix, postfix)
        with open(dot_name, "w", encoding="utf8") as f:
            f.write("digraph{\n")
            for node in self.cfg_slither.nodes:
                
                ext_call_flag, state_assign_flag = _get_function_call_info(node, debug=None)

                self.cfg_id_to_ast_id[str(node.node_id)] = str(node.node_ast_id)
                self.ast_id_to_cfg_id[str(node.node_ast_id)] = str(node.node_id)

                if is_modifier and "_" == str(node.type):
                    place_holder = str(node.node_id)
                
                if node.expression is None:
                    label_info = "{}  @{} @{}".format(str(node), node.node_ast_id, node.node_id)
                else:
                    label_info = "{}  @{} @{}".format(str(node.expression), node.node_ast_id, node.node_id)
                
                _ast_id = int(node.node_ast_id) + self.cfg_astid_offset # 对齐
                f.write (
                    f'{node.node_id}[label="{label_info}"][expr="{str(node)}"][ASTID="{str(_ast_id)}"]\
                        [stmt_type="{str(node.type)}"]\
                        [state_assign="{str(state_assign_flag)}"]\
                        [ext_call="{str(ext_call_flag)}"]\
                        [mk="{str(modifier_key)}"];\n' # mk: modifier key的缩写，表明当前节点是否来自modifier
                )
                
                for son in node.sons:
                    f.write(f"{node.node_id}->{son.node_id};\n")
            
            f.write("}\n")

        self.cfg = nx.drawing.nx_agraph.read_dot(dot_name)
        self.cfg.graph["name"] = self.ast_json_file_name 
        self.entry_ast_id = self.cfg.graph["entry_point_ast_id"] = self.cfg.nodes["0"]["ASTID"]
        self.cfg.graph["place_holder"] = place_holder
        
        # 如果是modifier 必须记录下来 以方便后续分析
        if is_modifier is True:
            self.logger.debug("开始modifier控制流图分解阶段")
            upper_half, bottom_half = self.construct_up_buttom_cfg_for_modifier()  # Modifier CFG 分割
            key = "{}-{}-modifier".format(self.c_name, self.f_name)
            self.target_infos_collector.set_cfg_graph_by_key(self.cfg, upper_half, bottom_half, key, is_modifier)

        # save the cfg or not    
        if not self.save_png: return
        png_name = self.sample_dir_with_path + "cfg_png//" + "{}-{}.png".format(name_prefix, postfix)
        subprocess.check_call(["dot", "-Tpng", dot_name, "-o", png_name])

    def concat_function_modifier_cfg(self):
        
        if self.is_modifier:
            self.logger.debug("这是一个modifier, 不需要连接CFG")
            self.final_cfg = self.normalized_cfg
            return

        # 预检测：如果缺失了modifier的CFG文件--> slither不支持
        for modifier_key in self.modifier_infos:
            if not self.modifier_infos[modifier_key]["sbp_tag"]:
                _info = self.modifier_infos[modifier_key]
                key = "{}-{}-modifier".format(_info["c_name"], _info["m_name"])
                if self.target_infos_collector.get_cfg_graph_by_key(key, is_modifier=1) is None:
                    self.logger.error("跳过: 发现了modifier的缺失: {}".format(key) )
                    self.final_cfg = self.normalized_cfg
                    return

        # 开始链接操作
        self.logger.debug("=====>开始CFG连接操作....<=======")

        function_cfg:nx.DiGraph = self.normalized_cfg

        # 获得当前使用的modifier
        for modifier_key in self.modifier_infos:
            if not self.modifier_infos[modifier_key]["sbp_tag"]:

                _info = self.modifier_infos[modifier_key]
                key = "{}-{}-modifier".format(_info["c_name"], _info["m_name"])
                modifier_cfgs_map = self.target_infos_collector.get_cfg_graph_by_key(key, is_modifier=1)

                modifier_up_cfg = modifier_cfgs_map["up"]
                modifier_buttom_cfg = modifier_cfgs_map["buttom"]

                # 连接 up fun buttom
                new_cfg:nx.DiGraph = nx.union_all((modifier_up_cfg, function_cfg, modifier_buttom_cfg), rename=("up-", "f-", "bu-"))
                
                # new edge 1   up_place_holder ==> f_entry
                up_place_holder = "up-{}".format(modifier_up_cfg.graph["place_holder"])
                f_entry = "f-{}".format(0)
                new_cfg.add_edge(up_place_holder, f_entry)

                # new edge 2   f_leaves ==> bu_place_holder
                bu_place_holder = "bu-{}".format(modifier_buttom_cfg.graph["place_holder"])
                for leaf in function_cfg.graph["leaves"]:
                    f_leaf = "f-{}".format(leaf)
                    new_cfg.add_edge(f_leaf, bu_place_holder)

                # 清除没用的部分: palce_holder
                new_cfg = _do_remove_node(new_cfg, up_place_holder)
                new_cfg = _do_remove_node(new_cfg, bu_place_holder)
                new_cfg.graph["name"] = self.normalized_cfg.graph["name"]
                
            if self.save_png:
                self.save_cfg_as_png(postfix="final", _graph=new_cfg)

            self.final_cfg = new_cfg
            return

        # 默认返回normalized_cfg
        self.final_cfg = self.normalized_cfg
        return
        
           
    def split_function_ast_stmts(self):
        """
            分解AST,得到语句粒度AST表示
            1. 分解原始ast中的top_stmt
            2. 根据cfg再细分top_stmt
        """

        ast_graph = self.normalized_ast_graph
        
        cfg_stmt_map = {}
        ast_stmt_map = {}
        function_stmts_map = {}
        debug_temp = []

        # AST中语句分割: 一次分割
        new_ast_graph = nx.DiGraph(ast_graph)
        _, function_stmts_ast = _split_function_stmts(new_ast_graph)

        debug_temp.clear()
        for stmt_ast in function_stmts_ast:
            debug_temp.append(stmt_ast.graph["root"])
            ast_stmt_map[stmt_ast.graph["root"]] = stmt_ast
        self.logger.debug("  一阶段分割:{}".format(debug_temp))
        

        # 语句再分割: 二次分割
        # 1. 二次分割节点之间嵌套关系抽取
        sub_stmts = ast_graph.graph["cfg_supplement_stmts"]
        self.logger.debug("  二阶段分割CFG补全:{}".format(sub_stmts))
        root_split = {}

        # 1.1 从属于同一ast_root_stmt
        debug_temp.clear()
        for sub_stmt in sub_stmts:
            for top_stmt in ast_graph.graph["top_stmts"]:
                if nx.has_path(ast_graph, top_stmt, sub_stmt):
                    if top_stmt not in root_split:
                        root_split[top_stmt] = [sub_stmt]
                    else:
                        root_split[top_stmt].append(sub_stmt)

        self.logger.debug("  二阶段分割root_split:{}".format(root_split))

        # 分割原则: 依照leaf-->root的顺序进行分割
        # AST由于是树状结构，点之间的路径有且只有一条
        for top_stmt in root_split:
            self.logger.debug("   开始排序 root:{}".format(top_stmt))

            sub_stmts_ordered_by_length = []

            # 计算 sub_node 到 root 的距离,并进行反向排序
            stmt_ast:nx.DiGraph = ast_stmt_map[top_stmt]
            for sub_stmt in root_split[top_stmt]:
                length = nx.shortest_path_length(stmt_ast, top_stmt, sub_stmt)
                sub_stmts_ordered_by_length.append((sub_stmt, length))
            sub_stmts_ordered_by_length.sort(key=_do_take_second, reverse=True)

            self.logger.debug("     依照距离排序:{}".format(sub_stmts_ordered_by_length))   

            # 依照排序的结果进行语句分割
            for (sub_stmt, _) in sub_stmts_ordered_by_length:
                sub_nodes = nx.dfs_tree(stmt_ast, sub_stmt)
                sub_stmt_ast = nx.DiGraph(nx.subgraph(stmt_ast, [node for node in sub_nodes]))
                sub_stmt_ast.graph["root"] = sub_stmt
                sub_stmt_ast.graph["name"] = "stmt_at_{}".format(sub_stmt)
                cfg_stmt_map[sub_stmt] = sub_stmt_ast
                stmt_ast.remove_nodes_from(sub_nodes) 
        
        # 记录原始AST中的语句
        for stmt_root in ast_stmt_map:
            function_stmts_map[stmt_root] = _remove_useless_leaf(ast_stmt_map[stmt_root])
            self.save_ast_as_png(postfix="splited", _graph=function_stmts_map[stmt_root])

        # 记录CFG中补全的语句
        for stmt_root in cfg_stmt_map:
            function_stmts_map[stmt_root] = _remove_useless_leaf(cfg_stmt_map[stmt_root])
            self.save_ast_as_png(postfix="splited", _graph=function_stmts_map[stmt_root])

        self.statements_ast = function_stmts_map
        return function_stmts_map
    
    def check_split_function_ast_stmts(self):
        for cfg_node in self.cfg_nodes_map:
            if self.cfg_nodes_map[cfg_node]["tag"] == "recored":
                if cfg_node not in self.statements_ast:
                    self.logger.warning("!!! CFG:{} not in stmt".format(cfg_node))

    def ast_slither_id_align(self):

        cnt_key = self.in_param_cnt
        self.cfg_slither =  self.target_infos_collector.get_slither_cfg_info_before_align(self.cfg_key, self.simple_key, cnt_key, self.is_modifier, "slither")
        
        ast_entry = int(self.ast_graph.graph["top_block"])
        cfg_entry = int(self.cfg_slither.entry_point.node_ast_id)
        self.cfg_astid_offset = ast_entry - cfg_entry
        
        self.cfg_info = self.target_infos_collector.get_slither_cfg_info_before_align(self.cfg_key, self.simple_key, cnt_key, self.is_modifier, "cfg_info")
        if self.cfg_info["offset"] == 0: 
            if self.cfg_astid_offset != 0:
                for cfg_node in self.cfg_info ["nodes"]:
                    cfg_node["ast_id"] = cfg_node["ast_id"] + self.cfg_astid_offset  # 对齐,加上offest
            
            self.cfg_info["offset"] = 1
        
        self.logger.debug("!! offset:{} ast_entry:{} cfg_entry:{}".format(self.cfg_astid_offset, ast_entry, cfg_entry))


    def save_ast_as_png(self, postfix="", _graph:nx.DiGraph=None):

        if not self.save_png:
            return 

        if _graph != None:
            graph = _graph
        
        elif postfix == "":
            graph = self.ast_graph

        elif postfix == "normalized":
            graph = self.normalized_ast_graph

        else:
            self.logger.error("!! WRONG SAVE PNG")
            return

        dot_name = self.sample_dir_with_path + "ast_dot//" + "{}-{}.dot".format(graph.graph["name"], postfix)
        png_name = self.sample_dir_with_path + "ast_png//" + "{}-{}.png".format(graph.graph["name"], postfix)

        nx_dot.write_dot(graph, dot_name)
        subprocess.check_call(["dot", "-Tpng", dot_name, "-o", png_name])
    
    def save_cfg_as_png(self, postfix="", _graph:nx.DiGraph=None):

        if not self.save_png:
            return 
        
        if _graph != None:
            graph = _graph

        elif postfix == "":
            graph = self.cfg
        
        elif postfix == "normalized":
            graph = self.normalized_cfg

        elif postfix == "typed_normalized":
            graph = nx.DiGraph(self.normalized_cfg)
            for cfg_node_id in graph.nodes:
                ast_node_id = self.cfg_id_to_ast_id[cfg_node_id]

                if ast_node_id in self.stmts_type_map: # end_if 可能不在
                    node_type = self.stmts_type_map[ast_node_id]
                    node_label = graph.nodes[cfg_node_id]["label"]
                    graph.nodes[cfg_node_id]["label"] = node_label + " @{}".format(node_type)

        else:
            self.logger.error("!! WRONG SAVE PNG")
            return
        
        dot_name = self.sample_dir_with_path + "cfg_dot//" + "{}-{}.dot".format(graph.graph["name"], postfix)
        png_name = self.sample_dir_with_path + "cfg_png//" + "{}-{}.png".format(graph.graph["name"], postfix)

        nx_dot.write_dot(graph, dot_name)
        subprocess.check_call(["dot", "-Tpng", dot_name, "-o", png_name])
    
    def save_statements_json_infos(self):

        final_stmts_ast_json = {}
        cfg_edge_map = {}
        cfg_edge_info = []

        # function的entry point信息生成
        entry_info = _do_save_entry_point_to_json(self.entry_ast_id, self.entry_ast_info, final_stmts_ast_json)
        if self.is_modifier is True:
            self.target_infos_collector.set_modifier_stmts(self.entry_ast_id, entry_info)

        # 保存来自function的语句
        for stmt_root_ast_id in self.statements_ast:

            stmt_ast:nx.DiGraph = self.statements_ast[stmt_root_ast_id]

            if str(stmt_root_ast_id) in self.stmts_type_map:
                stmt_type = self.stmts_type_map[str(stmt_root_ast_id)] 
            else:
                stmt_type = "EXPRESSION" # 默认为一个语句 LUBU_Inu-setSwapAndLiquifyByLimitOnly-2163 @0x62ca828e17b9c3C36D3bCFe0bf6C474355f67C71

            vul_label = 0
            vul_type = ""
            
            for node_id in stmt_ast.nodes:
                if "vul" in stmt_ast.nodes[node_id]:
                    vul_label = 1
                    vul_type = stmt_ast.nodes[node_id]["vul_type"]
                    break
            
            stmt_ast_info = _do_save_stmt_ast_to_json(stmt_ast, final_stmts_ast_json, stmt_root_ast_id, vul_label, vul_type, stmt_type)
            
            # modifier的信息进行持久化保存，方便函数的分析
            if self.is_modifier is True:
                self.target_infos_collector.set_modifier_stmts(stmt_root_ast_id, stmt_ast_info)

        # 保存来自modifier的语句
        if self.is_modifier is False:
            for (u, v) in self.final_cfg.edges:

                if self.final_cfg.nodes[u]["mk"] != "not-modifier":
                    from_ast_id = self.final_cfg.nodes[u]["ASTID"]
                    from_json_info = self.target_infos_collector.get_modifier_stmts(from_ast_id)
                    if from_json_info is not None:
                        final_stmts_ast_json[str(from_ast_id)] = from_json_info
                    else:
                        print("error:!!!! 缺失AST:{}".format(from_ast_id))

                if self.final_cfg.nodes[v]["mk"] != "not-modifier":
                    to_ast_id = self.final_cfg.nodes[v]["ASTID"]
                    to_json_info = self.target_infos_collector.get_modifier_stmts(to_ast_id)

                    # 此处需要手动添加exit节点的语句信息
                    if to_ast_id == "exit" and to_json_info is None:
                        to_json_info = _do_create_exit_point_to_json(final_stmts_ast_json)
                        
                    if to_json_info is not None:
                        final_stmts_ast_json[str(to_ast_id)] = to_json_info
                    else:
                        print("error:!!!! 缺失AST:{}".format(to_ast_id))        

        for (u, v) in self.final_cfg.edges:
            from_id = self.final_cfg.nodes[u]["ASTID"]
            to_id = self.final_cfg.nodes[v]["ASTID"]
            
            if from_id != to_id and str(from_id) in final_stmts_ast_json and str(to_id) in final_stmts_ast_json:
                cfg_edge_info.append({"from": from_id, "to": to_id})
                cfg_edge_map[str(from_id)] = 1
                cfg_edge_map[str(to_id)] = 1
        
        # 检查CFG中的孤儿节点: AST中存在但是CFG中不存在
        orphan_nodes = []
        for ast_id in final_stmts_ast_json:
            if str(ast_id) not in cfg_edge_map:
                self.logger.error("ERROR: the ast:{} 是一个孤儿节点".format(ast_id))
                orphan_nodes.append(ast_id)
        for orphan_node_id in orphan_nodes:
            final_stmts_ast_json.pop(orphan_node_id) # 删除孤儿节点

        final_stmts_ast_json["cfg_edges"] = cfg_edge_info

        json_file = self.sample_dir_with_path + "statement_ast_infos.json"
        with open(json_file, "w+") as f:
            f.write(json.dumps(final_stmts_ast_json, indent=4, separators=(",", ":")))
    
    def clean_up(self):
        """
            避免文件夹过多, 删除所有的dot文件夹和无内容的png文件夹
        """
        if not self.save_png:
            return
        
        shutil.rmtree(self.sample_dir_with_path + "ast_dot//")
        shutil.rmtree(self.sample_dir_with_path + "cfg_dot//")

        for png_dir in ["ast_png//", "cfg_png//"]:
            dirContents = os.listdir(self.sample_dir_with_path + png_dir)
            if not dirContents:
                shutil.rmtree(self.sample_dir_with_path + png_dir)
