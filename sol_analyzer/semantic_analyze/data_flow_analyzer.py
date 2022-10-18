import networkx as nx

from sol_analyzer.info_analyze.contract_analyze import ContractInfo
from sol_analyzer.info_analyze.function_analyze import FunctionInfo


def _data_flow_reorder(current_stmt_vars):
    """
    调整当前语句变量使用情况分析顺序
    保证：先分析def 再分析use
    """
    use_vars = []
    def_vars = []

    # issue2: 由于数据流分析是逆向分析，导致变量关系需要先分析def 再分析use
    for var_info in current_stmt_vars:
        if var_info["op_type"] == "use":

            use_vars.append(var_info)
        elif var_info["op_type"] == "def":
            def_vars.append(var_info)

    return def_vars + use_vars


def _get_ddg_edges(data_dependency_relations, edge_type):
    duplicate = {}
    ddg_edges = []

    for edge_info in data_dependency_relations:

        if edge_info["from"] == edge_info["to"]:
            continue

        key = "{}-{}".format(edge_info["from"], edge_info["to"])
        if key not in duplicate:
            duplicate[key] = 1
            ddg_edges.append((edge_info["from"], edge_info["to"], {'color': "green", "type": edge_type}))

    # print("DEBUG 数据依赖：", ddg_edges)
    return ddg_edges


# 数据流分析器
class DataFlowAnalyzer:

    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo, simple=0):

        self.simple_mode = simple

        self.function_info = function_info
        self.contract_info = contract_info

        self.data_flow_map = None

        # 数据相关语义边：初始为None, 初始化后赋值为[]
        self.data_flow_edges = None  # 数据流边
        self.data_dependency_edges = None  # 数据依赖边
        self.reverse_data_dependency_edges = None  # 反向全局变量数据依赖边
        self.loop_data_dependency_edges = None  # 循环体数据依赖关系增强

        self.transaction_states = None  # 交易语句涉及的全局变量
        self.trans_state_append_criteria = None  # 根据交易全局变量依赖额外切片准则

    #######################################################
    # 函数内 数据流分析                                      #
    #######################################################
    def data_flow_analyze(self):
        """
        交易行为相关数据流分析
        transaction：<to, money>
        反向分析控制流，得到数据流
        """
        data_flow_edges = []
        data_flow_map = {}

        duplicate = {}
        def_info = {}
        use_info = {}

        # 便利所有可能CFG路径
        cfg = self.function_info.simple_cfg
        cfg_paths = nx.all_simple_paths(cfg, source="0", target="EXIT_POINT")
        for cfg_path in cfg_paths:

            def_info.clear()  # 当路径变量定义信息
            use_info.clear()  # 当路径变量使用信息

            # NOTE: 逆向分析每条CFG执行路径，从叶子向根部分析
            for from_node in reversed(cfg_path):

                if from_node == 'EXIT_POINT':
                    continue

                # 语句变量信息： [{"list": rechecked_read_state_vars, "type": "state", "op_type": "use"}]
                _current_stmt_vars = self.function_info.stmts_var_info_maps[from_node]
                current_stmt_vars = _data_flow_reorder(_current_stmt_vars)  # 调整顺序：先分析def 后分析use
                for var_info in current_stmt_vars:

                    # 如果当前语句有写操作，查询之前语句对该变量是否有读操作
                    if var_info["op_type"] == "def":
                        for var in var_info["list"]:

                            if var in use_info:
                                for to_node in use_info[var]:

                                    # 避免自循环 a += 1
                                    if from_node == to_node:
                                        continue

                                    # 数据流: def_var flow to use_var
                                    key = "{}-{}".format(from_node, to_node)
                                    if key not in duplicate:

                                        duplicate[key] = 1  # 去重

                                        if to_node not in data_flow_map:
                                            data_flow_map[to_node] = [from_node]
                                        else:
                                            data_flow_map[to_node].append(from_node)

                                        data_flow_edges.append(
                                            (from_node, to_node, {'color': "blue", "type": "data_flow"}))

                                # kill：中断上一次def的flow 出读操作栈
                                del use_info[var]

                            def_info[var] = from_node

                    # 读变量：压栈，等待被def时分析
                    if var_info["op_type"] == "use":
                        for var in var_info["list"]:

                            if var not in use_info:
                                use_info[var] = [from_node]
                            else:
                                use_info[var].append(from_node)

        # 保存
        self.data_flow_map = data_flow_map
        self.data_flow_edges = data_flow_edges
        self.function_info.add_semantic_edges("data_flow", self.data_flow_edges)
        # self.function_info.semantic_edges["data_flow"] = self.data_flow_edges

    #######################################################
    # 函数内 数据依赖分析                                   #
    #######################################################
    def data_dependency_analyze(self):
        """
        数据依赖解析：获得 def-use chain
        """
        # 数据使用信息、简化cfg
        stmts_var_info_maps = self.function_info.stmts_var_info_maps
        cfg = self.function_info.simple_cfg

        var_def_use_chain = {}
        data_dependency_relations = []

        cfg_paths = nx.all_simple_paths(cfg, source="0", target="EXIT_POINT")
        for path in cfg_paths:
            var_def_use_chain.clear()

            for stmt in path[:-1]:  # 去尾，避免EXIT_POINT
                stmt_id = str(stmt)
                stmt_var_infos = stmts_var_info_maps[stmt_id]

                for var_info in stmt_var_infos:
                    for var_name in var_info["list"]:
                        info = {"id": stmt_id, "var_type": var_info["type"], "op_type": var_info["op_type"]}
                        if var_name not in var_def_use_chain:
                            chain = [info]
                            var_def_use_chain[var_name] = chain
                        else:
                            var_def_use_chain[var_name].append(info)

            # 计算当前执行路径下的def_use_chain分析
            for var in var_def_use_chain:
                last_def = None
                chain = var_def_use_chain[var]
                for chain_node in chain:

                    if chain_node["op_type"] == "def":
                        last_def = chain_node["id"]
                    else:
                        if last_def is not None:
                            edge_info = {"from": chain_node["id"], "to": last_def}
                            data_dependency_relations.append(edge_info)

        ddg_edges = _get_ddg_edges(data_dependency_relations, "data_dependency")

        # 保存
        self.data_dependency_edges = ddg_edges
        self.function_info.add_semantic_edges("data_dep", ddg_edges)
        # self.function_info.semantic_edges["data_dep"] += self.data_flow_edges

    #######################################################
    # 函数内 根据给定执行路径进行数据依赖分析                    #
    #######################################################
    def get_data_dependency_relations_by_path(self, path):

        stmts_var_info_maps = self.function_info.stmts_var_info_maps

        var_def_use_chain = {}
        data_dependency_relations = []

        for stmt in path:

            stmt_id = str(stmt)
            if stmt_id == 'EXIT_POINT':  # 避免 EXIT_POINT  不能统一去尾
                continue

            stmt_var_infos = stmts_var_info_maps[stmt_id]
            for var_info in stmt_var_infos:
                for var_name in var_info["list"]:
                    info = {"id": stmt_id, "var_type": var_info["type"], "op_type": var_info["op_type"]}
                    if var_name not in var_def_use_chain:
                        chain = [info]
                        var_def_use_chain[var_name] = chain
                    else:
                        var_def_use_chain[var_name].append(info)

        # 计算当前执行路径下的def_use chain分析
        for var in var_def_use_chain:
            last_def = None

            chain = var_def_use_chain[var]
            for chain_node in chain:
                if chain_node["op_type"] == "def":
                    last_def = chain_node["id"]
                else:
                    if last_def is not None:
                        edge_info = {"from": chain_node["id"], "to": last_def}
                        data_dependency_relations.append(edge_info)

        return data_dependency_relations

    #######################################################
    # 函数内 交易语句涉及全局变量分析                           #
    #######################################################
    def transaction_state_vars_analyze(self):

        """
        分析交易语句涉及的全局变量：
        在数据流的基础上提取交易语句涉及的全局变量的数据流, 并在数据流中寻找涉及的所有全局变量

        Note: 只需要分析那些交易行为使用的全局变量，过程中定义（def or write）的全局变量不用分析
        只分析如下情况的数据流：
            (注) var_info["type"] == "state":
            var_info["op_type"] == "use" and var_info["type"] == "state":
        """

        trans_stats = {}

        stmts_var_info_maps = self.function_info.stmts_var_info_maps
        data_flow_map = self.data_flow_map
        transaction_stmts = self.function_info.transaction_stmts

        # 遍历函数中所有交易语句
        for trans_stmt in transaction_stmts:

            trans_state_infos = []

            # DFS搜索数据流图，寻找所有与交易语句相关数据信息
            stack = [trans_stmt]
            while len(stack) != 0:

                to_id = stack.pop()
                if to_id in data_flow_map:
                    for from_id in data_flow_map[to_id]:

                        stmt_var_infos = stmts_var_info_maps[from_id]
                        for var_info in stmt_var_infos:

                            # NOTE: 只需要分析那些交易行为使用的全局变量，过程中定义的不用管
                            if var_info["op_type"] == "use" and var_info["type"] == "state":
                                trans_state_infos.append({"vars": var_info["list"], "stmt_id": from_id})

                        stack.append(from_id)

            trans_stats[trans_stmt] = trans_state_infos
            # print("DEBUG 切片准则：{}".format(trans_stmt))
            # print("DEBUG 涉及全局变量信息:{}".format(trans_stats[trans_stmt]))

        self.transaction_states = trans_stats
        self.function_info.transaction_states = self.transaction_states

    ################################################################
    # 函数内 交易相关全局变量反向数据依赖关系分析                           #
    # https://github.com/smartcontract-detect-yzu/slither/issues/6  #
    #################################################################
    def reverse_data_dependency_for_transaction_state(self):

        """
        全局变量的修改存在本次修改会影响下次执行的情况：
        storage A;
        function {1:a = A, 2:do send, 3:A = 1}
        数据存在 3->1 的逆向影响
        """

        trans_states = self.trans_state_append_criteria
        cfg = self.function_info.simple_cfg
        reverse_relations = []

        for trans_stmt in trans_states:
            state_related_stmts = trans_states[trans_stmt]

            # print(state_related_stmts)
            for state_related_stmt in state_related_stmts:

                # 寻找执行路径，并进行反向分析
                trans_state_paths = nx.all_simple_paths(cfg, source="0", target=str(state_related_stmt))
                for trans_state_path in trans_state_paths:
                    reverse_relations += self.get_data_dependency_relations_by_path(reversed(trans_state_path))

        reverse_ddg_edges = _get_ddg_edges(reverse_relations, "re_data_dependency")

        # 保存
        self.data_dependency_edges += reverse_ddg_edges
        self.function_info.add_semantic_edges("data_dep", reverse_ddg_edges)
        # self.function_info.semantic_edges["data_dep"] += reverse_ddg_edges

    ###########################################################
    # 函数内 交易涉及全局变量修改语句加入切片准则 (state_criteria_add)#
    ###########################################################
    def transaction_state_criteria_add(self):
        """
        寻找函数内修改了交易相关全局变量，作为切片准则
        """

        dup = {}
        criteria_append = {}

        state_def_stmts = self.function_info.state_def_stmts

        for transaction_stmt in self.transaction_states:
            dup.clear()
            trans_states = self.transaction_states[transaction_stmt]
            criteria_append[transaction_stmt] = []
            for state_info in trans_states:

                states = state_info["vars"]
                for state in states:

                    # 当前交易相关全局变量被修改，加入切片准则
                    if state in state_def_stmts and state not in dup:
                        dup[state] = 1
                        criteria_append[transaction_stmt].append(state_def_stmts[state])

        # print("criteria_append :", criteria_append)
        self.trans_state_append_criteria = criteria_append
        self.function_info.append_criterias = criteria_append
        return criteria_append

    #######################################################
    # 函数内 循环体内数据流增强                                #
    #######################################################
    def loop_data_dependency_relation_enhance(self):

        duplicate = {}
        ddg_edges = []

        transaction_stmts = self.function_info.transaction_stmts
        for criteria in transaction_stmts:

            loop_reverse_paths = self.function_info.loop_body_extreact(criteria)
            for path in loop_reverse_paths:

                loop_data_deps = self.get_data_dependency_relations_by_path(path)
                for edge_info in loop_data_deps:

                    if edge_info["from"] == edge_info["to"]:
                        continue

                    key = "{}-{}".format(edge_info["from"], edge_info["to"])
                    if key not in duplicate:
                        duplicate[key] = 1
                        ddg_edges.append(
                            (edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"})
                        )

        self.loop_data_dependency_edges = ddg_edges
        self.function_info.add_semantic_edges("data_dep", ddg_edges)
        # self.function_info.semantic_edges["data_dep"] += self.loop_data_dependency_edges

    #######################################################
    # 函数内 循环体内数据流增强                                #
    #######################################################
    def do_data_semantic_analyze(self):
        """
        数据流相关语义分析
        """
        self.data_flow_analyze()  # 数据流分析
        self.data_dependency_analyze()  # 数据依赖分析

        # simple mode no need to analyze it
        if not self.simple_mode:
            self.transaction_state_vars_analyze()  # 交易相关全局变量数据使用分析
            self.transaction_state_criteria_add()  # 交易相关全局变量数据依赖分析，生成新的切片准则
            self.reverse_data_dependency_for_transaction_state()  # 交易相关全局变量反向数据依赖分析

        self.loop_data_dependency_relation_enhance()  # 循环体内部数据依赖

        if self.data_flow_edges is None:
            raise RuntimeError("数据流未分析")

        if self.data_dependency_edges is None:
            raise RuntimeError("数据依赖未分析")

        if self.loop_data_dependency_edges is None:
            raise RuntimeError("循环体数据依赖为分析")

        return self.data_flow_edges, self.data_dependency_edges, self.loop_data_dependency_edges
