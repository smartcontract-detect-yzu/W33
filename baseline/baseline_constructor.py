import json
import os

from sklearn import model_selection
import jsonlines
from random import sample

from tqdm import tqdm
from baseline.RE_TMP import ReGraphExtractor, ReGraph2Vec
from sol_analyzer.info_analyze.file_psc_analyze import FilePscAnalyzer
from baseline.CBGRU.CBGRU_dataset import CBGRU_constructor

VUL_MAP = {
    "re": "nonReentrant",
}
SafeLowLevelCallMap = [
    'safeTransferFrom', 'safeTransfer', 
    "sendValue",
    "functionCallWithValue", "functionCall", "functionStaticCall",
    "transferFrom", "transfer",
]

def _do_normal_re(line):
    
    if "modifier" not in line and "nonReentrant" in line:
        line = line.replace("nonReentrant", "")
    
    return line

def _do_normal_llc(line):

    for llc in SafeLowLevelCallMap:
        if llc in line:
            line = line.replace(llc, "call.value")
            break
    
    return line

def __save_feature_to_text(feature_to_save, file_name):
    """
    将对应的node/edge feature存放到text文件中
    :param feature_to_save:
    :param file_name:
    :return:
    """

    with open(file_name, "w+") as f:

        for nodes_line in feature_to_save:  # 针对每一行单独解析
            line_src = ""  # 当前行内容
            for idx, node in enumerate(nodes_line):

                # 如果成员是列表，需要判断链表长度
                if isinstance(node, list):
                    if len(node) != 1:
                        raise RuntimeError("WTF:链表长度超过1")
                    content = str(node[0])
                else:
                    content = str(node)

                line_src += content
                if (idx + 1) == len(nodes_line):
                    line_src += "\n"
                else:
                    line_src += " "
            f.write(line_src)

def _save_feature(node_features, edge_features, prefix):
    """
        保存当前智能合约的节点和边特征
    """
    node_feature_file = prefix + "_node_feature.txt"
    __save_feature_to_text(node_features, node_feature_file)

    edge_feature_file = prefix + "_edge_feature.txt"
    __save_feature_to_text(edge_features, edge_feature_file)

class Baseline_Constructor:

    def __init__(self, dataset_dir, vul_type) -> None:

        self.dataset_dir = dataset_dir
        self.vul_type = vul_type
        self.type = VUL_MAP[vul_type]

        self.vul_samples = {}
        self.no_vul_samples = {}
        self.normal_vul_samples = {}
        self.normal_no_vul_samples = {}

        # TMP
        self.no_vul_vects = []
        self.vul_vects = []

        # Peculiar
        self.dataset_jsonl = []
        self.idx = 0
        self.id_labels = []

        self.line_normlizers = self.get_line_normalizer()
       

    def get_line_normalizer(self):

        if self.type == "nonReentrant":
            return [_do_normal_re, _do_normal_llc]
        
        return None

    def get_target_samples(self):
        """
            寻找目标样本
        """
        targets_list = os.listdir(self.dataset_dir)
        for target in targets_list:
            _target_path = f"{self.dataset_dir}//{target}//"
            with open(_target_path + "total_sbp.json") as f:

                sbp_infos = json.load(f)
                for sbp_key in sbp_infos:

                    # 样本基本信息获取
                    sample_sbp_info = sbp_infos[sbp_key]
                    vul_sample = 0
                    file_name = sample_sbp_info["sol_file_path"]
                    target_file_path = _target_path + file_name
                    
                    # 创建目录以存放normalized之后的样本
                    if not os.path.exists(_target_path + "{}".format(self.vul_type)):
                        os.mkdir(_target_path + "{}".format(self.vul_type))

                    # 寻找目标样本
                    for _f_sbp in sample_sbp_info["function_sbp_infos"]:
                        _f_lable = _f_sbp["lable_infos"]["label"]
                        if _f_lable == self.type:
                            vul_sample = 1
                            break
                    
                    # 记录目标样本
                    if vul_sample == 1:
                        # print(target_file_path)
                        self.vul_samples[target_file_path] = 1
                    else:
                        self.no_vul_samples[target_file_path] = 1


    def _construct_normalize_sample(self, file_path, new_file_path):
        lines = open(file_path, encoding="utf-8").readlines()
        
        with open(new_file_path, "w+", encoding="utf-8") as f:
            for line in lines:
                for do_normal in self.line_normlizers:
                    line = do_normal(line)
                f.write(line)
    
    def construct_normalized_dataset(self):

        for vul_sample in self.vul_samples:

            sol_name = str(vul_sample).split("//")[-1]
            normal_sol_name = "{}//{}".format(self.vul_type, sol_name)
            normalized_sample = vul_sample.replace(sol_name, normal_sol_name)
            
            if not os.path.exists(normalized_sample):
                self._construct_normalize_sample(vul_sample, normalized_sample)

            self.normal_vul_samples[normalized_sample] = 1
        
        for no_vul_sample in self.no_vul_samples:

            sol_name = str(no_vul_sample).split("//")[-1]
            normal_sol_name = "{}//{}".format(self.vul_type, sol_name)
            normalized_sample = no_vul_sample.replace(sol_name, normal_sol_name)

            if not os.path.exists(normalized_sample):
                self._construct_normalize_sample(no_vul_sample, normalized_sample)
            
            self.normal_no_vul_samples[normalized_sample] = 1
    
    
    """========================================================================
    =                                TMP/DR-GCN                               =
    ========================================================================"""
    def TMP_get_graph_for_Re(self, target, label):

        prefix = str(target).strip(".sol")
        node_vecs = []
        
        try:
            node_features, edge_features = ReGraphExtractor.create_graph_info_for_contract(target)
            _save_feature(node_features, edge_features, prefix)
        except: # 产生异常保存，并不退出
            print("ERROR for graph:{}".format(target))

        try:
            node_vec_array, graph_edge_array = ReGraph2Vec.get_vec_from_features(prefix)

            # 节点特征标准化
            for vec in node_vec_array:
                node_vecs.append(vec[1])
            
            # 保存结果
            if len(graph_edge_array) != 0:
                if label == "1":
                    self.vul_vects.append({
                            "targets": label,
                            "graph": graph_edge_array,
                            "contract_name": target,
                            "node_features": node_vecs
                    })
                else:
                    self.no_vul_vects.append({
                            "targets": label,
                            "graph": graph_edge_array,
                            "contract_name": target,
                            "node_features": node_vecs
                    })
        except: # 产生异常保存，并不退出
            print("ERROR for vec:{}".format(target))

    def TMP_create_feature_for_smaples(self):

        for normalized_vul_sample in self.normal_vul_samples:
            self.TMP_get_graph_for_Re(normalized_vul_sample, "1")

        for normalized_novul_sample in self.normal_no_vul_samples:
            self.TMP_get_graph_for_Re(normalized_novul_sample, "0")

        print("[INFO] 一共收集到 {} 样本 vul:{} no_vul:{}".format(self.vul_type,  len(self.vul_vects), len(self.no_vul_vects)))

        # 保存数据集
        with open("{}_vul_train.json".format(self.vul_type), "w+") as f:
            f.write(json.dumps(self.vul_vects))
        with open("{}_no_vul_train.json".format(self.vul_type), "w+") as f:
            f.write(json.dumps(self.no_vul_vects))

    def TMP_create_train_valid_dataset(self):

        vul_mask = {}
        no_vul_mask = {}

        with open("{}_vul_train.json".format(self.vul_type), "r") as jsonFile:
            vul_json = json.load(jsonFile)

        with open("{}_no_vul_train.json".format(self.vul_type), "r") as jsonFile:
            no_vul_json = json.load(jsonFile)

        vul_cnt = len(vul_json)
        no_vul_cnt = len(no_vul_json)
        total_cnt = vul_cnt + no_vul_cnt
        print("数据集大小 VUL:{} NO_VUL:{}  TOTAL:{}".format(vul_cnt, no_vul_cnt, total_cnt))

        # 构建数据集：3/7开
        vul_id_list = list(range(vul_cnt))
        no_vul_id_list = list(range(no_vul_cnt))

        # 随机选择漏洞样本/无漏洞样本
        vul_train_cnt = int(vul_cnt * 0.7)  # 漏洞训练集数量
        vul_sample_ids = sample(vul_id_list, vul_train_cnt)  # 漏洞样本的ID

        no_vul_train_cnt = int(no_vul_cnt * 0.7)  # 无漏洞训练集数量
        no_vul_sample_ids = sample(no_vul_id_list, no_vul_train_cnt)

        # 挑选训练集
        train_samples = []
        for vul_id in vul_sample_ids:
            vul_mask[vul_id] = 1
            train_samples.append(vul_json[vul_id])

        for no_vul_id in no_vul_sample_ids:
            no_vul_mask[no_vul_id] = 1
            train_samples.append(no_vul_json[no_vul_id])

        # 保存训练集
        with open("{}_final_train.json".format(self.vul_type), "w+") as f:
            f.write(json.dumps(train_samples))

        # 构建测试集
        test_samples = []
        for vul_id in vul_id_list:
            if vul_id not in vul_mask:
                test_samples.append(vul_json[vul_id])

        for no_vul_id in no_vul_id_list:
            if no_vul_id not in no_vul_mask:
                test_samples.append(no_vul_json[no_vul_id])

        # 保存测试集
        with open("{}_final_test.json".format(self.vul_type), "w+") as f:
            f.write(json.dumps(test_samples))

        print("最终数据集格式：")
        print("训练集：{}  测试集：{}".format(len(train_samples), len(test_samples)))

    """========================================================================
    =                  DR-GCN:ERROR 该功能暂不可用                             =
    ========================================================================"""
    def DRGCN_create_dataset(self):
        
        indicate = 0
        node_attributes = []    # SMARTCONTRACT_full_node_attributes
        node_labels = []        # SMARTCONTRACT_full_node_labels
        graph_labels = []       # SMARTCONTRACT_full_graph_labels
        graph_indicators = []   # SMARTCONTRACT_full_graph_indicator

        if not os.path.exists("{}_final_test.json".format(self.vul_type)) or not os.path.exists("{}_final_train.json".format(self.vul_type)):
            print("ERROR: 请首先调用TMP接口创建特征信息")
        
        test_samples = json.load(open("{}_final_test.json".format(self.vul_type), "r"))
        train_samples = json.load(open("{}_final_train.json".format(self.vul_type), "r"))
        
        for samples_list in [test_samples, train_samples]:
            for sample_info in samples_list:
                indicate += 1

                graph_label = sample_info["targets"]
                graph_labels.append(graph_label)

                for node_feature in sample_info["node_features"]:
                    node_labels.append(1)
                    graph_indicators.append(indicate)
                    node_attributes.append(node_feature)



    """========================================================================
    =                                Peculiar                                 =
    ========================================================================"""
    def Peculiar_get_sol(self, file_name):
            
        file_analyzer = FilePscAnalyzer(file_name, 1)

        file_analyzer.do_delete_comment()      # 删除其中的注释
        file_analyzer.do_change_to_sequence()  # 将函数转换为序列
        content = file_analyzer.get_content()  # 得到序列化的内容

        return content
    
    def Peculiar_create_idx_files(self):

        x = list(range(0, len(self.id_labels)))
        y = self.id_labels

        # total * 0.7 train / 0.3 valid+test
        X_train, X_rest, y_train, y_rest = model_selection.train_test_split(x, y, test_size=0.3, random_state=1234)
        with open("train.txt", "w+") as f:
            for x, y in zip(X_train, y_train):
                line_info = "{}\t{}\n".format(x, y)
                f.write(line_info)

        # 0.3 * total * 0.66 = test
        X_test, X_valid, y_test, y_valid = model_selection.train_test_split(X_rest, y_rest, test_size=0.33,
                                                                            random_state=5678)
        with open("test.txt", "w+") as f:
            for x, y in zip(X_test, y_test):
                line_info = "{}\t{}\n".format(x, y)
                f.write(line_info)

        # 0.3 * total * 0.33 = valid
        with open("valid.txt", "w+") as f:
            for x, y in zip(X_valid, y_valid):
                line_info = "{}\t{}\n".format(x, y)
                f.write(line_info)
    
    def Peculiar_create_feature_for_dataset(self):

        for normalized_vul_sample in self.normal_vul_samples:
            content = self.Peculiar_get_sol(normalized_vul_sample)

            self.dataset_jsonl.append({"contract": content, "idx": str(self.idx), "address": normalized_vul_sample})
            self.id_labels.append(1)
            self.idx += 1

        for normalized_novul_sample in self.normal_no_vul_samples:
            content= self.Peculiar_get_sol(normalized_novul_sample)

            self.dataset_jsonl.append({"contract": content, "idx": str(self.idx), "address": normalized_novul_sample})
            self.id_labels.append(0)
            self.idx += 1

    def Peculiar_create_train_valid_dataset(self):

        # 保存jsonl文件
        with jsonlines.open('data.jsonl', mode='w') as writer:
            for item_json in self.dataset_jsonl:
                writer.write(item_json)

        # 构建train\valid\test数据集 7：2：1
        self.Peculiar_create_idx_files()
    
    """========================================================================
    =                                CBGRU                                    =
    ========================================================================"""
    def CBGRU_create_feature_for_sample(self):

        cbgru = CBGRU_constructor()
        dataset_result = []
        
        total_cnt = len(self.vul_samples) + len(self.no_vul_samples)
        with tqdm(total=total_cnt) as pbar:

            for sample in self.vul_samples:
                file_analyzer = FilePscAnalyzer(sample, 1)
                lines = file_analyzer.do_delete_comment()
                dataset_result += cbgru.clean_fragment(lines.split('\n'), 1)
                pbar.update(1)

            for sample in self.no_vul_samples:
                file_analyzer = FilePscAnalyzer(sample, 1)
                lines = file_analyzer.do_delete_comment()
                dataset_result += cbgru.clean_fragment(lines.split('\n'), 0)
                pbar.update(1)

        dataset_file = f"CBGRU_dataset_{self.vul_type}.txt"
        with open(dataset_file, "w+") as f:
            f.writelines(dataset_result)


        

    