from random import sample

from tqdm import tqdm
import json
import os
import re
import time

from RE_TMP import ReGraph2Vec, ReGraphExtractor
from TS_TMP import TSgraph2vec, TSAutoExtractGraph
import jsonlines
from get_plain_sol_seq import FilePscAnalyzer
from sklearn import model_selection
from Peculiar import peculiar_dataset
DATASET_PREFIX = "dataset\\processed_data\\"
NO_VUL_DATASET_PREFIX = "dataset\\processed_data\\no_vul\\"
TEST_SOL = "dataset\\test\\0x8e64649c990c74aa55df1d651cacb5beee887a46_suicideContract.sol"


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
    :param node_features:
    :param edge_features:
    :return:
    """

    if not os.path.exists("IJICA_GRAPH"):
        os.mkdir("IJICA_GRAPH")

    feature_prefix = "IJICA_GRAPH\\"

    node_feature_file = feature_prefix + "{}_node_feature.txt".format(prefix)
    __save_feature_to_text(node_features, node_feature_file)

    edge_feature_file = feature_prefix + "{}_edge_feature.txt".format(prefix)
    __save_feature_to_text(edge_features, edge_feature_file)

    return "IJICA_GRAPH\\"


def _get_graph_for_TMP_Re(target, train_array, vul_lable, error_cnt, right_cnt):

    try:
        node_features, edge_features = ReGraphExtractor.create_graph_info_for_contract(target)
        save_path = _save_feature(node_features, edge_features, "re")

    except:
        # TODO: 产生异常保存，并不退出
        error_cnt += 1
        print("error:{}".format(target))

    try:
        node_vec_array, graph_edge_array = ReGraph2Vec.get_vec_from_features(save_path)
        node_vecs = []
        for vec in node_vec_array:
            node_vecs.append(vec[1])

        right_cnt += 1
        if len(graph_edge_array) != 0:
            train_array.append({
                "targets": vul_lable,
                "graph": graph_edge_array,
                "contract_name": target,
                "node_features": node_vecs
            })
    except:
        error_cnt += 1
        print("ERROR FOR VEC:{}".format(target))


def _get_graph_for_TMP_TS(target, train_array, vul_lable, error_cnt, right_cnt):

    try:
        node_features, edge_features = TSAutoExtractGraph.create_graph_info_for_contract_TS(target)
        save_path = _save_feature(node_features, edge_features, "ts")

    except:
        # TODO: 产生异常保存，并不退出
        error_cnt += 1
        print("error:{}".format(target))

    try:
        node_vec_array, graph_edge_array, var_embedding = TSgraph2vec.get_vec_from_features_TS(save_path)
        node_vecs = []
        for vec in node_vec_array:
            node_vecs.append(vec[1])

        for vec in var_embedding:
            node_vecs.append(vec[1])

        right_cnt += 1
        if len(graph_edge_array) != 0:
            train_array.append({
                "targets": vul_lable,
                "graph": graph_edge_array,
                "contract_name": target,
                "node_features": node_vecs
            })
    except:
        error_cnt += 1
        print("ERROR FOR VEC:{}".format(target))


def extract_graph_for_vul_dataset(dataset_root_dir, vul_lable, train_array, graph_type):
    """
    分析指定数据集：根据数据集json描述文件，遍历所有样本并生成对应的node/edge feature文件
    :param dataset_root_dir:
    :return:
    """

    # 统计信息
    error_cnt = right_cnt = 0

    # 当前目录
    pwd = os.getcwd()

    dataset_json = DATASET_PREFIX + dataset_root_dir + ".json"

    # 逐条分析数据集内部的合约
    with open(dataset_json, 'r') as jsonFile:
        data_infos = json.load(jsonFile)

        total = len(data_infos)
        with tqdm(total=total) as pbar:
            for data_case_info in data_infos:
                target_path = DATASET_PREFIX + data_case_info['path']
                target = data_case_info['name']

                # 进度条信息
                desc_info = 'Target: ' + target
                pbar.set_description(desc_info)
                pbar.update(1)

                #  进入目标合约工作目录
                os.chdir(target_path)

                # 只有info.json的才是目标合约
                if not os.path.exists('info.json'):
                    print('\033[1;35m 缺少INFO文件:%s \033[0m!' % data_case_info['name'])
                else:
                    if graph_type == "re":
                        _get_graph_for_TMP_Re(target, train_array, vul_lable, error_cnt, right_cnt)
                    elif graph_type == "ts":
                        _get_graph_for_TMP_TS(target, train_array, vul_lable, error_cnt, right_cnt)
                    else:
                        raise RuntimeError("错误的入参")
                # 还原工作目录
                os.chdir(pwd)

    print("图表示抽取完毕：成功：{}  失败：{}  总数：{}".format(right_cnt, error_cnt, (right_cnt + error_cnt)))


def extract_graph_for_no_vul_dataset(vul_lable, train_array, graph_type):
    """
       分析指定数据集：根据数据集json描述文件，遍历所有样本并生成对应的node/edge feature文件
       :param dataset_root_dir:
       :return:
       """

    # 统计信息
    error_cnt = right_cnt = 0

    # 当前目录
    pwd = os.getcwd()

    # 无漏洞数据集
    dataset_list = NO_VUL_DATASET_PREFIX + "selected_dataset.txt"

    # 逐条分析数据集内部的合约
    with open(dataset_list, 'r') as file_list:

        sample_list = file_list.readlines()
        total = len(sample_list)
        with tqdm(total=total) as pbar:
            for sample in sample_list:
                sample_address = str(sample).split("\\")[1]
                target = sample_address + ".sol"

                # 进度条信息
                desc_info = 'Target: ' + target
                pbar.set_description(desc_info)
                pbar.update(1)

                #  进入目标合约工作目录
                target_path = NO_VUL_DATASET_PREFIX + "NoVulSrcForAstSlither\\" + sample_address + "\\"
                os.chdir(target_path)

                if not os.path.exists('info.json'):
                    print('\033[1;35m 缺少INFO文件:%s \033[0m!' % target)
                else:
                    if graph_type == "re":
                        _get_graph_for_TMP_Re(target, train_array, vul_lable, error_cnt, right_cnt)
                    elif graph_type == "ts":
                        _get_graph_for_TMP_TS(target, train_array, vul_lable, error_cnt, right_cnt)
                    else:
                        raise RuntimeError("错误的入参")

                os.chdir(pwd)

    print("图表示抽取完毕：成功：{}  失败：{}  总数：{}".format(right_cnt, error_cnt, (right_cnt + error_cnt)))


def save_vul_dataset(graph_type):

    # if os.path.exists("vul_train.json"):
    #     return

    # 数据集
    train_array = []

    # 漏洞数据集：构建json格式数据集 DataSetFinal SolidiFIAST
    extract_graph_for_vul_dataset("DataSetFinal", "1", train_array, graph_type)
    extract_graph_for_vul_dataset("SolidiFIAST", "1", train_array, graph_type)

    # 保存数据集
    with open("{}_vul_train.json".format(graph_type), "w+") as f:
        f.write(json.dumps(train_array))


def save_no_vul_dataset(graph_type):

    # if os.path.exists("no_vul_train.json"):
    #     return

    # 数据集
    train_array = []

    # 无漏洞数据集：构建json格式数据集 NoVulSrcForAstSlither
    extract_graph_for_no_vul_dataset("0", train_array, graph_type)

    # 保存数据集
    with open("{}_no_vul_train.json".format(graph_type), "w+") as f:
        f.write(json.dumps(train_array))


def create_train_valid_dataset(graph_type):

    vul_mask = {}
    no_vul_mask = {}

    with open("{}_vul_train.json".format(graph_type), "r") as jsonFile:
        vul_json = json.load(jsonFile)

    with open("{}_no_vul_train.json".format(graph_type), "r") as jsonFile:
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
    with open("{}_final_train.json".format(graph_type), "w+") as f:
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
    with open("{}_final_test.json".format(graph_type), "w+") as f:
        f.write(json.dumps(test_samples))

    print("最终数据集格式：")
    print("训练集：{}  测试集：{}".format(len(train_samples), len(test_samples)))

    # file_analyzer.do_print_sequence()
    # file_analyzer.save_sequence_content_to_file()


def _get_peculiar_sol_content(file_name):
    # 删除其中的注释
    file_analyzer = FilePscAnalyzer(file_name, 1)
    file_analyzer.do_delete_comment()
    file_analyzer.do_change_to_sequence()

    content = file_analyzer.get_content()

    return content


def extract_peculiar_vul_dataset(dataset_root_dir, idx, ids_labels, jsonl_info):
    """
    分析指定数据集：根据数据集json描述文件，遍历所有样本并生成对应的node/edge feature文件
    :param dataset_root_dir:
    :return:
    """

    # 当前目录
    pwd = os.getcwd()
    dataset_json = DATASET_PREFIX + dataset_root_dir + ".json"

    # 逐条分析数据集内部的合约
    with open(dataset_json, 'r') as jsonFile:
        data_infos = json.load(jsonFile)

        total = len(data_infos)
        with tqdm(total=total) as pbar:
            for data_case_info in data_infos:
                target_path = DATASET_PREFIX + data_case_info['path']
                target = data_case_info['name']

                # 进度条信息
                desc_info = 'Target: ' + target
                pbar.set_description(desc_info)
                pbar.update(1)

                #  进入目标合约工作目录
                os.chdir(target_path)

                # 只有info.json的才是目标合约
                if not os.path.exists('info.json'):
                    print('\033[1;35m 缺少INFO文件:%s \033[0m!' % data_case_info['name'])
                else:
                    content = _get_peculiar_sol_content(target)
                    jsonl_info.append({
                        "contract": content,
                        "idx": str(idx),
                        "address": target
                    })
                    ids_labels.append(1)
                    idx += 1

                # 还原工作目录
                os.chdir(pwd)


def extract_peculiar_novul_dataset(idx, ids_labels, jsonl_info):
    """
       分析指定数据集：根据数据集json描述文件，遍历所有样本并生成对应的node/edge feature文件
       :param dataset_root_dir:
       :return:
   """

    # 当前目录
    pwd = os.getcwd()

    # 无漏洞数据集
    dataset_list = NO_VUL_DATASET_PREFIX + "selected_dataset.txt"

    # 逐条分析数据集内部的合约
    with open(dataset_list, 'r') as file_list:

        sample_list = file_list.readlines()
        total = len(sample_list)
        with tqdm(total=total) as pbar:
            for sample in sample_list:
                sample_address = str(sample).split("\\")[1]
                target = sample_address + ".sol"

                # 进度条信息
                desc_info = 'Target: ' + target
                pbar.set_description(desc_info)
                pbar.update(1)

                #  进入目标合约工作目录
                target_path = NO_VUL_DATASET_PREFIX + "NoVulSrcForAstSlither\\" + sample_address + "\\"
                os.chdir(target_path)

                if not os.path.exists('info.json'):
                    print('\033[1;35m 缺少INFO文件:%s \033[0m!' % target)
                else:
                    content = _get_peculiar_sol_content(target)
                    jsonl_info.append({
                        "contract": content,
                        "idx": str(idx),
                        "address": target
                    })
                    ids_labels.append(0)
                    idx += 1

                os.chdir(pwd)


def create_idx_files(id_labels):
    X = list(range(0, len(id_labels)))
    y = id_labels

    # total * 0.7 train / 0.3 valid+test
    X_train, X_rest, y_train, y_rest = model_selection.train_test_split(X, y, test_size=0.3, random_state=1234)
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


def create_peculiar_dataset():
    # idx
    idx = 0

    # 数据集
    dataset_jsonl = []
    id_labels = []

    # 漏洞数据集：构建json格式数据集 DataSetFinal SolidiFIAST
    extract_peculiar_vul_dataset("DataSetFinal", idx, id_labels, dataset_jsonl)
    extract_peculiar_vul_dataset("SolidiFIAST", idx, id_labels, dataset_jsonl)
    extract_peculiar_novul_dataset(idx, id_labels, dataset_jsonl)

    # 保存jsonl文件
    with jsonlines.open('data.jsonl', mode='w') as writer:
        for item_json in dataset_jsonl:
            writer.write(item_json)

    # 构建train\valid\test数据集 7：2：1
    create_idx_files(id_labels)


def create_tmp_dataset(graph_type):

    # TMP数据集构建
    save_vul_dataset(graph_type)
    save_no_vul_dataset(graph_type)
    create_train_valid_dataset(graph_type)


if __name__ == "__main__":

    # create_tmp_dataset("re")  # TMP方法的数据集: re/ts

    # create_peculiar_dataset() # peculiar方法的数据集

    peculiar_dataset.peculiar_calcu_metrics("test.txt", "predictions.txt")
