import argparse
import json
import os
import shutil
import subprocess
from ast_analyzer import FunctionAstAnalyzer
from target_info_collector import TrgetInfoCollector
from tqdm import tqdm
import click
from slither.slither import Slither

def _do_main(ast_analyzer:FunctionAstAnalyzer):
    """
        样本分析主循环
    """
    # 构建 function-level AST
    ast_analyzer.construct_ast_for_function_sample()
    
    # 不支持asm
    if ast_analyzer.check_function_asm_info():
        return
    
    # 构建 function-level CFG
    ast_analyzer.ast_slither_id_align()   # slither和AST的ID对齐
    ast_analyzer.construct_cfg_for_function_sample() 
    
    # sbp normalize in ast
    ast_analyzer.normalize_sbp_in_ast()

    # debug:保存AST图片
    ast_analyzer.save_ast_as_png(postfix="")
    ast_analyzer.save_ast_as_png(postfix="normalized")

    # AST 与 CFG 图对齐
    ast_analyzer.cfg_supplement_stmts_for_ast()
    ast_analyzer.get_function_entry_info()   # 在分割之前记录下函数入口信息
    ast_analyzer.split_function_ast_stmts()
    ast_analyzer.check_split_function_ast_stmts()
    
    # 根据对齐后的normalize_ast对cfg进行normalize
    ast_analyzer.normalize_sbp_in_cfg()
    ast_analyzer.save_cfg_as_png(postfix="normalized")
    
    # 语句属性特征分析
    ast_analyzer.get_stmts_types()
    ast_analyzer.set_stmts_types_in_cfg()
    
    # debug: 保存CFG图片
    ast_analyzer.save_cfg_as_png(postfix="")
    ast_analyzer.save_cfg_as_png(postfix="typed_normalized")
    
    # 整合 modifier和 cfg
    ast_analyzer.concat_function_modifier_cfg()

    # 创建虚拟节点
    ast_analyzer.construct_virtual_nodes() 
    ast_analyzer.save_cfg_as_png(postfix="final")
    
    # 保存结果到json文件
    ast_analyzer.save_statements_json_infos()

    # debug: 根据保存的cfg生成一个png
    ast_analyzer.save_cfg_from_json()
    
    # 清除不需要的中间结果
    ast_analyzer.clean_up()

    return

def _construct_all_stmts_ast_infos_wrapper(target_filter, target_info_collector, is_modifier=False, save_png=1):
     
     """
        包装器:构建AST语句级别信息
     """
     for c_name in target_filter:
        for f_name in target_filter[c_name]:

           # 目标信息 
           target_info = target_filter[c_name][f_name]

           # 构建解析器
           ast_analyzer = FunctionAstAnalyzer(target_info, target_info_collector,log_level=LOG_LEVEL, is_modifier=is_modifier, save_png=save_png)
           
           # 分析
           _do_main(ast_analyzer)
        
           
class AnalyzeWrapper:
    """
        智能合约分析包装器
    """
    def __init__(self, dataset_dir, save_png) -> None:
        self.save_png = save_png
        self.dataset_dir = dataset_dir

        self.AST_JSON_DIR = "ast_json"
        self.SBP_JSON_DIR = "sbp_json"

        # 执行状态
        self.CONSTRUCT_DONE = "OK"
        self.CONSTRUCT_FAIL = "ERROR"
        self.SLITHER_OK = "slither_ok"
        self.SLITHER_ERROR = "slither_error"

        # flag 文件
        self.AST_DONE_FLAG = "ast_done.flag"
        self.DONE_FLAG = "construct_done.flag"
        self.FAIL_FLAG = "construct_fail.flag"
        self.SLITHER_FAIL_FLAG = "slither_error.flag"
    
    def do_vulnerability_static_after_stmt_analyze(self):

        no_vul_stmts = no_vul_functions = 0
        vul_stmts = vul_functions = 0

        vul_type_cnt_map = {
            "SafeMath": 0,
            "low-level call": 0,
            "safe cast": 0,
            "transaction order dependency": 0,
            "nonReentrant": 0,
            "onlyOwner": 0,
            "resumable_loop": 0
        }

        sample_files = os.listdir(self.dataset_dir)

        for sample in sample_files:
            path_sample = "{}//{}//".format(self.dataset_dir, sample)
            if os.path.exists(path_sample + self.DONE_FLAG):
                sample_dir_path = path_sample + "sample//"
                c_f_samples = os.listdir(sample_dir_path)

                for c_f in c_f_samples:
                    c_f_sample_dir_path = sample_dir_path + c_f + "//"
                    if os.path.exists(c_f_sample_dir_path + "statement_ast_infos.json"):
                        
                        with open(c_f_sample_dir_path + "statement_ast_infos.json", "r") as f:
                            stmts_infos = json.load(f)
                            vul_flag = 0
                            for stmt_id in stmts_infos:
                                
                                if stmt_id == "cfg_edges":
                                    continue

                                # 开始统计
                                stmt_info = stmts_infos[stmt_id]
                                if stmt_info["vul"] == 0:
                                    no_vul_stmts += 1
                                
                                else:
                                    vul_stmts += 1
                                    vul_flag = 1

                                    vul_type_infos = stmt_info["vul_type"]
                                    for ast_id in vul_type_infos:
                                        vul_type_cnt_map[vul_type_infos[ast_id]] += 1

                                if vul_flag == 0:
                                    vul_functions += 1
                                else:
                                    no_vul_functions += 1
      
        print("===============统计结果===============")
        print("no_vul_stmts:", no_vul_stmts)
        print("vul_stmts:", no_vul_stmts)
        print("no_vul_functions:", no_vul_stmts)
        print("vul_functions:", no_vul_stmts)
        print("============详细统计结果===============")
        print(json.dumps(vul_type_cnt_map, indent=4, separators=(",", ":")))
        print("====================END===============")


    def do_vulnerability_static_after_ast_analyze(self):

        vul_type_cnt_map = {
            "SafeMath": 0,
            "low-level call": 0,
            "safe cast": 0,
            "transaction order dependency": 0,
            "nonReentrant": 0,
            "onlyOwner": 0,
            "resumable_loop": 0
        }
        
        sample_files = os.listdir(self.dataset_dir)
        total_samples_cnt = len(sample_files)

        with tqdm(total=total_samples_cnt) as pbar:
            for sample in sample_files:

                path_sample = "{}//{}//".format(self.dataset_dir, sample)
                pbar.set_description('Processing:{}'.format(path_sample))

                if os.path.exists(path_sample + self.AST_DONE_FLAG):
                    sbp_json_dir = path_sample + "sbp_json//"
                    sbp_json_files = os.listdir(sbp_json_dir)

                    for sbp_json_file in sbp_json_files:
                        with open(sbp_json_dir + sbp_json_file, "r") as f:
                            sbp_json = json.load(f)
                            function_sbp_infos = sbp_json["function_sbp_infos"]
                            
                            for function_sbp_info in function_sbp_infos:
                                vul_type = function_sbp_info["lable_infos"]["label"]
                                vul_type_cnt_map[vul_type] += 1
                
                pbar.update(1)
        
        print("============SBP详细统计结果===============")
        print(json.dumps(vul_type_cnt_map, indent=4, separators=(",", ":")))
        print("====================END===============")


    def do_slither_check(self):
        """
            提前预处理Slither, 判断当前合约是否支持CFG的创建
        """
        succ_cnt = fail_cnt = 0
        pwd = os.getcwd()
        sample_files = os.listdir(self.dataset_dir)
        total_samples_cnt = len(sample_files)

        with tqdm(total=total_samples_cnt) as pbar:
            for sample in sample_files:

                path_sample = "{}//{}//".format(self.dataset_dir, sample)
                pbar.set_description('Processing:{}'.format(path_sample))
                
                if os.path.exists(path_sample + self.SLITHER_FAIL_FLAG):
                    fail_cnt += 1

                else:
                    f = open(path_sample + "download_done.txt")
                    compile_info = json.load(f)
                    subprocess.check_call(["solc-select", "use", compile_info["ver"]])
                    
                    os.chdir(path_sample)
                    try:  
                        Slither(compile_info["name"])
                    except:
                        fail_cnt += 1
                        self.save_the_result_flag("", self.SLITHER_ERROR)
                        os.chdir(pwd)
                        pbar.update(1)
                        continue

                    succ_cnt += 1
                    os.chdir(pwd)
                    pbar.update(1)
            
            print("========check result=========")
            print("succ_cnt:", succ_cnt)
            print("fail_cnt:", fail_cnt)
            print("========check result=========")
 


    def do_analyze_for_target(self, target_dir):

        target_info_collector = TrgetInfoCollector(target_dir=target_dir)
        if target_info_collector.slither_error:
            return self.SLITHER_ERROR
            
        # Note: 此处发生异常的判断 ==> 依赖异常抛出
        if not target_info_collector.slither_error:
            
            modifier_filter = target_info_collector.get_modifier_filter()
            _construct_all_stmts_ast_infos_wrapper(modifier_filter, target_info_collector, is_modifier=True, save_png=self.save_png)

            target_filter = target_info_collector.get_target_filter()
            _construct_all_stmts_ast_infos_wrapper(target_filter, target_info_collector, save_png=self.save_png)

    
    def do_analyze_for_dataset(self):

        sample_files = os.listdir(self.dataset_dir)
        total_samples_cnt = len(sample_files)
        error_pass = {}
        slither_pass = {}

        with tqdm(total=total_samples_cnt) as pbar:
            for sample in sample_files:
                path_sample = "{}//{}//".format(self.dataset_dir, sample)

                if os.path.isdir(path_sample):
                    pbar.set_description('Processing:{}'.format(sample))

                    if os.path.exists(path_sample + self.DONE_FLAG) or os.path.exists(path_sample + self.SLITHER_FAIL_FLAG):
                        pass  # if already have the flag, pass
                    else:
                        try: 
                            flag = self.do_analyze_for_target(path_sample)
                            if flag == self.SLITHER_ERROR:
                                slither_pass[sample] = 1
                                self.save_the_result_flag(path_sample, self.SLITHER_ERROR)
                            else:
                                # 记录下已经完成的标志位
                                self.save_the_result_flag(path_sample, self.CONSTRUCT_DONE)
                        except:
                            error_pass[sample] = 1
                            self.save_the_result_flag(path_sample, self.CONSTRUCT_FAIL)
                
                pbar.update(1)
            
        print(json.dumps(error_pass, indent=4 ,separators=(",", ":")))
        print("total faile = {}".format(len(error_pass)))
        print("slither faile = {}".format(len(slither_pass)))

    def save_the_result_flag(self, dir_name, flag):
        """
            After code representation construction, we have to save the flag file
        """
        if flag == self.CONSTRUCT_DONE:
            flag_file = self.DONE_FLAG

            # remove the fail_flag first
            if os.path.exists(dir_name + self.FAIL_FLAG):
                os.remove(dir_name + self.FAIL_FLAG)
            
        elif flag == self.CONSTRUCT_FAIL:
            flag_file = self.FAIL_FLAG

        elif flag == self.SLITHER_ERROR:
            flag_file = self.SLITHER_FAIL_FLAG
            
        else:
            raise RuntimeError("wrong flag key word")

        consturct_done_flag = dir_name + flag_file
        if not os.path.exists(consturct_done_flag):
            with open(consturct_done_flag, "w+") as f:
                f.write(flag)

    def clean_up(self):
        """
            删除所有flag文件
        """
        sample_files = os.listdir(self.dataset_dir)
        total_samples_cnt = len(sample_files)
        with tqdm(total=total_samples_cnt) as pbar:
            for sample in sample_files:
                
                path_sample = "{}//{}//".format(self.dataset_dir, sample)
                pbar.set_description('Cleaning:{}'.format(path_sample))

                if os.path.isdir(path_sample): 
                            
                    if os.path.exists(path_sample + self.AST_JSON_DIR):
                        shutil.rmtree(path_sample + self.AST_JSON_DIR)
                    
                    if os.path.exists(path_sample + self.SBP_JSON_DIR):
                        shutil.rmtree(path_sample + self.SBP_JSON_DIR)

                    if os.path.exists(path_sample + self.AST_DONE_FLAG):
                        os.remove(path_sample + self.AST_DONE_FLAG)

                    if os.path.exists(path_sample + self.DONE_FLAG):
                        os.remove(path_sample + self.DONE_FLAG)

                    if os.path.exists(path_sample + self.FAIL_FLAG):
                        os.remove(path_sample + self.FAIL_FLAG)
                        
                pbar.update(1)

    def clean_done_flag(self): 
        """
            删除所有done flag文件
        """
        sample_files = os.listdir(self.dataset_dir)
        for sample in sample_files:
            path_sample = "{}//{}//".format(self.dataset_dir, sample)
            if os.path.isdir(path_sample) and os.path.exists(path_sample + self.DONE_FLAG):
                os.remove(path_sample + self.DONE_FLAG)
            if os.path.isdir(path_sample) and os.path.exists(path_sample + self.FAIL_FLAG):
                os.remove(path_sample + self.FAIL_FLAG)

def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-clean', type=int, default=0)  # clean all flag
    parser.add_argument('-add', type=str, default=None)
    parser.add_argument('-t', type=str, default=None)
    parser.add_argument('-cfg', type=int, default=0)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-cd', type=int, default=0) # clean done flag
    parser.add_argument('-static', type=int, default=0)
    parser.add_argument('-slither_check', type=int, default=0)

    args = parser.parse_args()
    return args.clean, args.add, args.t, args.cfg, args.dataset, args.cd, args.static, args.slither_check



if __name__ == '__main__':

    clean, address, test, cfg, dataset, clean_done, static, slither_check = argParse()

    if address is not None:

        LOG_LEVEL = 10 # log debug
        analyze_wrapper = AnalyzeWrapper("dummy", save_png=1) # 创建假的
        analyze_wrapper.do_analyze_for_target("dataset//resumable_loop_2//{}//".format(address))
        # analyze_wrapper.do_analyze_for_target("example//{}//".format(address))

    elif test:

        LOG_LEVEL = 10 # log debug
        analyze_wrapper = AnalyzeWrapper("dummy", save_png=1)  # 创建假的
        analyze_wrapper.do_analyze_for_target("example//0x06a566e7812413bc66215b48d6f26321ddf653a9//")

    elif clean:

        if dataset is None:
            print("ERROR: 请输入数据集名称")

        elif click.confirm('!!! Clean all flag in dataset {} !!!!'.format(dataset), default=True):
            if click.confirm('!!! Again Clean all flag in dataset {} !!!!'.format(dataset), default=True):
                analyze_wrapper = AnalyzeWrapper("dataset//{}".format(dataset), save_png=0)
                analyze_wrapper.clean_up()

    elif dataset:

        analyze_wrapper = AnalyzeWrapper("dataset//{}".format(dataset), save_png=0)

        if static:
            analyze_wrapper.do_vulnerability_static_after_stmt_analyze()
            
        elif slither_check:
            analyze_wrapper.do_slither_check()

        elif clean_done:
            analyze_wrapper.clean_done_flag()
            print("=======>Do Clean Done Flag..........[DONE]<====================")
        
        else:
            LOG_LEVEL = 30 # log warning
            analyze_wrapper.do_analyze_for_dataset()
    
    else:
        print("!!! ERROR: 参数错误")
        


    
    
  
    
