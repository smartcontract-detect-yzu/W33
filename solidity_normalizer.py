import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from tqdm import tqdm

vul_type_map = {
    "SafeMath": 0,
    "low-level call": 0,
    "safe cast": 0,
    "transaction order dependency": 0,
    "nonReentrant": "re",
    "onlyOwner": 0,
    "resumable_loop": 0
}
class SrcNormalize:
    def __init__(self, target_dir, contract, vul_type) -> None:
        self.target_dir = target_dir
        self.sbp_dir = self.target_dir + "sbp_json//"
        self.norm_dir = target_dir + f"normalized_src_{vul_type}//"
        self.contract = contract
        self.target_sols = {}
        self.vul_type = vul_type
        self.vul_lable = 0
        
        self.SafeLowLevelCallMap = ['safeTransferFrom', 'safeTransfer', "sendValue","functionCallWithValue",  "functionCall", "functionStaticCall"]
        self.TodCallMap = [".safeApprove", ".approve", ".safeIncreaseAllowance", ".safeDecreaseAllowance"]
        
        self.compile_file = None
        self.compile_ver = None

        self.environment_prepare()

    def environment_prepare(self):

        if os.path.exists(self.norm_dir):
            shutil.rmtree(self.norm_dir)
        os.mkdir(self.norm_dir)

        # 获得当前样本的标签
        sbp_json_files = os.listdir(self.sbp_dir)
        for sbp_file in sbp_json_files:
            with open(sbp_file, "r") as f:
                sbp_infos = json.load(f)
                function_sbp_infos = sbp_infos["function_sbp_infos"]
                for _f_info in function_sbp_infos:
                    _f_label = _f_info["lable_infos"]["label"]
                    if vul_type_map[_f_label] == self.vul_type:
                        self.vul_lable = 1
                        break
        
        # if not os.path.exists(self.target_dir + "download_done.txt"):
        if 'sbp_dataset' in self.target_dir:
            for dataset in ["dataset//reentrancy//", 'dataset//dataset_reloop//']:
                if os.path.exists(dataset + self.contract):
                    sol_files = os.listdir(dataset + self.contract)
                    for sol_file in sol_files:
                        if str(sol_file).endswith(".sol") or str(sol_file) == "download_done.txt":
                            src = "{}//{}".format(dataset + self.contract, sol_file)
                            shutil.copy(src, self.target_dir)
                    break
                    
		
        # shutil.copy(self.target_dir + "download_done.txt", self.norm_dir)
        with open(self.target_dir + "download_done.txt") as f:
            sol_infos = json.load(f)
            self.compile_file = sol_infos["name"]
            self.compile_ver = sol_infos["ver"]
		
        _temp_files = os.listdir(self.target_dir)
        for _temp_file in _temp_files:
            if str(_temp_file).endswith(".sol"):
                self.target_sols[self.target_dir + _temp_file] = _temp_file
        
        # print(self.target_sols)

    def __do_normal_safecast(self, line):

        if "toUint" in line:
            result = re.search(r'\.toUint(.*?)\(\)',line)
            # if result is not None and result.group(1).isdigit(): # 只能识别 toIntxxx 不能识别 toIntxxxSafe 等接口
            if result is not None:
                _cast = "uint{}".format(result.group(1))
                if "Safe" in _cast:
                    _cast = _cast.strip("Safe")
                    # return line

                pos = result.span()[0] - 1

                while pos > 0:
                    # if (not line[pos].isalpha()) and (line[pos] not in [".", "[", "]"]):
                    if line[pos].isspace():  # xxx var.toUint(xxx) 由.向前找第一个不是空的位置
                        pos += 1
                        break
                    pos -= 1

                var_name = line[pos:result.span()[0]]
                
                org = "{}{}".format(var_name, result.group(0))
                new = "{}({})".format(_cast, var_name)
                line = line.replace(org, new)
                
        
        if "toInt" in line:
            result = re.search(r'\.toInt(.*?)\(\)',line)
            # if result is not None and result.group(1).isdigit(): # 只能识别 toIntxxx 不能识别 toIntxxxSafe 等接口
            if result is not None:
                _cast = "int{}".format(result.group(1))
                if "Safe" in _cast:
                    _cast = _cast.strip("Safe")
                    # return line

                pos = result.span()[0] - 1

                # xxx var.toUint(xxx) 由.向前找第一个不是空字符的位置
                while pos > 0:
                    # if (not line[pos].isalpha()) and (line[pos] not in [".", "[", "]"]):
                    if line[pos].isspace():
                        pos += 1
                        break
                    pos -= 1
                var_name = line[pos:result.span()[0]]
                org = "{}{}".format(var_name, result.group(0))
                new = "{}({})".format(_cast, var_name)
                line = line.replace(org, new)

        return line                

    def __do_normal_safemath(self, line):

        if ".add" in line:
            result = re.search(r'\.add\((.*?)\)',line) # .sub(匹配)
            if result is None:
                # rewardPerTokenStored.add(
                #     lastTimeRewardApplicable().sub(lastUpdateTime).mul(rewardRate).mul(1e18).div(derivedSupply)
                # );
                pass

            else:
                _params = result.group(1)  # amount, "SafeERC20: decreased allowance below zero"
                _first_sep = str(_params).find(',') # 第一个 ","
                if -1 != _first_sep:
                    key = _params[_first_sep:]
                    op_new = ""
                    line = line.replace(key, op_new)
                else:
                    op_new = " {} ".format("+")
                    key = ".{}".format("add")
                    line = line.replace(key, op_new)

        if ".sub" in line:
            result = re.search(r'\.sub\((.*?)\)',line) # .sub(匹配)
            if result is None:
                pass
            else:
                _params = result.group(1)  # amount, "SafeERC20: decreased allowance below zero"
                _first_sep = str(_params).find(',') # 存在',' 表面方法存在两个入参
                if -1 != _first_sep:
                    key = _params[_first_sep:]
                    op_new = ""
                    line = line.replace(key, op_new)
                else:
                    op_new = " {} ".format("-")
                    key = ".{}".format("sub")
                    line = line.replace(key, op_new)

        if ".mul" in line:
            result = re.search(r'\.mul\((.*?)\)',line) # .sub(匹配)
            if result is None:
                pass
            else:
                _params = result.group(1)  # amount, "SafeERC20: decreased allowance below zero"
                _first_sep = str(_params).find(',') # 第一个 ","
                if -1 != _first_sep:
                    key = _params[_first_sep:]
                    op_new = ""
                    line = line.replace(key, op_new)
                else:
                    op_new = " {} ".format("*")
                    key = ".{}".format("mul")
                    line = line.replace(key, op_new)

        if ".div" in line:
            result = re.search(r'\.div\((.*?)\)',line) # .sub(匹配)
            if result is None:
                pass
            else:
                _params = result.group(1)  # amount, "SafeERC20: decreased allowance below zero"
                _first_sep = str(_params).find(',') # 第一个 ","
                if -1 != _first_sep:
                    key = _params[_first_sep:]
                    op_new = ""
                    line = line.replace(key, op_new)
                else:    
                    op_new = " {} ".format("/")
                    key = ".{}".format("div")
                    line = line.replace(key, op_new)
        
        return line
    

    def __do_normal_re(self, line):

        if  "function" in line and "nonReentrant" in line:
            line = line.replace("nonReentrant", "")
        
        return line

    def __do_normal_onlyowner(self, line):
        if  "function" in line and "onlyOwner" in line:
            if "onlyOwner()" in line:
                line = line.replace("onlyOwner()", "")
            else:
                line = line.replace("onlyOwner", "")
        return line

    def __do_normal_llc(self, line):

        for llc in self.SafeLowLevelCallMap:
            if llc in line:
                line = line.replace(llc, "call")

        return line

    def __do_normal_tod(self, line):
        delete_flag = 0
        for tod in self.TodCallMap:
            if tod in line:
                if tod != ".approve": 
                    line = line.replace(tod, ".approve")
                else:
                    if "0" in line:
                        delete_flag = 1
        return delete_flag, line

    def _do_normalize_file(self, lines, norm_sol_file, vul_type):
        with open(norm_sol_file, "w+") as n_f:
            for line in lines:
                if len(line.lstrip()) > 0 and line.lstrip()[0] == ".":
                    pass
                
                delete_flag = 0
                if vul_type == "re":
                    line = self.__do_normal_re(line)
                    
                else:
                    line = self.__do_normal_safecast(line)
                    line = self.__do_normal_safemath(line)
                    line = self.__do_normal_re(line)
                    line = self.__do_normal_onlyowner(line)
                    line = self.__do_normal_llc(line)
                    delete_flag, line = self.__do_normal_tod(line)

                if 0 == delete_flag:
                    n_f.write(line)


    def _do_test_compile(self):
        """
        预编译，查看是否下载的代码能够编译成功
        """
        pwd = os.getcwd() # 记住当前目录
        os.chdir(self.norm_dir)
        try:
            # For different OS, with different solc select method
            if platform.system() == "Windows":
                solc_path = "{}{}{}".format("solc_compiler\\", self.compile_ver, "\\solc-windows.exe")
                compile_ret = subprocess.check_call([solc_path, self.compile_ver])
            else:
                subprocess.check_call(["solc-select", "use", self.compile_ver])
                compile_ret = subprocess.check_call(["solc", self.compile_file])
        except: 

            # NOTE: 一旦编译失败就会抛出异常，此处必须接住异常并且切换工作目录
            print("编译失败:{} {}".format(self.compile_file, self.compile_ver, self.norm_dir))
            compile_ret = 1  # 编译失败
            os.chdir(pwd)
            return 1

        os.chdir(pwd)
        return 0
    
    def normalize_files(self):

        for target in self.target_sols:
            lines = open(target).readlines()
            norm_sol_file = self.norm_dir + self.target_sols[target]
            self._do_normalize_file(lines, norm_sol_file, self.vul_type)
        
        return self._do_test_compile()
        
def normalize_dataset(dataset_dir, vul_type):
    contract_samples = os.listdir(dataset_dir)
    total_samples_cnt = len(contract_samples)
    fail_cnt = 0 
    with tqdm(total=total_samples_cnt) as pbar:
        for _contract in contract_samples:
            path_sample = "{}//{}//".format(dataset_dir, _contract)

            if os.path.isdir(path_sample):
                pbar.set_description('Processing:{}'.format(_contract))
                
                # if already have the flag, pass
                if dataset_dir != "dataset//sbp_dataset" and not os.path.exists(path_sample + "construct_done.flag"):
                    pass 
                else:
                    nom = SrcNormalize(path_sample, _contract, vul_type)
                    fail_cnt += nom.normalize_files()
                    
            pbar.update(1)
    print("失败个数为: ", fail_cnt)

def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-vul', type=str, default=None)
    args = parser.parse_args()
    return args.dataset, args.vul

if __name__ == '__main__':

    dataset, _type = argParse()
    if dataset is not None:
        normalize_dataset("dataset//" + dataset, _type)
        


    # norlamize_function_with_safemath(EXAMPLE_PERFIX + '0x06a566e7812413bc66215b48d6f26321ddf653a9/' + "Gauge.sol")
    # nom = SrcNormalize("example//0x00f401c1e60C9eBf48b1c22c0D87250Cc54F979f//")
    # nom.normalize_files()
    