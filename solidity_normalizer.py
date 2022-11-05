import os
import re
    
class SrcNormalize:
    def __init__(self, target_dir) -> None:
        self.target_dr = target_dir
        self.norm_dir = target_dir + "normalized_src//"
        self.target_sols = {}
        
        self.SafeLowLevelCallMap = ['safeTransferFrom', 'safeTransfer', "sendValue","functionCallWithValue",  "functionCall", "functionStaticCall"]
        self.TodCallMap = [".safeApprove", ".approve", ".safeIncreaseAllowance", ".safeDecreaseAllowance"]
        
        self.environment_prepare()

    def environment_prepare(self):

        if not os.path.exists(self.norm_dir):
            os.mkdir(self.norm_dir)
        
        _temp_files = os.listdir(self.target_dr)
        for _temp_file in _temp_files:
            if str(_temp_file).endswith(".sol"):
                self.target_sols[self.target_dr + _temp_file] = _temp_file
        
        # print(self.target_sols)

    def normalize_files(self):
        for target in self.target_sols:
            
            lines = open(target).readlines()
            norm_sol_file = self.norm_dir + self.target_sols[target]
            self._do_normalize_file(lines, norm_sol_file)

    def _do_normalize_file(self, lines, norm_sol_file):
        with open(norm_sol_file, "w+") as n_f:
            for line in lines:
                delete_flag = 0
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

                    op_new = " {} ".format("+")
                    key = ".{}".format("add")
                    line = line.replace(key, op_new)

                if ".sub" in line:
                    result = re.search(r'\.sub\((.*?)\)',line) # .sub(匹配)
                    if result is None:
                        pass
                    else:
                        _params = result.group(1)  # amount, "SafeERC20: decreased allowance below zero"
                        _first_sep = str(_params).find(',') # 第一个 ","
                        if -1 != _first_sep:
                            key = _params[_first_sep:]
                            op_new = ""
                            line = line.replace(key, op_new)

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

                    op_new = " {} ".format("/")
                    key = ".{}".format("div")
                    line = line.replace(key, op_new)

                if  "function" in line and "nonReentrant" in line:
                    line = line.replace("nonReentrant", "")
                
                if  "function" in line and "onlyOwner" in line:
                    line = line.replace("onlyOwner", "")
                
                for llc in self.SafeLowLevelCallMap:
                    if llc in line:
                        line = line.replace(llc, "call")
                
                for tod in self.TodCallMap:
                    if tod in line:
                        if tod != ".approve": 
                            line = line.replace(tod, ".approve")
                        else:
                            if "0" in line:
                                delete_flag = 1

                if "toUint" in line:
                    result = re.search(r'\.toUint(.*?)\(\)',line)
                    if result is not None:
                        _cast = "uint{}".format(result.group(1))
                        pos = result.span()[0] - 1

                        # xxx var.toUint(xxx) 由.向前找第一个不是字符的位置
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
                        
                
                if "toInt" in line:
                    result = re.search(r'\.toInt(.*?)\(\)',line)
                    if result is not None:
                        _cast = "int{}".format(result.group(1))
                        pos = result.span()[0] - 1

                        # xxx var.toUint(xxx) 由.向前找第一个不是字符的位置
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
                        

                if 0 == delete_flag:
                    n_f.write(line)


if __name__ == '__main__':
    
    # norlamize_function_with_safemath(EXAMPLE_PERFIX + '0x06a566e7812413bc66215b48d6f26321ddf653a9/' + "Gauge.sol")
    nom = SrcNormalize("example//0x00f401c1e60C9eBf48b1c22c0D87250Cc54F979f//")
    nom.normalize_files()
    