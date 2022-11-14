def peculiar_calcu_metrics(test_file, predicate_file):
    prefix = "baseline_dataset_construct\\Peculiar\\"

    id_lable_map = {}
    TP = TN = FP = FN = 0

    with open(prefix + test_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            idx, label = line.split('\t')
            # print("idx :{} label:{}".format(idx, label))
            id_lable_map[idx] = int(label[:-1])

    with open(prefix + predicate_file, "r") as f:
        lines = f.readlines()
        for line in lines:

            # 结束符
            if "================" in line:
                break

            idx, predicate = line.split('\t')
            predicate = int(predicate[:-1])

            if idx in id_lable_map:
                lable = id_lable_map[idx]
                print("idx :{} predicate:{} label:{}".format(idx, predicate, lable))

                if predicate == lable:
                    if lable == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if lable == 1:
                        FP += 1
                    else:
                        FN += 1

    # FP -= 300
    # TN -= 550
    # FN -= 50
    print("TP {} TN {} FP {} FN {}".format(TP, TN, FP, FN))

    ACC = (TP + TN) / (TP + TN + FP + FN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print("ACC {} P {} R {} F1 {}".format(ACC, P, R, F1))
