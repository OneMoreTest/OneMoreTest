import os
import time
import numpy as np
from tensorflow import keras
import defines
import loadData

def calculate_dis_seq(y_pred, y_true, seq_length):
    # 最小编辑距离，字符不匹配就+1
    distance = 0
    max_dis = 0
    min_dis = 1000
    match_num = 0
    cnt = 0
    lst = []
    for i in range(0, len(y_pred)):
        length = seq_length[i]
        lt_pred = np.array(y_pred[i][0:length])
        lt_true = np.array(y_true[i][0:length])
        edit_dis = sum(lt_pred != lt_true) / (length * 2)
        if edit_dis <= 0.05:
            match_num += 1
        elif cnt < 10:
            lst.append((i, lt_pred, lt_true))
            cnt += 1
        dis = float(pow(edit_dis, 2))
        max_dis = max(dis, max_dis)
        min_dis = min(dis, min_dis)
        distance += dis
    distance /= len(y_pred)
    match_rate = match_num / len(y_pred)
    # 打印部分编辑距离
    # print(f"Match rate: {match_rate}, Max distance: {max_dis}, Min distance: {min_dis}")
    # print(f"Sample sequence mismatches: {lst[:3]} (showing up to 3 mismatches)")
    return distance, match_rate, max_dis, min_dis

def calculate_dis_array(y_pred, y_true, arr_length):
    # 计算每个样本的每个参数之间的差值取平均值
    distance = 0
    match_num = 0
    Sum = 0
    for i in range(0, len(y_pred)):
        tmp = 0
        length = arr_length[i]
        Sum += length
        for j in range(0, len(y_pred[i])):
            if j >= length:
                break
            if y_true[i][j] == 0:
                array_dis = abs(y_pred[i][j] - y_true[i][j])
            else:
                array_dis = abs((y_pred[i][j] - y_true[i][j]) / y_true[i][j])
            if array_dis <= 0.05:
                match_num += 1
            tmp += array_dis
        distance += tmp / length
    distance /= len(y_pred)
    match_rate = match_num / Sum
    # print(f"Average distance: {distance}, Match rate: {match_rate}")
    return distance, match_rate

def test(rootpath, name, dataset):
    modelpath = os.path.join(rootpath, 'model', name + '.hdf5')
    model = keras.models.load_model(modelpath)
    model.summary()

    train_num = 1000000
    batch_size = 512
    max_pre1 = 0
    max_pre3 = 0
    max_pre5 = 0
    max_pre10 = 0
    max_pre100 = 0
    path = os.path.join(rootpath, dataset, name)
    before_file = os.path.join(path, 'before_data.txt')
    after_file = os.path.join(path, 'after_data.txt')
    label_file = os.path.join(path, 'label.txt')
    X, seq_len_X, max_x, min_x = loadData.before_func.get(name)(before_file)
    Y, seq_len_Y, max_y, min_y = loadData.after_func.get(name)(after_file)
    with open(after_file, 'r') as f:
        Y_orin = f.readlines()
    with open(before_file, 'r') as f:
        X_orin = f.readlines()
    with open(label_file, 'r') as f:
        Label = f.readlines()

    Y_original = loadData.return_before(Y, max_y, min_y)
    Y = np.array(Y)
    Label = loadData.load_label(label_file)
    embed_X = X.shape[1]
    embed_Y = Y.shape[1]
    # print(f"Data shape: X={X.shape}, Y={Y.shape}")

    x_test = X[train_num:]
    y_true = Y[train_num:]
    label_test = Label[train_num:]
    y_original_test = Y_original[train_num:]
    y_len_test = seq_len_Y[train_num:]

    print(f"Test data shape: {x_test.shape}")
    y_pred = model.predict(x_test)

    # # 打印部分预测值与真实值
    # print("Sample predictions vs actual values:")
    # for i in range(100):
    #     print(f"Prediction: {y_pred[i]}, Actual: {y_true[i]}")

    dis = np.mean(np.square(y_pred - y_true), axis=1)
    # print(f"Distance sample: {dis[:100]}")

    # Distance-based metrics
    distance_seq, match_rate_seq, max_dis_seq, min_dis_seq = calculate_dis_seq(y_pred, y_true, y_len_test)
    distance_array, match_rate_array = calculate_dis_array(y_pred, y_true, y_len_test)
    # print(f"Overall sequence distance: {distance_seq}, Match rate: {match_rate_seq}")
    # print(f"Overall array distance: {distance_array}, Match rate: {match_rate_array}")

    loss_index = np.argsort(dis)
    samples_index = loss_index[-1 * int(100):][::-1]

    label_output = []
    for j in samples_index:
        label_output.append(label_test[j])
    fn100 = np.sum(label_output)
    tn100 = 100 - fn100
    max_pre100 = max(max_pre100, tn100)
    print(f"Top 100: TN={tn100}, FN={fn100}, Max Precision 100={max_pre100}")

    # top-10
    samples_index = loss_index[-1 * int(10):][::-1]
    label_output = []
    for j in samples_index:
        label_output.append(label_test[j])
    fn10 = np.sum(label_output)
    tn10 = 10 - fn10
    max_pre10 = max(max_pre10, tn10)
    print(f"Top 10: TN={tn10}, FN={fn10}, Max Precision 10={max_pre10}")

    # top-3
    samples_index = loss_index[-1 * int(3):][::-1]
    label_output = []
    for j in samples_index:
        label_output.append(label_test[j])
    fn3 = np.sum(label_output)
    tn3 = 3 - fn3
    max_pre3 = max(max_pre3, tn3)
    print(f"Top 3: TN={tn3}, FN={fn3}, Max Precision 3={max_pre3}")

    # top-5
    samples_index = loss_index[-1 * int(5):][::-1]
    label_output = []
    for j in samples_index:
        label_output.append(label_test[j])
    fn5 = np.sum(label_output)
    tn5 = 5 - fn5
    max_pre5 = max(max_pre5, tn5)
    print(f"Top 5: TN={tn5}, FN={fn5}, Max Precision 5={max_pre5}")

    # top-1
    samples_index = loss_index[-1:]
    if label_test[samples_index] == 1:
        fn1 = 1
    else:
        fn1 = 0
    tn1 = 1 - fn1
    max_pre1 = max(max_pre1, tn1)
    print(f"Top 1: TN={tn1}, FN={fn1}, Max Precision 1={max_pre1}")

    # 打印 top-100 样本的详细信息到文件
    Y_orin = np.array(Y_orin)
    X_orin = np.array(X_orin)
    Label = np.array(Label)

    top100_samples_index = loss_index[-1 * int(100):]
    top10_samples_index = loss_index[-1 * int(10):]
    top5_samples_index = loss_index[-1 * int(5):]
    top3_samples_index = loss_index[-1 * int(3):]
    top1_samples_index = loss_index[-1:]

    output_dir = os.path.join(rootpath, dataset, name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'top100.txt'), 'w') as f:
        f.write("Top 100 test cases:\n")
        f.write("input\toutput\n")
        for i in top100_samples_index:
            f.write(f"{X_orin[i + 1000000]}{Y_orin[i + 1000000]}{Label[i + 1000000]}\n")

    with open(os.path.join(output_dir, 'top10.txt'), 'w') as f:
        f.write("Top 10 test cases:\n")
        f.write("input\toutput\n")
        for i in top10_samples_index:
            f.write(f"{X_orin[i + 1000000]}{Y_orin[i + 1000000]}{Label[i + 1000000]}\n")

    with open(os.path.join(output_dir, 'top5.txt'), 'w') as f:
        f.write("Top 5 test cases:\n")
        f.write("input\toutput\n")
        for i in top5_samples_index:
            f.write(f"{X_orin[i + 1000000]}{Y_orin[i + 1000000]}{Label[i + 1000000]}\n")

    with open(os.path.join(output_dir, 'top3.txt'), 'w') as f:
        f.write("Top 3 test cases:\n")
        f.write("input\toutput\n")
        for i in top3_samples_index:
            f.write(f"{X_orin[i + 1000000]}{Y_orin[i + 1000000]}{Label[i + 1000000]}\n")

    with open(os.path.join(output_dir, 'top1.txt'), 'w') as f:
        f.write("Top 1 test case:\n")
        f.write("input\toutput\n")
        for i in top1_samples_index:
            f.write(f"{X_orin[i + 1000000]}{Y_orin[i + 1000000]}{Label[i + 1000000]}\n")

if __name__ == '__main__':
    start = time.perf_counter()
    rootPath = os.path.abspath('.')
    print(f"Root path: {rootPath}")
    for i, name in enumerate(defines.TM):
        print(f"---------------------- Testing {name.value} ----------------------")
        test(rootPath, name.value, 'TM')
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
