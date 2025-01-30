from datetime import datetime

import numpy as np


def embeding(data, embed_size):
    result = []
    for X in data:
        tmp = np.array([np.concatenate([x, [0] * (embed_size - len(x))]) if len(x) < embed_size else x for x in X])
        result.append(tmp)
    return result


def normalization_array(data):
    max_num = -1e10
    min_num = 1e10

    for line in data:
        max_num = max(max_num, max(line))
        min_num = min(min_num, min(line))
    # print(max_num,min_num)
    out = []
    for line in data:
        epsilon = 1e-8
        temp = [(x - min_num) / (max_num - min_num + epsilon) for x in line]
        out.append(temp)
    return out, max_num, min_num


def normalization_seq(data):
    max_num = -1e10
    min_num = 1e10

    for line in data:
        for word in line:
            max_num = max(max_num, max(word))
            min_num = min(min_num, min(word))
    words = []
    out = []
    for line in data:
        words = []
        for word in line:
            temp = [(x - min_num) / (max_num - min_num) for x in word]
            words.append(temp)
        out.append(words)
    return out


def padding_array(X, max_len, padding=0):
    return np.array([np.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in X])


def padding_sequence(X, max_len, embed_size, padding=0):
    return np.array(
        [np.concatenate([x, [[padding] * embed_size] * (max_len - len(x))]) if len(x) < max_len else x for x in X])


def return_before(dataSet, max_num, min_num):
    out = []
    out = np.round(dataSet * (max_num - min_num) + min_num)
    '''for line in dataSet:
        temp = [round(x * (max_num - min_num)) + min_num for x in line]
        out.append(temp)'''
    return out


def load_array(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            # Remove '$' characters from the line
            line = line.replace('$', '')
            if line.strip() == "ERROR":
                # 将 "ERROR" 替换为一个包含单一值的列表
                dataSet.append([np.finfo(float).max])
            elif line.strip() == "":
                dataSet.append([np.finfo(float).min])  # Treat empty line as NaN
            else:
                words = line.split()
                data = [float(x) if x != 'NaN' else np.finfo(float).min if x != 'Infinity' else np.finfo(float).max for
                        x in words]
                dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], -1)
    return dataSet, seq_len, Max, Min


def load_array_split(filename, spliter):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split(spliter)
            data = [float(x) if len(x) > 0 else 0 for x in words]

            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], -1)
    return dataSet, seq_len, Max, Min


def load_data(filename):
    dataSet = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != "":
                if line == "null":
                    dataSet.append([0])  # 将"null"值替换为0，可以根据需要进行修改
                else:
                    try:
                        dataSet.append([int(line)])
                    except ValueError:
                        ascii_val = [ord(char) for char in line]
                        dataSet.append(ascii_val)
        # 归一化
        dataSet, max_num, min_num = normalization_array(dataSet)

        # 填充
        max_len = max(len(x) for x in dataSet)  # 获取最大长度
        dataSet = padding_sequence(dataSet, max_len, embed_size=1)

        # 构建返回格式
        seq_len = [len(x) for x in dataSet]
        dataSet = np.array(dataSet)
        Max = np.array([max_num])  # 最大值，以数组形式存储
        Min = np.array([min_num])  # 最小值，以数组形式存储

        return dataSet, seq_len, Max, Min


def write_data_to_file(dataSet, output_file):
    with open(output_file, 'w') as f:
        for data in dataSet:
            line = " ".join(str(x) for x in data)
            f.write(line + "\n")


# def load_sequence(filename):
#     dataSet = []
#
#     with open(filename, 'r') as f:
#         line = f.readline().strip()
#         while line:
#             # Check if the line contains 'Infinity$'
#             if 'Infinity$' in line or "ERROR$" in line:
#                 # Encode 'Infinity$' as a large positive integer
#                 encoded_line = [np.finfo(float).max]
#             else:
#                 # Encode other characters as their ASCII codes
#                 encoded_line = [ord(x) for x in line]
#             dataSet.append(encoded_line)
#             line = f.readline().strip()
#
#     # normalization
#     dataSet, Max, Min = normalization_array(dataSet)
#
#     # padding
#     max_len = len(max(dataSet, key=lambda x: len(x)))
#     dataSet = padding_array(dataSet, max_len)
#     # mask
#     seq_len = [len(x) for x in dataSet]
#     dataSet = np.array(dataSet)
#     dataSet = dataSet.reshape(dataSet.shape[0], max_len)
#
#     return dataSet, seq_len, Max, Min

def load_sequence(filename):
    dataSet = []

    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            # print(f"Processing line: {line}")  # 调试打印每一行
            if 'Infinity$' in line or "ERROR$" in line:
                encoded_line = [np.finfo(float).max]
            else:
                try:
                    # 编码所有字符，包括 # 在内
                    encoded_line = [ord(x) for x in line]
                except Exception as e:
                    print(f"Error processing line: {line}, Error: {e}")
                    line = f.readline().strip()
                    continue
            dataSet.append(encoded_line)
            line = f.readline().strip()
    if not dataSet:
        raise ValueError(f"No valid data found in {filename}!")

    # Normalization
    dataSet, Max, Min = normalization_array(dataSet)

    # Padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)

    # Mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet).reshape(dataSet.shape[0], max_len)
    print(f"Total samples in {filename}: {len(dataSet)}")
    return dataSet, seq_len, Max, Min


def write_data_to_file(dataSet, output_file):
    with open(output_file, 'w') as f:
        for data in dataSet:
            line = " ".join(str(x) for x in data)
            f.write(line + "\n")


def load_sequence_split(filename, spliter):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        lst = line.split(spliter)
        line = ' '.join(lst)
        while line:
            # Check if the line contains 'Infinity$'
            if 'Infinity$' in line or "ERROR$" in line:
                # Encode 'Infinity$' as a large positive integer
                encoded_line = [10000000000]
            else:
                # Encode other characters as their ASCII codes
                encoded_line = [ord(x) for x in line]
            dataSet.append(encoded_line)
            line = f.readline().strip()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)

    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Max, Min


def load_label(filename):
    labelSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            arr = line.split()
            label = [int(x) for x in arr]
            labelSet.append(label)
            line = f.readline()
    labelSet = np.array(labelSet)
    labelSet = labelSet.reshape(labelSet.shape[0], 1)
    return labelSet


def normalization_dimention(dataSet):
    maxs = np.max(dataSet, axis=0)
    mins = np.min(dataSet, axis=0)
    out = []
    for line in dataSet:
        temp = []
        for j in range(len(line)):
            if maxs[j] == mins[j]:
                num = (line[j] - mins[j]) / 1
            else:
                num = (line[j] - mins[j]) / (maxs[j] - mins[j])
            temp.append(num)
        out.append(temp)
    return out, maxs, mins


def load_before_time14(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = []
            words = line.split('#')
            data.append(int(words[0]))
            nums = words[1].split('-')
            for x in nums:
                if x != '':
                    data.append(int(x))
            dataSet.append(data)
            line = f.readline()

    # normalization
    # dataSet = normalization_dimention(dataSet)
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    # dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Max, Min


def load_after_time14(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = []
            nums = line.split('-')
            for x in nums:
                if x != '':
                    data.append(int(x))
            dataSet.append(data)
            line = f.readline()

    # normalization
    # dataSet = normalization_dimention(dataSet)
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    # dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Max, Min


def preprocess_datetime2(date_string):
    # 解析日期字符串
    date_object = datetime.strptime(date_string, "%a %b %d %H:%M:%S %Z %Y")
    # 将日期时间转换为数值类型（例如，Unix时间戳）
    timestamp = date_object.timestamp()
    return timestamp


def load_after_jacksondatabind_87(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = []
            date_string = line
            # 预处理日期时间
            timestamp = preprocess_datetime2(date_string)
            data.append(timestamp)
            dataSet.append(data)
            line = f.readline()

    # 归一化
    dataSet, Max, Min = normalization_array(dataSet)

    # padding
    max_len = max(len(x) for x in dataSet)
    # dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)

    return dataSet, seq_len, Max, Min


def load_data_time3(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split('#')
            data = [int(x) for x in words]
            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Maxs, Mins = normalization_dimention(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Maxs, Mins


def load_data_mockito6(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split()
            if words[0] != 'null':
                data = [int(x) for x in words]
            else:
                for word in words:
                    data = [ord(x) for x in list(word)]
            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], -1)
    return dataSet, seq_len, Max, Min


def load_data_math15(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split()
            if words[0] != 'Infinity':
                data = [float(x) for x in words]
            else:
                data = [0]
            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], -1)
    return dataSet, seq_len, Max, Min


def load_data_exp_time7(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split()
            data = [float(word) for word in words]
            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], -1)
    return dataSet, seq_len, Max, Min


def load_data_compress40(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split('#')
            lst = words[0].split()
            data = [int(x) for x in lst]
            data.append(int(words[1]))
            dataSet.append(data)
            line = f.readline()

    # normalization
    dataSet, Max, Min = normalization_array(dataSet)
    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Max, Min


def load_classification(filename):
    dataSet = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = []
            if line == 'true$\n' or line == 'true\n':
                data.append(1)
            elif line == 'ERROR$\n':
                data.append(float('inf'))  # 添加正无穷大
            elif line == 'false$\n' or line == 'false\n':
                data.append(0)
            dataSet.append(data)
            line = f.readline()
    # normalization
    dataSet, Max, Min = normalization_array(dataSet)

    # padding
    max_len = max(len(x) for x in dataSet)
    dataSet = padding_array(dataSet, max_len)
    # mask
    seq_len = [len(x) for x in dataSet]
    dataSet = np.array(dataSet)
    dataSet = dataSet.reshape(dataSet.shape[0], max_len)
    return dataSet, seq_len, Max, Min


def load_M1_before(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M1_after(filename):
    Y, seq_len_Y, max_y, min_y = load_classification(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M2_before(filename):
    Y, seq_len_Y, max_y, min_y = load_array_split(filename, '$')
    return Y, seq_len_Y, max_y, min_y


def load_M2_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M3_before(filename):
    X, seq_len_X, max_x, min_x = load_sequence(filename)
    return X, seq_len_X, max_x, min_x


def load_M3_after(filename):
    Y, seq_len_Y, max_y, min_y = load_sequence(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M4_before(filename):
    X, seq_len_X, max_x, min_x = load_array(filename)
    return X, seq_len_X, max_x, min_x


def load_M4_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M5_before(filename):
    X, seq_len_X, max_x, min_x = load_data_time3(filename)
    return X, seq_len_X, max_x, min_x


def load_M5_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M6_before(filename):
    X, seq_len_X, max_x, min_x = load_data_time3(filename)
    return X, seq_len_X, max_x, min_x


def load_M6_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M7_before(filename):
    X, seq_len_X, max_x, min_x = load_data_time3(filename)
    return X, seq_len_X, max_x, min_x


def load_M7_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M8_before(filename):
    X, seq_len_X, max_x, min_x = load_array_split(filename, '#')
    return X, seq_len_X, max_x, min_x


def load_M8_after(filename):
    Y, seq_len_Y, max_y, min_y = load_array(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M9_before(filename):
    X, seq_len_X, max_x, min_x = load_before_time14(filename)
    return X, seq_len_X, max_x, min_x


def load_M9_after(filename):
    Y, seq_len_Y, max_y, min_y = load_after_time14(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M10_before(filename):
    X, seq_len_X, max_x, min_x = load_before_time14(filename)
    return X, seq_len_X, max_x, min_x


def load_M10_after(filename):
    Y, seq_len_Y, max_y, min_y = load_after_time14(filename)
    return Y, seq_len_Y, max_y, min_y


def load_M11_before(filename):
    X, seq_len_X, max_x, min_x = load_sequence_split(filename, '$')
    return X, seq_len_X, max_x, min_x


before_func = {
    # TM
    'chart_1': load_M1_before,
    'chart_2': load_M11_before,
    'chart_3': load_M11_before,
    'chart_5': load_M2_before,
    'chart_7': load_M2_before,
    'chart_9': load_M11_before,
    'chart_10': load_M3_before,
    'chart_13': load_M4_before,
    'chart_16': load_M3_before,
    'chart_18': load_M2_before,
    'chart_20': load_M2_before,
    'chart_21': load_M3_before,
    'chart_22': load_M4_before,
    'chart_23': load_M11_before,
    'chart_24': load_M4_before,
    'time_3_addYears': load_M5_before,
    'time_3_addMonths': load_M6_before,
    'time_3_addWeeks': load_M7_before,
    'time_3_addDays': load_M8_before,
    'time_4': load_M4_before,
    'time_6': load_M3_before,
    'time_7_parseInto': load_M3_before,
    'time_7_mutableDateTime': load_M2_before,
    'time_8': load_M4_before,
    'time_9': load_M2_before,
    'time_10': load_M2_before,
    'time_12': load_M2_before,
    'time_13': load_M2_before,
    'time_15': load_M4_before,
    'time_14_minusMonths': load_M9_before,
    'time_14_plusMonths': load_M10_before,
    'time_17': load_M2_before,
    'time_18': load_M2_before,
    'time_22': load_M11_before,
    'time_24': load_M3_before,
    'time_26': load_M2_before,
    'time_27': load_M11_before,
    'lang_1': load_M3_before,
    'lang_6': load_M3_before,
    'lang_7': load_M3_before,
    'lang_8': load_M3_before,
    'lang_9': load_M1_after,
    'lang_10': load_M1_after,
    'lang_11': load_M4_before,
    'lang_12': load_M3_before,
    'lang_14': load_M3_before,
    'lang_16': load_M3_before,
    'lang_17': load_M3_before,
    'lang_19': load_M3_before,
    'lang_20': load_M3_before,
    'lang_21': load_M2_before,
    'lang_22': load_M2_before,
    'lang_24': load_M3_before,
    'lang_26': load_M2_before,
    'lang_27': load_M3_before,
    'lang_28': load_M3_before,
    'lang_29': load_M3_before,
    'lang_33': load_M11_before,
    'lang_36': load_M11_before,
    'lang_38': load_M1_before,
    'lang_39': load_M3_before,
    'lang_41': load_M3_before,
    'lang_43': load_M3_before,
    'lang_44': load_M11_before,
    'lang_45': load_M11_before,
    'lang_46': load_M3_before,
    'lang_48': load_M11_before,
    'lang_49': load_M2_before,
    'lang_50': load_M3_before,
    'lang_51': load_M2_before,
    'lang_52': load_M3_before,
    'lang_53': load_M1_before,
    'lang_54': load_M11_before,
    'lang_55': load_M3_before,
    'lang_57': load_M2_before,
    'lang_58': load_M3_before,
    'lang_59': load_M11_before,
    'lang_60': load_M2_before,
    'lang_61': load_M2_before,
    'lang_62': load_M3_before,
    'lang_64': load_M2_before,
    'lang_65': load_M3_before,
    'math_4j_1': load_M2_after,
    'math_4j_2': load_M2_after,
    'math_4j_9': load_M11_before,
    'math_4j_10': load_M2_before,
    'math_4j_11': load_M4_before,
    'math_4j_15': load_M4_before,
    'math_4j_16': load_M4_after,
    'math_4j_17': load_M4_after,
    'math_4j_21': load_M3_before,
    'math_4j_22': load_M2_before,
    'math_4j_26': load_M4_before,
    'math_4j_27': load_M4_before,
    'math_4j_30': load_M4_before,
    'math_4j_31': load_M4_before,
    'math_4j_32': load_M2_before,
    'math_4j_33': load_M2_before,
    'math_4j_34': load_M2_before,
    'math_4j_41': load_M2_before,
    'math_4j_42': load_M4_before,
    'math_4j_43': load_M4_before,
    'math_4j_47': load_M4_before,
    'math_4j_48': load_M3_before,
    'math_4j_49': load_M4_before,
    'math_4j_53': load_M4_before,
    'math_4j_58': load_M4_before,
    'math_4j_59': load_M2_before,
    'math_4j_62': load_M1_before,
    'math_4j_63': load_M2_before,
    'math_4j_64': load_M2_before,
    'math_4j_65': load_M2_before,
    'math_4j_66': load_M4_before,
    'math_4j_73': load_M4_before,
    'math_4j_74': load_M4_before,
    'math_4j_75': load_M4_before,
    'math_4j_79': load_M4_before,
    'math_4j_80': load_M4_before,
    'math_4j_81': load_M4_before,
    'math_4j_85': load_M4_before,
    'math_4j_88': load_M3_before,
    'math_4j_90': load_M4_before,
    'math_4j_91': load_M4_before,
    'math_4j_94': load_M4_before,
    'math_4j_95': load_M4_before,
    'math_4j_96': load_M3_before,
    'math_4j_97': load_M4_before,
    'math_4j_98': load_M2_before,
    'math_4j_105': load_M2_before,
    'math_4j_106': load_M2_before,
    'closure_3': load_M3_before,
    'closure_9': load_M3_before,
    'closure_15': load_M3_before,
    'closure_17': load_M3_before,
    'closure_19': load_M3_before,
    'closure_20': load_M3_before,
    'closure_23': load_M3_before,
    'closure_26': load_M3_before,
    'closure_27': load_M3_before,
    'closure_28': load_M3_before,
    'closure_29': load_M3_before,
    'closure_35': load_M3_before,
    'closure_37': load_M3_before,
    'closure_39': load_M3_before,
    'closure_40': load_M3_before,
    'closure_41': load_M3_before,
    'closure_44': load_M3_before,
    'closure_50': load_M4_before,
    'closure_56': load_M3_before,
    'closure_57': load_M3_before,
    'closure_58': load_M3_before,
    'closure_62': load_M3_before,
    'closure_65': load_M3_before,
    'closure_67': load_M3_before,
    'closure_68': load_M3_before,
    'closure_71': load_M1_before,
    'closure_77': load_M2_before,
    'closure_83': load_M1_before,
    'closure_85': load_M2_before,
    'closure_87': load_M3_before,
    'closure_88': load_M3_before,
    'closure_91': load_M11_before,
    'closure_97': load_M3_before,
    'closure_103': load_M3_before,
    'closure_105': load_M4_before,
    'closure_107': load_M3_before,
    'closure_108': load_M4_before,
    'closure_109': load_M3_before,
    'closure_112': load_M3_before,
    'closure_114': load_M3_before,
    'closure_115': load_M2_before,
    'closure_116': load_M3_before,
    'closure_117': load_M3_before,
    'closure_118': load_M11_before,
    'closure_125': load_M3_before,
    'closure_126': load_M11_before,
    'closure_130': load_M2_before,
    'closure_133': load_M4_before,
}

after_func = {
    # TM
    'chart_1': load_M2_before,
    'chart_2': load_M3_after,
    'chart_3': load_M11_before,
    'chart_5': load_M3_after,
    'chart_7': load_M4_after,
    'chart_9': load_M3_after,
    'chart_10': load_M3_after,
    'chart_13': load_M3_after,
    'chart_16': load_M3_after,
    'chart_18': load_M3_after,
    'chart_20': load_M3_after,
    'chart_21': load_M3_after,
    'chart_22': load_M4_after,
    'chart_23': load_M11_before,
    'chart_24': load_M1_after,
    'time_3_addYears': load_M5_after,
    'time_3_addMonths': load_M6_after,
    'time_3_addWeeks': load_M7_after,
    'time_3_addDays': load_M8_after,
    'time_4': load_M3_after,
    'time_6': load_M3_after,
    'time_7_parseInto': load_M3_after,
    'time_7_mutableDateTime': load_M3_after,
    'time_8': load_M3_after,
    'time_9': load_M11_before,
    'time_10': load_M3_after,
    'time_12': load_M3_after,
    'time_13': load_M11_before,
    'time_14_minusMonths': load_M9_after,
    'time_14_plusMonths': load_M10_after,
    'time_15': load_M3_after,
    'time_17': load_M3_after,
    'time_18': load_M3_after,
    'time_22': load_M3_after,
    'time_24': load_M3_after,
    'time_26': load_M3_after,
    'time_27': load_M3_after,
    'lang_1': load_M3_after,
    'lang_6': load_M3_after,
    'lang_7': load_M3_after,
    'lang_8': load_M3_after,
    'lang_9': load_M3_after,
    'lang_10': load_M3_after,
    'lang_11': load_M3_after,
    'lang_12': load_M3_after,
    'lang_14': load_M1_after,
    'lang_16': load_M3_after,
    'lang_17': load_M3_after,
    'lang_19': load_M3_after,
    'lang_20': load_M3_after,
    'lang_21': load_M11_before,
    'lang_22': load_M2_after,
    'lang_24': load_M1_after,
    'lang_26': load_M3_after,
    'lang_27': load_M1_after,
    'lang_28': load_M3_after,
    'lang_29': load_M3_after,
    'lang_33': load_M3_after,
    'lang_36': load_M11_before,
    'lang_38': load_M3_after,
    'lang_39': load_M3_after,
    'lang_41': load_M3_after,
    'lang_43': load_M1_after,
    'lang_44': load_M3_after,
    'lang_45': load_M11_before,
    'lang_46': load_M3_after,
    'lang_48': load_M1_after,
    'lang_49': load_M3_after,
    'lang_50': load_M3_after,
    'lang_51': load_M2_after,
    'lang_52': load_M3_after,
    'lang_53': load_M3_after,
    'lang_54': load_M3_after,
    'lang_55': load_M1_after,
    'lang_57': load_M3_after,
    'lang_58': load_M3_after,
    'lang_59': load_M3_after,
    'lang_60': load_M1_after,
    'lang_61': load_M2_after,
    'lang_62': load_M1_after,
    'lang_64': load_M3_after,
    'lang_65': load_M1_after,
    'math_4j_1': load_M3_after,
    'math_4j_2': load_M4_after,
    'math_4j_9': load_M4_after,
    'math_4j_10': load_M4_after,
    'math_4j_11': load_M4_after,
    'math_4j_15': load_M4_after,
    'math_4j_16': load_M4_after,
    'math_4j_17': load_M3_after,
    'math_4j_21': load_M2_after,
    'math_4j_22': load_M2_after,
    'math_4j_26': load_M3_after,
    'math_4j_27': load_M4_after,
    'math_4j_30': load_M1_after,
    'math_4j_31': load_M3_after,
    'math_4j_32': load_M3_after,
    'math_4j_33': load_M4_after,
    'math_4j_34': load_M4_after,
    'math_4j_41': load_M4_after,
    'math_4j_42': load_M4_after,
    'math_4j_43': load_M4_after,
    'math_4j_47': load_M4_after,
    'math_4j_48': load_M3_after,
    'math_4j_49': load_M3_after,
    'math_4j_53': load_M4_after,
    'math_4j_58': load_M1_after,
    'math_4j_59': load_M3_after,
    'math_4j_62': load_M1_after,
    'math_4j_63': load_M4_after,
    'math_4j_64': load_M4_after,
    'math_4j_65': load_M4_after,
    'math_4j_66': load_M4_after,
    'math_4j_73': load_M4_after,
    'math_4j_74': load_M4_after,
    'math_4j_75': load_M4_after,
    'math_4j_79': load_M3_after,
    'math_4j_80': load_M4_after,
    'math_4j_81': load_M1_after,
    'math_4j_85': load_M3_after,
    'math_4j_88': load_M4_after,
    'math_4j_90': load_M4_after,
    'math_4j_91': load_M4_after,
    'math_4j_94': load_M4_after,
    'math_4j_95': load_M3_after,
    'math_4j_96': load_M3_after,
    'math_4j_97': load_M1_after,
    'math_4j_98': load_M3_after,
    'math_4j_105': load_M4_after,
    'math_4j_106': load_M4_after,
    'closure_3': load_M3_after,
    'closure_9': load_M3_after,
    'closure_15': load_M3_after,
    'closure_17': load_M3_after,
    'closure_19': load_M3_after,
    'closure_20': load_M3_after,
    'closure_23': load_M4_after,
    'closure_26': load_M3_after,
    'closure_27': load_M3_after,
    'closure_28': load_M3_after,
    'closure_29': load_M3_after,
    'closure_35': load_M4_after,
    'closure_37': load_M1_after,
    'closure_39': load_M4_after,
    'closure_40': load_M1_after,
    'closure_41': load_M3_after,
    'closure_44': load_M4_after,
    'closure_50': load_M4_after,
    'closure_56': load_M3_after,
    'closure_57': load_M3_after,
    'closure_58': load_M11_before,
    'closure_62': load_M3_after,
    'closure_65': load_M3_after,
    'closure_67': load_M3_after,
    'closure_68': load_M3_after,
    'closure_71': load_M4_after,
    'closure_77': load_M3_after,
    'closure_83': load_M4_after,
    'closure_85': load_M4_after,
    'closure_87': load_M1_after,
    'closure_88': load_M1_after,
    'closure_91': load_M3_after,
    'closure_97': load_M3_after,
    'closure_103': load_M3_after,
    'closure_105': load_M3_after,
    'closure_107': load_M3_after,
    'closure_108': load_M3_after,
    'closure_109': load_M3_after,
    'closure_112': load_M3_after,
    'closure_114': load_M3_after,
    'closure_115': load_M3_after,
    'closure_116': load_M3_after,
    'closure_117': load_M3_after,
    'closure_118': load_M3_after,
    'closure_125': load_M3_after,
    'closure_126': load_M1_after,
    'closure_130': load_M3_after,
    'closure_133': load_M3_after,
}
