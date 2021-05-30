import pandas as pd
import numpy as np
import random

saving = []


def Ent(data_set):
    """
    计算交叉熵
    :param data_set: 输入一组样本数据n*(features; labels) list
    :return: 本组数据的交叉熵 float
    """
    classify = []  # labels
    number = []  # label对应数量
    length = 0
    for i in range(len(data_set)):
        if data_set[i][-1] in classify:
            idx = classify.index(data_set[i][-1])
            number[idx] = number[idx] + 1
        else:
            classify.append(data_set[i][-1])
            number.append(1)
            length = length + 1

    entropy = 0
    for i in range(length):
        pro = number[i] / len(data_set)
        if pro is not 0:
            entropy = entropy + (-pro) * np.log2(pro)

    return entropy


def info_gain(data_origin, data_list):
    """
    计算信息增益
    :param data_origin: 原始的数据组 array
    :param data_list: 划分后的n组数据 n*(features; labels)*n1 list 元素为array
    :return: 划分前后的信息增益 float
    """
    after_entropy = 0
    for item in iter(data_list):
        after_entropy = after_entropy + Ent(item)

    return Ent(data_origin) - after_entropy


def data_load(filename='iris.csv'):
    """
    将数据读入并进行处理
    :param filename:
    :return:
    """
    csv_er = pd.read_csv(filename)
    val = csv_er.values
    DataList = []
    for i in range(val.shape[0]):
        DataList.append(val[i])
    return DataList


def select_boundary(iris, num_features):
    """

    :param num_features:
    :param iris: 等待划分的连续数据变量 list
    :return:
    """
    boundary = []
    for i in num_features:
        # 对iris[i]进行排序并求中位点
        iris.sort(key=lambda d: d[i])
        gain = 0
        bound = -1
        for j in range(len(iris) - 1):
            middle = (iris[j][i] + iris[j + 1][i]) / 2
            list1 = []
            list2 = []
            for item in iris:
                if item[i] >= middle:
                    list1.append(item)
                else:
                    list2.append(item)
            gain = max(info_gain(iris, [list1, list2]), gain)
            if info_gain(iris, [list1, list2]) >= gain:
                bound = middle
        boundary.append([gain, bound])
    return boundary


def is_same_cat(processed_data):
    """

    :param processed_data: 处理后的数据 list
    :return: True or False
    """
    if bool(processed_data) is False:
        return False
    else:
        compare = processed_data[0][-1]
        for item in processed_data:
            if item[-1] is not compare:
                return False
        return True


def get_decision_tree(loaded_data, layer, parent=None, to_be_split=None):
    if to_be_split is None:
        to_be_split = [0, 1, 2, 3]
    if is_same_cat(loaded_data) is False and layer < 4:
        criterion = select_boundary(loaded_data, to_be_split)
        # pick the max criterion as the bound
        # delete the index of this criterion in [0 1 2 3]
        idx = criterion.index(max(criterion))
        list1 = []
        list2 = []
        for item in loaded_data:
            if item[to_be_split[idx]] >= criterion[idx][1]:
                list1.append(item)
            else:
                list2.append(item)
        if bool(list1) and bool(list2) is True:
            '''
            print(
                "Decision layer:{} from:{}  class1:{}  class2:{}  criterion:{}  threshold:{}".format(layer, parent,
                                                                                                     len(list1),
                                                                                                     len(list2),
                                                                                                     to_be_split[idx],
                                                                                                     criterion[idx][1]))
                                                                                                     '''
            saving.append([layer, parent, to_be_split[idx], criterion[idx][1], None, None])
            to_be_split.pop(idx)
            temp1 = []
            temp2 = []
            for item in to_be_split:
                temp1.append(item)
                temp2.append(item)
            get_decision_tree(list1, layer + 1, parent='class1', to_be_split=temp1)
            get_decision_tree(list2, layer + 1, parent='class2', to_be_split=temp2)
            return
        else:
            """
            select the most label as the category
            """
            cat = []
            times = []
            for item in loaded_data:
                if item[-1] in cat:
                    times[cat.index(item[-1])] = times[cat.index(item[-1])] + 1
                else:
                    cat.append(item[-1])
                    times.append(1)
            position = times.index(max(times))
            # print("In Decision layer:{} class:{}  category:{}".format(layer - 1, parent, cat[position]))
            for item in saving:
                if item[0] == layer - 1:
                    if parent is 'class1':
                        item[4] = cat[position]
                    elif parent is 'class2':
                        item[5] = cat[position]
            return

    else:
        if layer >= 4:
            pass
        else:
            # print("In Decision layer:{} class:{}  category:{}".format(layer - 1, parent, loaded_data[0][-1]))
            for item in saving:
                if item[0] == layer - 1:
                    if parent is 'class1':
                        item[4] = loaded_data[0][-1]
                    elif parent is 'class2':
                        item[5] = loaded_data[0][-1]
        return


def model(one_data, critic):
    i = 0
    path = [None]
    label = None
    for item in critic:
        if item[0] == i and path[i] == item[1]:
            if one_data[item[2]] >= item[3]:
                path.append('class1')
                if item[4] is not None:
                    label = item[4]
            else:
                path.append('class2')
                if item[5] is not None:
                    label = item[5]
            i = i + 1
    if label is one_data[-1]:
        return 1
    else:
        return 0


def validation(test_data, critic):
    acc = 0
    for item in test_data:
        acc = acc + model(item, critic)
    return acc / len(test_data) * 100


if __name__ == '__main__':
    data = data_load(filename='iris.csv')
    train = []
    for i in range(100):
        x = random.randint(0, len(data) - 1)
        train.append(data[x])
        data.pop(x)
    get_decision_tree(train, layer=0)
    print(saving)
    print('Test accuracy is {}%'.format(validation(data, critic=saving)))
