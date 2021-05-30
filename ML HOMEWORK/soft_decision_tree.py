import torch.nn as nn
import torch


class Decision_node:
    def __init__(self, l_child='None', r_child='None', parent='None', value='None'):
        self.l_child = l_child
        self.r_child = r_child
        self.parent = parent
        self.Fm = value
        self.wm = torch.rand((1, 4), requires_grad=True)

    def get_l(self):
        return self.l_child

    def get_r(self):
        return self.r_child

    def get_parent(self):
        return self.parent

    def get_value(self):
        return self.Fm


def get_loss_func(task_type):
    if task_type is 'classification':
        return nn.CrossEntropyLoss()
    elif task_type is 'regression':
        return nn.MSELoss()


def Fm(node, x):
    """
    if node is leaf node:
        return the_value_of_node
    else:
        gm = sigmoid(Linear(input_vector))
        return gm*FL+(1-gm)*FR
    FL和FR是从该节点下一层传上来的Fm
    :param x:
    :param node: Decision_node 类
    :return: Fm值
    """
    if node.get_l is None and node.get_r is None:
        return node.get_value
    else:
        gm = torch.sigmoid(torch.mm(node.wm, torch.tensor(x)))
        return node.get_l.get_value * gm + node.get_r.get_value


loss = get_loss_func(task_type='classification')


