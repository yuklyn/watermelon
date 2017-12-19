class TreeNode(object):
    def __init__(self, data_set=None, target_set=None, feature_index=None, feature_value=None,
                 has_calc_feature_indexes=None, remain_feature_indexes=None):
        self.data_set = data_set
        self.target_set = target_set
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.has_calc_feature_indexes = has_calc_feature_indexes
        self.remain_feature_indexes = remain_feature_indexes

        self.classification = None  # 保存的是针对当前分支的结果，有值则表示该点是叶子节点
        self.selected_feature_index = None
        self.threshold = None
        self.children = []
        self.entropy = None

        print('此节点的数据集大小：', len(self.data_set[0]))


class ID3Tree:
    def __init__(self):
        self.data_set = []
        self.target_set = []
        self.feature_names = []
        self.target_names = []
        self.tree_root = None

    def create_tree(self, data_set, target_set, feature_names, target_names):
        self.data_set = data_set
        self.target_set = target_set
        self.feature_names = feature_names
        self.target_names = target_names
        print('target_names：', target_names)

        self.tree_root = TreeNode(data_set, target_set, -1, -1, [],
                                  list(i for i in range(len(self.data_set))))
        self.tree_generate(self.tree_root)

    def tree_generate(self, node: TreeNode):
        if ID3Tree.is_same_classification(node.target_set):
            ID3Tree.label_as_leaf(node)
            return node
        if not node.remain_feature_indexes or ID3Tree.is_all_same_feature_values(node.data_set):
            ID3Tree.label_as_leaf(node)
            return node
        max_info_gain_feature_index, split_value = self.get_max_info_gain_feature(node)
        print('最大增益属性的索引：', max_info_gain_feature_index)
        node.classification = None  # 保存的是针对当前分支的结果，有值则表示该点是叶子节点
        node.selected_feature_index = max_info_gain_feature_index
        node.threshold = split_value
        ID3Tree.generate_child_node(node)
        node.entropy = ID3Tree.compute_entropy(node.target_set)
        for child_node in node.children:
            # 用不用检查是否为[]？
            if not child_node.data_set:
                ID3Tree.label_as_leaf(node)
                return node
            self.tree_generate(child_node)
        node.data_set = None
        node.target_set = None
        node.has_calc_feature_indexes = None
        node.remain_feature_indexes = None
        return node

    @staticmethod
    def is_same_classification(target_set):
        return all(target == target_set[0] for target in target_set)

    @staticmethod
    def label_as_leaf(node: TreeNode):
        node.classification = ID3Tree.get_leaf_classification(node.target_set)

    @staticmethod
    def get_leaf_classification(target_set):
        from collections import Counter
        return Counter(target_set)

    @staticmethod
    def is_all_same_feature_values(data_set):
        return all(data == data_set[0] for data in data_set)

    def get_max_info_gain_feature(self, node: TreeNode):
        max_gain_feature_index = -1
        max_gain = -1000000
        max_gain_split_value = None
        for i in range(len(node.remain_feature_indexes)):
            print('node.remain_feature_indexes[i]', node.remain_feature_indexes[i])
            print('node.data_set[node.remain_feature_indexes[i]][0]', node.data_set[node.remain_feature_indexes[i]][0])
            if ID3Tree.is_continuous(node.data_set[node.remain_feature_indexes[i]][0]):
                print('remain_feature_indexes', node.remain_feature_indexes)
                feature_values = node.data_set[node.remain_feature_indexes[i]]
                split_values = sorted(set(list(float(value) for value in feature_values)))
                for j in range(len(split_values) - 1):
                    split_values[j] = (split_values[j] + split_values[j + 1]) * 0.5
                del split_values[-1]

                temp_max_split_gain = -10000
                temp_max_split_gain_value = 0
                for split_value in split_values:
                    temp_gain = ID3Tree.compute_info_gain_continuous_feature(node.data_set, node.target_set,
                                                                             node.remain_feature_indexes[i],
                                                                             split_value)
                    if temp_gain > temp_max_split_gain:
                        temp_max_split_gain = temp_gain
                        temp_max_split_gain_value = split_value

                    if temp_gain > max_gain:
                        max_gain = temp_gain
                        max_gain_feature_index = node.remain_feature_indexes[i]
                        max_gain_split_value = split_value
                print('列：', node.remain_feature_indexes[i], ' 特征属性：', self.feature_names[i], ' 连续值: ',
                      temp_max_split_gain_value, " 增益：",
                      temp_max_split_gain)
            else:
                temp_gain = ID3Tree.compute_info_gain(node.data_set, node.target_set, node.remain_feature_indexes[i])
                print('列：', node.remain_feature_indexes[i], ' 特征属性：', self.feature_names[i], '离散值：', -1, " 增益：",
                      temp_gain)
                if temp_gain > max_gain:
                    max_gain = temp_gain
                    max_gain_feature_index = node.remain_feature_indexes[i]

        return max_gain_feature_index, max_gain_split_value

    @staticmethod
    def is_continuous(feature_value):
        return all(value in "0123456789.-" for value in feature_value)

    @staticmethod
    def compute_info_gain_continuous_feature(data_set, target_set: list, feature_index, feature_split_value):
        ent = ID3Tree.compute_entropy(target_set)
        feature_values = data_set[feature_index]
        left_child_indexes = []
        right_child_indexes = []
        for i in range(len(feature_values)):
            feature_value = feature_values[i]
            if float(feature_value) <= feature_split_value:
                left_child_indexes.append(i)
            else:
                right_child_indexes.append(i)
        left_child_target_set = list(
            target_set[i] for i in range(len(target_set)) if i in left_child_indexes)
        right_child_target_set = list(
            target_set[i] for i in range(len(target_set)) if i in right_child_indexes)
        ent_left_child = (len(left_child_target_set) / float(len(target_set))) * ID3Tree.compute_entropy(
            left_child_target_set)
        ent_right_child = (len(right_child_target_set) / float(len(target_set))) * ID3Tree.compute_entropy(
            right_child_target_set)

        return ent - ent_left_child - ent_right_child

    @staticmethod
    def compute_info_gain(data_set, target_set: list, feature_index):
        ent = ID3Tree.compute_entropy(target_set)
        feature_values = data_set[feature_index]
        feature_dict = {}
        for i in range(len(feature_values)):
            value = feature_values[i]
            if value not in feature_dict:
                feature_dict[value] = []
            feature_dict[value].append(i)
        ent_children = 0
        for key in feature_dict:
            indexes = feature_dict[key]
            child_target_set = list(target_set[i] for i in range(len(target_set)) if i in indexes)
            ent_children -= (len(child_target_set) / float(len(target_set))) * ID3Tree.compute_entropy(
                child_target_set)
        return ent + ent_children

    @staticmethod
    def compute_entropy(target_set):
        import math as math
        target_dict = dict((i, target_set.count(i)) for i in set(target_set))
        ent = 0
        for key in target_dict:
            pk = target_dict[key] / float(len(target_set))
            ent -= pk * math.log2(pk)
        return ent

    @staticmethod
    def generate_child_node(node: TreeNode):
        if node.threshold:
            feature_values = node.data_set[node.selected_feature_index]
            left_child_indexes = []
            right_child_indexes = []
            for i in range(len(feature_values)):
                feature_value = feature_values[i]
                if float(feature_value) <= node.threshold:
                    left_child_indexes.append(i)
                else:
                    right_child_indexes.append(i)

            left_child_data_set = [[] for col in range(len(node.data_set))]
            right_child_data_set = [[] for col in range(len(node.data_set))]
            for i in range(len(node.data_set)):
                col_data = node.data_set[i]
                for j in range(len(col_data)):
                    if j in left_child_indexes:
                        left_child_data_set[i].append(col_data[j])
                    else:
                        right_child_data_set[i].append(col_data[j])

            left_child_target_set = list(
                node.target_set[i] for i in range(len(node.target_set)) if i in left_child_indexes)
            right_child_target_set = list(
                node.target_set[i] for i in range(len(node.target_set)) if i in right_child_indexes)
            if not node.has_calc_feature_indexes:
                node.has_calc_feature_indexes = []
            has_calc_feature_indexes = list(node.has_calc_feature_indexes).append(node.selected_feature_index)
            list(node.remain_feature_indexes).remove(node.selected_feature_index)
            left_node = TreeNode(left_child_data_set, left_child_target_set, node.selected_feature_index,
                                 has_calc_feature_indexes=has_calc_feature_indexes,
                                 remain_feature_indexes=node.remain_feature_indexes)
            right_node = TreeNode(right_child_data_set, right_child_target_set, node.selected_feature_index,
                                  has_calc_feature_indexes=has_calc_feature_indexes,
                                  remain_feature_indexes=node.remain_feature_indexes)
            node.children = [left_node, right_node]
        else:
            feature_values = node.data_set[node.selected_feature_index]
            feature_dict = {}
            for i in range(len(feature_values)):
                value = feature_values[i]
                if value not in feature_dict:
                    feature_dict[value] = []
                feature_dict[value].append(i)
            node.children = []
            for key in feature_dict:
                indexes = feature_dict[key]
                child_data_set = [[] for col in range(len(node.data_set))]
                for i in range(len(node.data_set)):
                    col_data = node.data_set[i]
                    for j in range(len(col_data)):
                        if j in indexes:
                            child_data_set[i].append(col_data[j])
                child_target_set = list(node.target_set[i] for i in range(len(node.target_set)) if i in indexes)
                if not node.has_calc_feature_indexes:
                    node.has_calc_feature_indexes = []
                has_calc_feature_indexes = list(node.has_calc_feature_indexes).append(node.selected_feature_index)
                if node.selected_feature_index in node.remain_feature_indexes:
                    del node.remain_feature_indexes[
                        list(node.remain_feature_indexes).index(node.selected_feature_index)]
                child_node = TreeNode(child_data_set, child_target_set, node.selected_feature_index, key,
                                      has_calc_feature_indexes=has_calc_feature_indexes,
                                      remain_feature_indexes=node.remain_feature_indexes)
                node.children.append(child_node)
