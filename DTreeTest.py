import ID3Tree

# 二维列表，data[1]代表训练集中所有样本第二个特征属性的值的列表
data_set = []
target_set = []
feature_names = []
target_names = []

filename = "D:\\yuklyn\python\PycharmProjects\watermelon\西瓜数据集3.0.txt"
file = open(filename, 'r', encoding='UTF-8')
has_order = True
line_count = 0
while 1:
    line = file.readline()
    line = line.strip('\n')
    line_data = line.split(',')
    if not line:
        break
    if has_order:
        del line_data[0]
    if line_count == 0:
        feature_names = list(data for data in line_data if data != line_data[-1])
        target_names.append(line_data[-1])
        data_set = [[] for col in range(len(line_data) - 1)]
        target_set = []
    else:
        for i in range(len(line_data) - 1):
            data_set[i].append(line_data[i])
        target_set.append(line_data[-1])
    line_count = line_count + 1

target_names = list(set(target_set))
print('feature_names：', feature_names)
print('data_set:', data_set)
print('target_set:', target_set)

tree = ID3Tree.ID3Tree()
tree.create_tree(data_set, target_set, feature_names, target_names)

classifications = []
test_set = [[] for row in range(len(data_set[0]))]
for col in range(len(data_set)):
    col_data = data_set[col]
    for row in range(len(test_set)):
        test_set[row].append(col_data[row])
for test_data in test_set:
    classification = tree.predict(test_data)
    classifications.append(classification)
print('classifications:', classifications)

