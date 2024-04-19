import json
import os
import statistics
file_path = './HongHu/20/True/ss0.3'
json_files = []
max_oa = []
for file_name in os.listdir(file_path):
    if file_name.endswith('.json'):
        json_files.append(file_name)
# print(json_files)
for file_name in json_files:
    data_path = os.path.join(file_path, file_name)
    with open(data_path, 'r') as file:
        data = json.load(file)
        oa = max(data['train_oa']['value'])
        max_oa.append(oa)
mean = sum(max_oa[:5])/len(max_oa[:5])
print(max_oa[:5])
print(mean)
print(statistics.variance(max_oa[:5]))
# print(max(oa))
