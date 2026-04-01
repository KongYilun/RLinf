import torch 
data=torch.load("/data/users/kongyilun/code/RLinf/step1_embedding/libero_object_per_instruction_clusters_proj_all.pt")
grouped_data = {}
print("data keys:", data.keys())
print(data['embeds'].shape)
for i, inst in enumerate(data['instructions']):
    if inst not in grouped_data:
        grouped_data[inst] = {
            'labels': [],
            'indices': []
        }
    grouped_data[inst]['labels'].append(data['labels'][i])
    grouped_data[inst]['indices'].append(i)

# 2. 输出结果
for inst, content in grouped_data.items():
    print(f"\nInstruction: {inst}")
    print(f"Labels: {content['labels']}")
    
    # 如果 cluster_centers 是与每个样本对应的：
    if len(data['cluster_centers']) == len(data['instructions']):
        for idx in content['indices']:
            print(data['cluster_centers'][idx][:5])
        # relevant_centers = [data['cluster_centers'][idx][:5] for idx in content['indices']]
        # print(f"Cluster Centers: {relevant_centers}")
    else:
        # 如果 cluster_centers 是全局的，通常直接输出
        print(f"Global Cluster Centers: {data['cluster_centers']}")