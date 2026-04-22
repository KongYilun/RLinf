import torch
dict = torch.load('libero_10_per_instruction_centers.pt')
new_dict={}
for i,name in enumerate(dict):
    print(i,name)
    new_dict[name]=torch.tensor(i)
torch.save(new_dict,'libero_10_instruction_to_task_id_map.pt')