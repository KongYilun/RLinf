import torch
from tqdm import tqdm  # 用于显示进度条，如果没有请 pip install tqdm

def process_and_check_pt(input_path, output_path):
    print(f"正在读取文件: {input_path} ...")
    # 1. 读取数据 (建议映射到 CPU 防止显存溢出)
    data = torch.load(input_path, map_location='cpu')
    
    # 确保所有需要的 key 都存在
    required_keys = ['labels', 'instructions', 'embeds', 'cluster_centers', 'episode_ids']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"输入文件缺少必要的 key: {key}")

    # 获取数据长度
    num_samples = len(data['instructions'])
    print(f"数据总条目数: {num_samples}")

    # 2. 初始化容器
    # 结果字典: key=instruction, value=set(cluster_center_tuple) -> 用集合自动去重
    result_map = {} 
    
    # 检查字典: key=instruction, value={ label: cluster_center_tensor }
    # 用于检查同一指令下，同一 label 是否对应同一 center
    consistency_check_map = {}
    
    mismatch_count = 0

    # 3. 遍历数据
    for i in tqdm(range(num_samples), desc="处理中"):
        # 提取当前条目的数据
        instr = data['instructions'][i]
        
        # 处理 label：如果是 tensor scalar，转为 python 数值，方便作为 dict key
        label = data['labels'][i]
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        center = data['cluster_centers'][i] # 这是一个 Tensor

        # --- A. 初始化层级 ---
        if instr not in result_map:
            result_map[instr] = set()
            consistency_check_map[instr] = {}

        # --- B. 收集去重 (Requirement 1 & 2) ---
        # Tensor 不能直接放入 set，需转为 tuple (假设 center 是 1D 向量)
        # 如果 center 是多维的，可以用 tuple(center.flatten().tolist())
        center_tuple = tuple(center.tolist())
        result_map[instr].add(center_tuple)

        # --- C. 一致性检查 (Requirement 3) ---
        # 逻辑：同一 instruction 下，同一 label 必须对应相同的 cluster_center
        if label in consistency_check_map[instr]:
            existing_center = consistency_check_map[instr][label]
            
            # 使用 torch.equal 比较两个 tensor 是否相同
            if not torch.equal(existing_center, center):
                mismatch_count += 1
                # 仅打印前5个错误，防止刷屏
                if mismatch_count <= 5:
                    print(f"\n[一致性警告] Instruction: '{instr}'")
                    print(f"Label: {label}")
                    print(f"冲突: 之前记录的 center 与当前 center 不一致!")
        else:
            # 记录该 label 对应的 center
            consistency_check_map[instr][label] = center

    # 4. 格式化输出数据
    print("正在构建输出数据...")
    final_output = {}
    
    for instr, center_tuples_set in result_map.items():
        # 将 tuple 转回 Tensor List
        # 注意：这里会丢失原始 Tensor 的 device 信息，默认存为 CPU tensor
        center_list = [torch.tensor(c) for c in center_tuples_set]
        final_output[instr] = center_list

    # 5. 保存
    print(f"正在保存到: {output_path} ...")
    torch.save(final_output, output_path)
    
    # 6. 总结
    print("-" * 30)
    print("处理完成!")
    print(f"包含的唯一 Instructions 数量: {len(final_output)}")
    if mismatch_count > 0:
        print(f"⚠️ 发现 {mismatch_count} 个 Label-Center 不一致的情况！请检查原始数据。")
    else:
        print("✅ 一致性检查通过：所有相同 Instruction 下的相同 Label 均对应唯一的 Center。")

# ==========================================
# 测试代码 (生成伪数据验证逻辑)
# ==========================================
if __name__ == "__main__":
    # 生成一个伪造的 .pt 文件用于测试
    dummy_path = "libero_10_per_instruction_clusters_proj_all.pt"
    save_path = "libero_10_per_instruction_centers.pt"
    
    # print("生成测试数据...")
    # dummy_data = {
    #     'instructions': [
    #         "open the door", "open the door", "open the door", # 组1
    #         "pick up apple", "pick up apple"                   # 组2
    #     ],
    #     'labels': [
    #         1, 1, 2,  # label 1 应该对应相同的 center
    #         5, 5
    #     ],
    #     'cluster_centers': [
    #         torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.5]), # 第1、2个完全相同
    #         torch.tensor([0.9, 0.9]), torch.tensor([0.8, 0.8])  # ! 故意制造冲突：同指令同label(5)，但center不同
    #     ],
    #     'embeds': [torch.randn(10)] * 5,    # 占位
    #     'episode_ids': [0, 1, 2, 3, 4]      # 占位
    # }
    # torch.save(dummy_data, dummy_path)
    
    # 运行处理函数
    process_and_check_pt(dummy_path, save_path)
    
    # 验证结果
    res = torch.load(save_path)
    print("\n输出文件内容概览:")
    for k, v in res.items():
        print(f"Instruction: {k}, Centers Count: {len(v)}")