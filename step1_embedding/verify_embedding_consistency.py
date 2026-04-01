import os
import torch


def main():
    base_dir = os.path.dirname(__file__)
    embed_path = os.path.join(base_dir, "libero_object_embeded.pt")
    per_instr_path = os.path.join(base_dir, "libero_object_per_instruction_clusters.pt")

    print(f"Loading original embeddings from: {embed_path}")
    data_orig = torch.load(embed_path, map_location="cpu")
    embeds_orig = data_orig["embeds"]          # [N, D]
    instr_orig = data_orig["instructions"]     # List[str]

    print(f"Loading per-instruction clusters from: {per_instr_path}")
    data_cluster = torch.load(per_instr_path, map_location="cpu")
    embeds_cluster = data_cluster["embeds"]    # [N, D]
    instr_cluster = data_cluster["instructions"]

    # 1) 检查形状是否一致
    print("\n[Check] shapes")
    print(f"  embeds_orig shape   = {embeds_orig.shape}")
    print(f"  embeds_cluster shape= {embeds_cluster.shape}")
    print(f"  len(instr_orig)     = {len(instr_orig)}")
    print(f"  len(instr_cluster)  = {len(instr_cluster)}")

    ok_shape = (
        embeds_orig.shape == embeds_cluster.shape
        and len(instr_orig) == len(instr_cluster)
        and embeds_orig.shape[0] == len(instr_orig)
    )
    print(f"  -> shape match: {ok_shape}")

    # 2) 检查 instructions 是否完全一致（逐条字符串比较）
    print("\n[Check] instructions equality")
    same_len = len(instr_orig) == len(instr_cluster)
    same_instr = same_len and all(a == b for a, b in zip(instr_orig, instr_cluster))
    print(f"  -> instructions identical: {same_instr}")

    if not same_instr:
        # 打印前若干条不一致的例子
        print("  Mismatched instructions (first 10):")
        for i, (a, b) in enumerate(zip(instr_orig, instr_cluster)):
            if a != b:
                print(f"    idx {i}: orig='{a}' | cluster='{b}'")
            if i >= 9:
                break

    # 3) 检查 embeds 是否逐元素完全一致
    print("\n[Check] embedding tensor equality")
    # 使用 allclose 允许非常小的浮点误差；如果你要求 bit-level 一致，可以再加 equal 检查
    same_allclose = torch.allclose(embeds_orig, embeds_cluster, atol=0.0, rtol=0.0)
    same_equal = torch.equal(embeds_orig, embeds_cluster)
    print(f"  -> torch.allclose (rtol=0, atol=0): {same_allclose}")
    print(f"  -> torch.equal (bit-level):        {same_equal}")

    if not same_allclose:
        # 找出前几个有差异的元素
        print("  Found differences in embeddings; showing first few positions:")
        diff = (embeds_orig != embeds_cluster)
        # 展平后取前几个索引
        diff_idx = diff.nonzero(as_tuple=False)
        for i in range(min(10, diff_idx.shape[0])):
            idx = diff_idx[i]
            n, d = idx.tolist()
            v1 = embeds_orig[n, d].item()
            v2 = embeds_cluster[n, d].item()
            print(f"    idx (sample={n}, dim={d}): orig={v1}, cluster={v2}")

    # 汇总
    print("\n[Summary]")
    if ok_shape and same_instr and same_allclose and same_equal:
        print("All checks passed: order and raw embedding values are completely identical.")
    else:
        print("There are mismatches. See logs above for details.")


if __name__ == "__main__":
    main()
