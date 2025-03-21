import torch

# 计算axis-angle向量的余弦相似度
def cosine_similarity(a, b):
    norm_a = torch.norm(a, dim=2) + 1e-3
    norm_b = torch.norm(b, dim=2) + 1e-3
    dot_product = torch.sum(a * b, dim=2)
    return dot_product / (norm_a * norm_b)

def vector_distance(a, b):
    diff = a - b
    return torch.norm(diff, dim=2)  # [batch_size, n_joints]

# 计算每个参考样本与输入样本之间的平均相似度
def find_most_similar(input, refs, sim_type='axis-angle'):
    # input: [1, 4, 3]
    # refs: [100, 4, 3]
    # 扩展input为 [100, 4, 3]
    input_expanded = input.expand(refs.size(0), -1, -1)
    
    # 计算每个参考样本和输入样本的关节相似度
    if sim_type == 'euclidean':
        l2_loss = vector_distance(input_expanded, refs)
        avg_l2_loss = l2_loss.mean(dim=1)  # [100]
        return torch.topk(avg_l2_loss, 20, largest=False), avg_l2_loss
        # return torch.argmin(avg_l2_loss), avg_l2_loss
    else:
        sim = cosine_similarity(input_expanded, refs)  # [100, 4]
        avg_sim = sim.mean(dim=1)  # [100]
        return torch.argmax(avg_sim), avg_sim

# # 测试数据
# input = torch.randn(1, 4, 3)  # 输入样本，形状 [1, 4, 3]
# refs = torch.randn(100, 4, 3)  # 参考样本，形状 [100, 4, 3]

# # 计算相似度
# index, avg_sim = find_most_similar(input, refs)

# # 输出最相似的样本索引和相似度
# print("最相似的样本索引是：", index.item())
# print("每个参考样本的平均相似度：", avg_sim)
