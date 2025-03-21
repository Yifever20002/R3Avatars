import torch
from sklearn.decomposition import PCA

def apply_PCA(j_embedding):

    j_embedding_np = j_embedding.numpy()
    pca = PCA(n_components=1)
    j_embedding_pca_np = pca.fit_transform(j_embedding_np)
    j_embedding_pca = torch.from_numpy(j_embedding_pca_np)
    # print("降维后的数据形状:", j_embedding_pca.shape)  # 输出: torch.Size([100, 1])
    # print(j_embedding_pca)
    # assert False
    return j_embedding_pca, pca