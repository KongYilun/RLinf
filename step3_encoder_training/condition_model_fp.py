import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoProcessor


class SiglipConditionModel(nn.Module):
    """
    输入：SigLIP 视觉 encoder 的像素输入（以及可选的语言输入）。

    - classifier_head: 预测聚类类别 id（与 `labels` 中一致）。
    - residual_head: 预测 residual，维度与 `projected_embeds` / `cluster_centers` 一致。
    """

    def __init__(self, model_path: str, num_classes: int, residual_dim: int):
        super().__init__()

        # 1. 加载预训练的 SigLIP 模型
        self.model_path = model_path
        self.siglip = AutoModel.from_pretrained(model_path)
        self._processor = None

        # 获取视觉 embedding 维度 (hidden_size)
        self.embed_dim = self.siglip.config.vision_config.hidden_size
        self.num_classes = num_classes
        self.residual_dim = residual_dim

        # 2. 基于 embedding 预测分类 C
        # self.classifier_head = nn.Linear(self.embed_dim*2, num_classes)
        hidden_dim=2048
        self.classifier_head = nn.Sequential(
            # 第一层：降维并提取特征
            nn.Linear(self.embed_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二层：进一步非线性变换
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三层：输出分类 logits
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 3. 基于 embedding 和分类结果 C 预测 residual
        #    residual 的维度与 projected_embeds / cluster_centers 相同（例如 4096）
        self.residual_head = nn.Linear(self.embed_dim*2 + num_classes, residual_dim)

    def _get_processor(self) -> AutoProcessor:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        return self._processor

    def forward(self, images, instructions, **kwargs):
        """
        images: List[PIL.Image] 或 ndarray(uint8, HxWx3) 的 batch
        instructions: List[str]，与 images 对齐
        """
        # import ipdb; ipdb.set_trace()
        processor = self._get_processor()

        proc = processor(images=images, text=instructions, padding="max_length", return_tensors="pt")
        device = next(self.siglip.parameters()).device
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        outputs = self.siglip(**proc, **kwargs)

        # 通常 SigLIP 提供 pooler_output 或类似全局 embedding
        embedding = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)  # [B, embed_dim]

        # 分类 logits
        logits_c = self.classifier_head(embedding)  # [B, num_classes]

        # 拼接 embedding 与分类结果后回归 residual
        combined_input = torch.cat((embedding, logits_c), dim=1)  # [B, embed_dim + num_classes]
        residual_embedding = self.residual_head(combined_input)   # [B, residual_dim]

        return {
            "embedding": embedding,
            "logits_c": logits_c,
            "residual_embedding": residual_embedding,
        }


__all__ = ["SiglipConditionModel"]