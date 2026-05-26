import torch
import torch.nn as nn


class BioFuseVF(nn.Module):
    """
    BioFuse-VF model for multi-class virulence factor classification.

    Inputs:
        pro:     ProtT5 embedding, shape = (B, 1, 1024)
        aaindex: PCA-reduced AAindex features, shape = (B, 1, 12)

    Output:
        logits:  class prediction logits, shape = (B, num_classes)
    """

    def __init__(self, num_classes=7):
        super(BioFuseVF, self).__init__()
        self.num_classes = num_classes

        # ProtT5 semantic branch: residual CNN block I
        self.pro_cnn1 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.pro_cnn2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512)
        )

        self.pro_cnn3 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512)
        )

        # ProtT5 semantic branch: residual CNN block II
        self.pro_cnn4 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.pro_cnn5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256)
        )

        self.pro_cnn6 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256)
        )

        # AAindex physicochemical branch
        self.aaindex_branch = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Attention-based feature weighting
        self.attention = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.relu = nn.ReLU()

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, pro, aaindex):
        # ProtT5 branch
        # pro: (B, 1, 1024) -> (B, 1024, 1)
        pro = pro.permute(0, 2, 1)

        pro_main_1 = self.pro_cnn1(pro)
        pro_main_1 = self.pro_cnn2(pro_main_1)

        pro_res_1 = self.pro_cnn3(pro)

        pro_out_1 = self.relu(pro_main_1 + pro_res_1)

        pro_main_2 = self.pro_cnn4(pro_out_1)
        pro_main_2 = self.pro_cnn5(pro_main_2)

        pro_res_2 = self.pro_cnn6(pro_out_1)

        pro_out = self.relu(pro_main_2 + pro_res_2)

        # (B, 256, 1) -> (B, 1, 256)
        pro_out = pro_out.permute(0, 2, 1)

        # AAindex branch
        # aaindex: (B, 1, 12) -> (B, 1, 64)
        aaindex_out = self.aaindex_branch(aaindex)

        # Feature concatenation before attention
        combined = torch.cat([pro_out, aaindex_out], dim=-1)

        # Attention weights for two branches
        attn_weights = self.attention(combined).squeeze(1)

        # Remove sequence dimension
        pro_out = pro_out.squeeze(1)
        aaindex_out = aaindex_out.squeeze(1)

        # Apply branch weights
        weighted_pro = pro_out * attn_weights[:, 0].unsqueeze(1)
        weighted_aaindex = aaindex_out * attn_weights[:, 1].unsqueeze(1)

        # Final feature fusion
        fused = torch.cat([weighted_pro, weighted_aaindex], dim=-1)

        # Classification
        logits = self.classifier(fused)

        return logits
