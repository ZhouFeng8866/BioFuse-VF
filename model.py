import torch
import torch.nn as nn


class BioFuseVF(nn.Module):
    """
    BioFuse-VF model.

    Inputs:
        pro:     ProtT5 embedding, shape = (B, 1, 1024)
        aaindex: PCA-reduced AAindex feature, shape = (B, 1, 12)

    Output:
        logits:  shape = (B, num_classes)
    """

    def __init__(self, num_classes=7):
        super(BioFuseVF, self).__init__()
        self.num_classes = num_classes

        # ProtT5 semantic branch
        self.pro_cnn1 = nn.Sequential(
            nn.Conv1d(1024, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.pro_cnn2 = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512)
        )

        self.pro_cnn3 = nn.Sequential(
            nn.Conv1d(1024, 512, 3, padding=1),
            nn.BatchNorm1d(512)
        )

        self.pro_cnn4 = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.pro_cnn5 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256)
        )

        self.pro_cnn6 = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
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

        self.act = nn.ReLU()

        # Fusion classifier
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, pro, aaindex):
        # pro: (B, 1, 1024) -> (B, 1024, 1)
        pro_c = pro.permute(0, 2, 1)

        # Residual CNN block 1
        pro1 = self.pro_cnn1(pro_c)
        pro1 = self.pro_cnn2(pro1)
        pro2 = self.pro_cnn3(pro_c)
        pro_out1 = self.act(pro1 + pro2)

        # Residual CNN block 2
        pro3 = self.pro_cnn4(pro_out1)
        pro3 = self.pro_cnn5(pro3)
        pro4 = self.pro_cnn6(pro_out1)
        pro_out2 = self.act(pro3 + pro4).permute(0, 2, 1)

        # AAindex branch
        aaindex_out = self.aaindex_branch(aaindex)

        # Feature fusion
        fused = torch.cat([pro_out2, aaindex_out], dim=-1)

        # Classification
        x = self.fc1(fused)
        logits = self.fc2(x)

        logits = logits.squeeze(1)

        return logits
