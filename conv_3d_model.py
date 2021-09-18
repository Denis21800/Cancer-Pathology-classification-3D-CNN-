import torch.nn as nn
import torch


class ImageSeq(nn.Module):
    def __init__(self, device, num_classes):
        super(ImageSeq, self).__init__()
        self.device = device
        self.cube_cnn = nn.Sequential(
            nn.Conv3d(3, 16, (7, 9, 9), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.PReLU(),
            nn.MaxPool3d((3, 3, 3)),
            nn.Conv3d(16, 16, (5, 7, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.PReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, (5, 7, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 32, (2, 5, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            nn.Conv3d(32, 64, (2, 5, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.Conv3d(64, 128, (1, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            nn.Conv3d(128, 256, (1, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.PReLU(),
            nn.Conv3d(256, 512, (1, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.PReLU(),
        )

        self.fc_1 = nn.Linear(512 * 2 * 2 * 2, 64)
        self.fc_2 = nn.Linear(64, num_classes)
        self.p_relu = nn.PReLU()

    def forward(self, x):
        stack_seq = torch.stack(x)
        stack_seq = stack_seq.permute(1, 2, 0, 3, 4)
        stack_seq = stack_seq.to(self.device).float()
        out = self.cube_cnn(stack_seq)
        out = out.view(-1, 512 * 2 * 2 * 2)
        out = torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)
        out = self.p_relu(self.fc_1(out))
        out = self.fc_2(out)
        return out


def create_conv_3d(num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ImageSeq(device, num_classes=4)
    return model
