import torch
import torch.nn as nn

class ECTNet(nn.Module):
    def __init__(self, input_shape):
        super(ECTNet, self).__init__()
        self.input_height, self.input_width = input_shape

        self.features = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        ])

        h = self.input_height // 4
        w = self.input_width // 4
        self.fc_input_dim = 64 * h * w 

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.classifier(x.view(x.size(0), -1))