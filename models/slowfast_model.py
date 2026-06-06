import torch
import torch.nn as nn

class SuspiciousActivityModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SuspiciousActivityModel, self).__init__()

        # FIX: Pull the official pretrained model directly from Meta's PyTorchVideo hub
        # This will automatically download the Kinertics-400 pretrained weights (~240MB)
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        # Keep the head mutation logic—this part was actually correct!
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Expects x to be a split list: [slow_pathway, fast_pathway]
        """
        return self.model(x)