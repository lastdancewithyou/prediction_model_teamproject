import torch.nn as nn

class Classifier(nn.Module):
    """
    Location과 Pitch Type에 대한 Classification
    """
    def __init__(self, output_size):
        super(Classifier, self).__init__()
        
        self.output_size = output_size
        self.classifier = nn.LazyLinear(self.output_size)
        
    def forward(self, x):
        return self.classifier(x)