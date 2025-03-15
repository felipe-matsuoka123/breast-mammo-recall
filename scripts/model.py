import timm
import torch.nn as nn

class MammoModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model( 'efficientnet_b0', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
    
    def state_dict(self):
        return super().state_dict()
    
class MammoModelConvNext(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
    
    def state_dict(self):
        return super().state_dict()
    
class ConvnextMammoModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model('convit_base', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    


class MaxPoolingMammoModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.backbone.global_pool = nn.AdaptiveMaxPool2d(1)

        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.flatten(1)
        return self.classifier(features)