import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class PolarisSeg(nn.Module):
    def __init__(self):
        super(PolarisSeg, self).__init__()
        
        self.encoder = timm.create_model('cspdarknet53', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))

        ch = self.encoder.feature_info.channels()
        
        self.cbam3 = CBAM(ch[2])
        self.cbam2 = CBAM(ch[1])
        self.cbam1 = CBAM(ch[0]) 
        
        self.upconv3 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(ch[2] + ch[2], ch[2]) 
        
        self.upconv2 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(ch[1] + ch[1], ch[1]) 
        
        self.upconv1 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(ch[0] + ch[0], ch[0]) 
        
        self.upconv0 = nn.ConvTranspose2d(ch[0], 64, kernel_size=2, stride=2)
        self.dec0 = DecoderBlock(64, 64)
        
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        features = self.encoder(x)
        s1 = features[0] 
        s2 = features[1] 
        s3 = features[2] 
        bottom = features[3] 
        
        # --- CBAM ---
        s3_filtered = self.cbam3(s3)
        s2_filtered = self.cbam2(s2)
        s1_filtered = self.cbam1(s1)
        
        # --- DECODER ---
        d3 = self.upconv3(bottom)
        d3 = torch.cat([d3, s3_filtered], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, s2_filtered], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, s1_filtered], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.upconv0(d1)
        d0 = self.dec0(d0)
        
        logits = self.out_conv(d0)
        return logits


class PolarisMultimodal(nn.Module):
    def __init__(self, num_classes=7, meta_features=4):
        super(PolarisMultimodal, self).__init__()
        
        self.vision_encoder = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained=True, num_classes=0)
        vision_out_dim = self.vision_encoder.num_features 
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_out_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.clinical_mlp = nn.Sequential(
            nn.Linear(meta_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.attention_gate = nn.Sequential(
            nn.Linear(512 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, metadata):
        v_feat = self.vision_proj(self.vision_encoder(img))
        c_feat = self.clinical_mlp(metadata)
        
        concat_feat = torch.cat((v_feat, c_feat), dim=1) 
        gate_scores = self.attention_gate(concat_feat)
        
        weight_vision = gate_scores[:, 0].unsqueeze(1) 
        weight_clinical = gate_scores[:, 1].unsqueeze(1)
        
        v_feat_attended = v_feat * weight_vision
        c_feat_attended = c_feat * weight_clinical
        
        final_features = torch.cat((v_feat_attended, c_feat_attended), dim=1)
        logits = self.classifier(final_features)
        return logits