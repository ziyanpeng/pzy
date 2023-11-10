# import segmentation_models_pytorch as smp
# import torch
# import torch.nn as nn
#
# def get_backbone(encoder_name, pretrain_model_path):
#     backbone = smp.Unet(
#         encoder_name=encoder_name,
#         encoder_weights=None,
#         in_channels=3
#     ).encoder
#     backbone.load_state_dict(torch.load(pretrain_model_path))
#     return backbone
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs
#
# class Unet(nn.Module):
#     def __init__(self, encoder, num_classes, pretrained_model_path):
#         super(Unet, self).__init__()
#         self.backbone = get_backbone(encoder, pretrained_model_path)
#         in_filters = [192, 256, 512, 1536]
#         out_filters = [32, 64, 128, 256]
#
#         # in_filters = [224, 352, 704, 1152]       #swin_T
#         # out_filters = [64, 128, 256,512]
#
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])
#
#         self.up_conv = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             #allconv3335(out_filters[0],out_filters[0])
#             nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)
#
#     def forward(self, inputs):
#         [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)[1:]
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         up1 = self.up_conv(up1)
#
#         final = self.final(up1)
#
#         return final
#
#     def freeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = True
#
#
# if __name__ == '__main__':
#     model = Unet(num_classes=2, )
#     x = torch.randn(4, 3, 256, 256)
#     output = model(x)
#     print(output.size())


import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class PLKC5310(nn.Module):                 #
    def __init__(self, in_planes,expands=2):
        super(PLKC5310, self).__init__()
        # print(in_planes)
        self.conv0=nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.convres3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )

    def forward(self, x):
        out1 = F.relu(self.conv0(x), inplace=True)
        out2 = F.relu(self.conv1(x), inplace=True)
        out3 = F.relu(self.conv5(x) + self.convres3(x), inplace=True)
        out4 = F.relu(self.conv3(x), inplace=True)
        out = out1 + out2 + out3 + out4
        return out

class PLKC7531(nn.Module):
    def __init__(self, in_planes,expands=2):
        super(PLKC7531, self).__init__()
        # print(in_planes)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=7, padding=3,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.convres3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )

    def forward(self, x):
        out1 = F.relu(self.conv1(x), inplace=True)
        out2 = F.relu(self.conv7(x) + self.convres3(x), inplace=True)
        out3 = F.relu(self.conv5(x) + self.convres3(x), inplace=True)
        out4 = F.relu(self.conv3(x), inplace=True)
        out = out1 + out2 + out3 + out4
        return out
class PLKC9753(nn.Module):
    def __init__(self, in_planes,expands=2):
        super(PLKC9753, self).__init__()
        # print(in_planes)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=7, padding=3,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.convres3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=9, padding=4,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )


    def forward(self, x):
        out1 = F.relu(self.conv3(x), inplace=True)
        out2 = F.relu(self.conv7(x) + self.convres3(x), inplace=True)
        out3 = F.relu(self.conv5(x) + self.convres3(x), inplace=True)
        out4 = F.relu(self.conv7(x), inplace=True)
        out = out1 + out2 + out3 + out4
        return out


class PLKC11975(nn.Module):
    def __init__(self, in_planes,expands=2):
        super(PLKC11975, self).__init__()
        # print(in_planes)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=7, padding=3,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=9, padding=4,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=11, padding=5,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.convres3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )

    def forward(self, x):
        out1 = F.relu(self.convres3(x), inplace=True)
        out2 = F.relu(self.conv9(x) + self.convres3(x), inplace=True)
        out3 = F.relu(self.conv7(x) + self.convres3(x), inplace=True)
        out4 = F.relu(self.conv5(x)+ self.convres3(x), inplace=True)
        out = out1 + out2 + out3 + out4
        return out

class PLKC131197(nn.Module):
    def __init__(self, in_planes,expands=2):
        super(PLKC131197, self).__init__()
        # print(in_planes)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=9, padding=4,bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=11, padding=5,bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=13, padding=6,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )
        self.convres3=nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // expands, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes // expands, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_planes // expands),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes // expands, out_channels=in_planes, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_planes),
        )

    def forward(self, x):
        out1=F.relu(self.conv7(x)+self.convres3(x),inplace=True)
        out2=F.relu(self.conv9(x)+self.convres3(x),inplace=True)
        out3=F.relu(self.conv11(x)+self.convres3(x),inplace=True)
        out4 = F.relu(self.convres3(x),inplace=True)
        out = out1+out2+out3+out4
        return out

def get_backbone(encoder_name, pretrain_model_path):
    backbone = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3
    ).encoder
    backbone.load_state_dict(torch.load(pretrain_model_path))
    return backbone

class unetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp2, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1,bias=True)
        self.bn=nn.BatchNorm2d(out_size)
        self.conv2 =PLKC131197(out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs=F.relu(self.bn(outputs),inplace=True)
        identity = outputs
        outputs = self.conv2(outputs)
        return outputs + identity

class unetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp3, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = PLKC11975(out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = F.relu(self.bn(outputs), inplace=True)
        identity = outputs
        outputs = self.conv2(outputs)
        return outputs + identity


class unetUp4(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp4, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = PLKC9753(out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = F.relu(self.bn(outputs), inplace=True)
        identity = outputs
        outputs = self.conv2(outputs)
        return outputs + identity

class unetUp5(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp5, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = PLKC7531(out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = F.relu(self.bn(outputs), inplace=True)
        identity = outputs
        outputs = self.conv2(outputs)
        return outputs + identity


class Unet(nn.Module):
    def __init__(self, encoder, num_classes, pretrained_model_path):
        super(Unet, self).__init__()
        self.backbone = get_backbone(encoder, pretrained_model_path)
        in_filters = [192, 256, 512, 1536]
        out_filters = [32, 64, 128, 256]

        # in_filters = [224, 352, 704, 1152]       #swin_T
        # out_filters = [64, 128, 256,512]

        self.up_concat4 = unetUp2(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp3(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp4(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp5(in_filters[0], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            PLKC5310(out_filters[0]),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1,bias=True),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)


    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)[1:]
        # torch.Size([4, 128, 128, 128])
        # torch.Size([4, 128, 64, 64])
        # torch.Size([4, 256, 32, 32])
        # torch.Size([4, 512, 16, 16])
        # torch.Size([4, 1024, 8, 8])
        up4 = self.up_concat4(feat4, feat5)   #torch.Size([4, 256, 16, 16])
        up3 = self.up_concat3(feat3, up4)     #torch.Size([4, 128, 32, 32])
        up2 = self.up_concat2(feat2, up3)      #torch.Size([4, 64, 64, 64])
        up1 = self.up_concat1(feat1, up2)      #torch.Size([4, 32, 128, 128])
        up1 = self.up_conv(up1)                #torch.Size([4, 32, 256, 256])
        final = self.final(up1)                #torch.Size([4, 2, 256, 256])

        return final

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    model = Unet(num_classes=2)
    x = torch.randn(4, 3, 256, 256)
    output = model(x)
    print(output.size())
