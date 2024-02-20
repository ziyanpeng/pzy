import torch
from thop import profile
from thop import clever_format
device=torch.device('cpu')
import segmentation_models_pytorch as smp
from nets.network import Unet


# myprint=torch.zeros((3,224,224))
# flops,params=profile(BANet(weight_path=None).to(device),inputs=myprint,)
# flops,params=clever_format([flops,params],"%.3f")
# print(flops,params)

if __name__=='__main__':
    map_location = torch.device('cpu')
    #model = MANet()
    #model = BANet(weight_path=None)
    #model = Unet()
    #model = PSPNet(num_classes=2, backbone="mobilenet", downsample_factor=8, pretrained=False, aux_branch=False)
    #model = DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=8, pretrained=False)
    model = Unet(encoder="Swin-B", num_classes=2,
                  pretrained_model_path="model_data/swin_base_patch4_window7_224.pth")
    input = torch.randn(4, 3, 512, 512)
    flops, params = profile(model, inputs=(input,))
    print('flops', flops)  ## 打印计算量
    print('params', params)  ## 打印参数量
    flopsm, paramsm = clever_format([flops, params], "%.3f")
    print('flops', flopsm)  ## 打印计算量
    print('params', paramsm)  ## 打印参数量