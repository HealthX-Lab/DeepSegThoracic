import json
import torch
from torch import nn
from thop import profile
from flopth import flopth
from torchsummary import summary
from calflops import calculate_flops
from fvcore.nn import FlopCountAnalysis
from pytorch_benchmark import benchmark
from ptflops import get_model_complexity_info
from monai.networks.nets import AttentionUnet, SwinUNETR
from nnunet.network_architecture.STUNet import STUNet
from nnunet.network_architecture.SwinUnetConventional import SwinUnet as SwinUnetConventional
from nnunet.network_architecture.SwinUnet3D import SwinUnet
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from deepspeed.profiling.flops_profiler import get_model_profile

print(torch.__version__)
if __name__ == '__main__':
    nnunet_model = Generic_UNet(1, 32, 26,
                                len([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]]),
                                2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
                                nn.Dropout3d, {'p': 0, 'inplace': True},
                                nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False, False,
                                lambda x: x, InitWeights_He(1e-2),
                                [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
                                [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], False,
                                True, True)
    AU_model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=26,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(2, 2, 2, 2, 2),
        # dropout=0.1
        #     drop_rate=0.25,
    )
    STU_model = STUNet(1, 26, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 320, 320],
                       pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
                       conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]])
    SUR_model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=26,
        feature_size=48,
        #     drop_rate=0.25,
        use_checkpoint=True,
    )
    SU_model = SwinUnetConventional(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        patch_size=4,
        # upsample=("expand", "finalExpand"),
        # skip_connection='linear',
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV1_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        upsample=("expandV2", "finalExpandV2"),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV2_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        upsample=("expandV3", "finalExpandV3"),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV3_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        patch_size=2,
        upsample=("expandV3", "finalExpandV3"),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV4_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        upsample=("expandV3", "finalExpandV3"),
        downsample='mergingv3',
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    all_models = [('nnunet', nnunet_model), ('STUNet', STU_model), ('AttentionUNet', AU_model),
                  ('SwinUNETR', SUR_model),
                  ('SwinUNet', SU_model)]
    input = torch.randn(1, 1, 96, 96, 96)
    for model_name, model in all_models:
        print(model_name)
        macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print(f'{model_name}: MACS {macs}, Params {params}')
        summary(model, input_size=(1, 96, 96, 96), device='cpu')
        print(100 * '-')
