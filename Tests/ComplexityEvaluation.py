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
from nnunet.network_architecture.focalunetr import FocalUNETR
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
    nnunet_4_model = Generic_UNet(1, 32, 26,
                                  len([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]]) - 1,
                                  2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
                                  nn.Dropout3d, {'p': 0, 'inplace': True},
                                  nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False, False,
                                  lambda x: x, InitWeights_He(1e-2),
                                  [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
                                  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], False,
                                  True, True,
                                  max_num_features=256)
    AU_model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=26,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(2, 2, 2, 2, 2),
        # dropout=0.1
        #     drop_rate=0.25,
    )
    AU_4_model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=26,
        channels=(32, 64, 128, 256, 256),
        strides=(2, 2, 2, 2),
        # dropout=0.1
        #     drop_rate=0.25,
    )

    STU_model = STUNet(1, 26, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 320, 320],
                       pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
                       conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]])
    STU_4_model = STUNet(1, 26, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 256],
                         pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]],
                         conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]])

    SUR_model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=26,
        feature_size=48,
        #     drop_rate=0.25,
        use_checkpoint=True,
    )
    focalUnetr = FocalUNETR(
        img_size=96, num_classes=26, in_chans=1, embed_dim=48,
        patch_size=2, use_conv_embed=True
    )
    SU_model = SU(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SU_3_p2_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        patch_size=2,
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SU_4_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        patch_size=2,
        depths=(2, 2, 2, 2, 2),
        num_heads=(3, 6, 12, 24, 48),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV1_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        upsample=("expand", "finalExpand"),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV2_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        upsample=("expandV2", "finalExpandV2"),
        #     drop_rate=0.25,
        use_checkpoint=True,
    )

    SUV3_model = SwinUnet(
        img_size=96,
        in_channels=1,
        out_channels=26,
        feature_size=48,
        # patch_size=2,
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

    # Test one model at a time for more accurate results

    all_models = [
        ('nnunet', nnunet_model),
        # ('STUNet', STU_model),
        # ('AttentionUNet', AU_model),
        # ('SwinUNETR', SUR_model),
        # ('SwinUNet', SU_model),
        # ('nnunets4', nnunet_4_model),
        # ('STUNets4', STU_4_model),
        # ('AttentionUNets4', AU_4_model),
        # ('SwinUNets3p2', SU_3_p2_model),
        # ('SwinUNets4', SU_4_model),
        # ('SwinUNetV1', SUV1_model),
        # ('SwinUNetV2', SUV2_model),
        # ('SwinUNetV3', SUV3_model),
        # ('SwinUNetV4', SUV4_model),
    ]
    input = torch.randn(1, 1, 96, 96, 96)
    sample = torch.randn(1, 1, 96, 96, 96, device='cuda')  # (B, C, H, W, D)
    for model_name, model in all_models:
        print(model_name)
        results = benchmark(model.to('cuda'), sample, num_runs=1000)
        print(json.dumps(results, indent=4))
        print(100 * '-')
