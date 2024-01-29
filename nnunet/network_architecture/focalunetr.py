import torch
import torch.nn as nn
from focalnet import *
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from timm.models.layers import to_3tuple
import torch.nn.functional as F


def get_attentional_residual(residual, vesselness_map):
    if vesselness_map is not None:
        attention = F.interpolate(vesselness_map, size=residual.shape[2:], mode='trilinear', align_corners=True)
        return residual * attention
    else:
        return residual


class FocalUNETR(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=2,
                 in_chans=1,
                 num_classes=2,
                 embed_dim=24,
                 depths=(2, 2, 2, 2),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 focal_levels=(2, 2, 2, 2),
                 focal_windows=(3, 3, 3, 3),
                 use_conv_embed=False,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 **kwargs
                 ):
        super().__init__()

        self.num_layers = len(depths)

        self.num_classes = num_classes
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers + 1)]

        self.patch_embed = PatchEmbed(
            img_size=to_3tuple(img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if self.patch_norm else None,
            is_stem=True
        )

        self.patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder layers
        self.focal_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.focal_layers.append(BasicLayer(dim=self.embed_dim[i_layer],
                                                out_dim=self.embed_dim[i_layer + 1],
                                                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                                  self.patches_resolution[1] // (2 ** i_layer),
                                                                  self.patches_resolution[2] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchEmbed,
                                                focal_level=focal_levels[i_layer],
                                                focal_window=focal_windows[i_layer],
                                                use_conv_embed=use_conv_embed,
                                                use_checkpoint=use_checkpoint,
                                                use_layerscale=use_layerscale,
                                                layerscale_value=layerscale_value,
                                                use_postln=use_postln,
                                                use_postln_in_modulation=use_postln_in_modulation,
                                                normalize_modulator=normalize_modulator
                                                ))

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * embed_dim,
            out_channels=2 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * embed_dim,
            out_channels=4 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * embed_dim,
            out_channels=16 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16 * embed_dim,
            out_channels=8 * embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=num_classes)
        self.focal_layers.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in, vesselness_map=None):
        x, D, H, W = self.patch_embed(x_in)
        hidden_states_out = [reshape_tokens_to_volumes(x, D, H, W, True)]

        for layer in self.focal_layers:
            x, D, H, W = layer(x, D, H, W)
            hidden_states_out.append(reshape_tokens_to_volumes(x, D, H, W, True))

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, get_attentional_residual(hidden_states_out[3], vesselness_map))
        dec2 = self.decoder4(dec3, get_attentional_residual(enc3, vesselness_map))
        dec1 = self.decoder3(dec2, get_attentional_residual(enc2, vesselness_map))
        dec0 = self.decoder2(dec1, get_attentional_residual(enc1, vesselness_map))
        out = self.decoder1(dec0, get_attentional_residual(enc0, vesselness_map))
        logits = self.out(out)
        return logits


def reshape_tokens_to_volumes(x: torch.Tensor, d, h, w, normalize=False):
    b, L, c = x.shape
    x = x.reshape(b, d, h, w, c)
    if normalize:
        x = F.layer_norm(x, [c])
    x = x.permute(0, 4, 1, 2, 3)
    return x
