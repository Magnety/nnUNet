from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, mid_planes)
        # self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        # self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.gn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXtDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtDilatedBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(mid_planes)
        self.gn1 = nn.GroupNorm(32, mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        # self.bn2 = nn.BatchNorm3d(mid_planes)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.gn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt3D(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        self.gn1 = nn.GroupNorm(32, 64)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(ResNeXtDilatedBottleneck, 512, layers[2], shortcut_type, cardinality, stride=1)
        self.layer4 = self._make_layer(ResNeXtDilatedBottleneck, 1024, layers[3], shortcut_type, cardinality, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    # nn.BatchNorm3d(planes * block.expansion)
                    nn.GroupNorm(32, planes * block.expansion),
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnext3d10(**kwargs):
    """Constructs a ResNeXt3D-10 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [1, 1, 1, 1], **kwargs)
    return model


def resnext3d18(**kwargs):
    """Constructs a ResNeXt3D-18 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnext3d34(**kwargs):
    """Constructs a ResNeXt3D-34 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext3d50(**kwargs):
    """Constructs a ResNeXt3D-50 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext3d101(**kwargs):
    """Constructs a ResNeXt3D-101 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext3d152(**kwargs):
    """Constructs a ResNeXt3D-152 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext3d200(**kwargs):
    """Constructs a ResNeXt3D-200 model."""
    model = ResNeXt3D(ResNeXtBottleneck, [3, 24, 36, 3], **kwargs)
    return model

class BackBone3D(nn.Module):
    def __init__(self):
        super(BackBone3D, self).__init__()
        net = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], num_classes=2)
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        # and if we use the resnet3d-101, change the block list with [3, 4, 23, 3]
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        # the layer0 contains the first convolution, bn and relu
        self.layer1 = nn.Sequential(*net[3:5])
        # the layer1 contains the first pooling and the first 3 bottle blocks
        self.layer2 = net[5]
        # the layer2 contains the second 4 bottle blocks
        self.layer3 = net[6]
        # the layer3 contains the media bottle blocks
        # with 6 in 50-layers and 23 in 101-layers
        self.layer4 = net[7]
        # the layer4 contains the final 3 bottle blocks
        # according the backbone the next is avg-pooling and dense with num classes uints
        # but we don't use the final two layers in backbone networks

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DAF3D(nn.Module):
    def __init__(self):
        super(DAF3D, self).__init__()
        self.backbone = BackBone3D()

        self.down4 = nn.Sequential(
            nn.Conv3d(2048, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(1024, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.attention4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.refine4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()

        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(),)

        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)

        self.predict1_4 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv3d(128, 1, kernel_size=1)

        self.predict2_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv3d(64, 1, kernel_size=1)

        self.predict = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        # Top-down
        down4 = self.down4(layer4)
        down3 = torch.add(
            F.upsample(down4, size=layer3.size()[2:], mode='trilinear'),
            self.down3(layer3)
        )
        down2 = torch.add(
            F.upsample(down3, size=layer2.size()[2:], mode='trilinear'),
            self.down2(layer2)
        )
        down1 = torch.add(
            F.upsample(down2, size=layer1.size()[2:], mode='trilinear'),
            self.down1(layer1)
        )
        down4 = F.upsample(down4, size=layer1.size()[2:], mode='trilinear')
        down3 = F.upsample(down3, size=layer1.size()[2:], mode='trilinear')
        down2 = F.upsample(down2, size=layer1.size()[2:], mode='trilinear')

        predict1_4 = self.predict1_4(down4)
        predict1_3 = self.predict1_3(down3)
        predict1_2 = self.predict1_2(down2)
        predict1_1 = self.predict1_1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))

        predict2_4 = self.predict2_4(refine4)
        predict2_3 = self.predict2_3(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict2_1 = self.predict2_1(refine1)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)

        aspp = self.aspp_gn(self.aspp_conv(aspp))

        predict = self.predict(aspp)

        predict1_1 = F.upsample(predict1_1, size=x.size()[2:], mode='trilinear')
        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='trilinear')
        predict1_3 = F.upsample(predict1_3, size=x.size()[2:], mode='trilinear')
        predict1_4 = F.upsample(predict1_4, size=x.size()[2:], mode='trilinear')

        predict2_1 = F.upsample(predict2_1, size=x.size()[2:], mode='trilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='trilinear')
        predict2_3 = F.upsample(predict2_3, size=x.size()[2:], mode='trilinear')
        predict2_4 = F.upsample(predict2_4, size=x.size()[2:], mode='trilinear')

        predict = F.upsample(predict, size=x.size()[2:], mode='trilinear')

        if self.training:
            return predict1_1, predict1_2, predict1_3, predict1_4, \
                   predict2_1, predict2_2, predict2_3, predict2_4, predict
        else:
            return predict
