import torch
import torch.nn as nn
import torchvision.models as models


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=9):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            # Use pretrained ResNet18 as backbone
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

            # Modify first conv if needed
            if input_channels != 3:
                resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Remove final FC layer and avgpool
            self.features = nn.Sequential(*list(resnet.children())[:-2])

            # Add PatchGAN head
            self.classifier = nn.Sequential(
                nn.Conv2d(512, 256, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 128, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 1, 4, padding=1)
            )
        else:
            # Standard PatchGAN discriminator
            def discriminator_block(in_filters, out_filters, normalize=True):
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *discriminator_block(input_channels, 64, normalize=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, 1, 4, padding=1)
            )
            self.features = None

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
            return self.classifier(x)
        else:
            return self.model(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
