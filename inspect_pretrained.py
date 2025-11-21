import torch
import torchvision.models as models
from models import Discriminator


def inspect_resnet18_pretrained():
    """Inspect pre-trained ResNet18 architecture"""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    print("=" * 80)
    print("ResNet18 Pre-trained Architecture")
    print("=" * 80)

    # Show all layers
    print("\n1. Named Children (Main Blocks):")
    print("-" * 80)
    for name, module in resnet.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:15s} | {str(type(module).__name__):20s} | {num_params:,} params")

    # Show detailed structure
    print("\n2. Full Model Structure:")
    print("-" * 80)
    print(resnet)

    # Show parameter counts
    print("\n3. Parameter Statistics:")
    print("-" * 80)
    total_params = sum(p.numel() for p in resnet.parameters())
    trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return resnet


def inspect_discriminator(use_pretrained=True):
    """Inspect Discriminator with/without pre-trained weights"""
    print("\n" + "=" * 80)
    print(f"Discriminator (use_pretrained={use_pretrained})")
    print("=" * 80)

    disc = Discriminator(use_pretrained=use_pretrained)

    if use_pretrained:
        print("\n1. Feature Extractor (Pre-trained ResNet18):")
        print("-" * 80)
        for i, (name, module) in enumerate(disc.features.named_children()):
            num_params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen = "FROZEN" if trainable == 0 else "TRAINABLE"
            print(f"[{i}] {name:10s} | {str(type(module).__name__):20s} | "
                  f"{num_params:,} params | {frozen}")

        print("\n2. Classifier Head (Trainable):")
        print("-" * 80)
        for i, (name, module) in enumerate(disc.classifier.named_children()):
            num_params = sum(p.numel() for p in module.parameters())
            print(f"[{i}] {str(type(module).__name__):20s} | {num_params:,} params")
    else:
        print("\n1. PatchGAN Discriminator:")
        print("-" * 80)
        for i, module in enumerate(disc.model):
            num_params = sum(p.numel() for p in module.parameters())
            print(f"[{i}] {str(type(module).__name__):20s} | {num_params:,} params")

    total_params = sum(p.numel() for p in disc.parameters())
    trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"\nTotal: {total_params:,} params | Trainable: {trainable_params:,}")

    return disc


def freeze_pretrained_layers(discriminator, freeze_until_layer=None):
    """
    Freeze pre-trained layers in discriminator

    Args:
        discriminator: Discriminator model
        freeze_until_layer: Freeze up to this layer name (e.g., 'layer3')
                           If None, freeze all feature extractor layers
    """
    if not hasattr(discriminator, 'features'):
        print("No pre-trained features to freeze")
        return discriminator

    print("\n" + "=" * 80)
    print("Freezing Pre-trained Layers")
    print("=" * 80)

    if freeze_until_layer is None:
        # Freeze all features
        for param in discriminator.features.parameters():
            param.requires_grad = False
        print("Frozen: All feature extractor layers")
    else:
        # Freeze until specific layer
        freeze = True
        for name, module in discriminator.features.named_children():
            if freeze:
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Frozen: {name}")
            else:
                print(f"Trainable: {name}")

            if name == freeze_until_layer:
                freeze = False

    # Show updated statistics
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nTotal: {total_params:,} params")
    print(f"Frozen: {frozen_params:,} params ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable: {trainable_params:,} params ({trainable_params/total_params*100:.1f}%)")

    return discriminator


def compare_discriminators():
    """Compare pre-trained vs non-pretrained discriminators"""
    print("\n" + "=" * 80)
    print("COMPARISON: Pre-trained vs Non-pretrained")
    print("=" * 80)

    disc_pretrained = Discriminator(use_pretrained=True)
    disc_scratch = Discriminator(use_pretrained=False)

    params_pretrained = sum(p.numel() for p in disc_pretrained.parameters())
    params_scratch = sum(p.numel() for p in disc_scratch.parameters())

    print(f"Pre-trained discriminator: {params_pretrained:,} params")
    print(f"From-scratch discriminator: {params_scratch:,} params")
    print(f"Difference: {abs(params_pretrained - params_scratch):,} params")


def show_layer_output_shapes():
    """Show output shapes for each layer"""
    print("\n" + "=" * 80)
    print("Layer Output Shapes (Input: 3x256x256)")
    print("=" * 80)

    disc = Discriminator(use_pretrained=True)
    disc.eval()

    x = torch.randn(1, 3, 256, 256)

    print("\nFeature Extractor:")
    print("-" * 80)
    for i, layer in enumerate(disc.features):
        x = layer(x)
        print(f"After layer {i}: {list(x.shape)}")

    print("\nClassifier Head:")
    print("-" * 80)
    for i, layer in enumerate(disc.classifier):
        x = layer(x)
        print(f"After layer {i}: {list(x.shape)}")

    print(f"\nFinal output shape: {list(x.shape)}")


if __name__ == "__main__":
    # 1. Inspect pre-trained ResNet18
    resnet = inspect_resnet18_pretrained()

    # 2. Inspect discriminator architectures
    disc_pretrained = inspect_discriminator(use_pretrained=True)
    disc_scratch = inspect_discriminator(use_pretrained=False)

    # 3. Compare them
    compare_discriminators()

    # 4. Show layer output shapes
    show_layer_output_shapes()

    # 5. Demonstrate freezing
    print("\n" + "=" * 80)
    print("EXAMPLE: Freezing Strategies")
    print("=" * 80)

    # Strategy 1: Freeze all pre-trained layers
    disc1 = Discriminator(use_pretrained=True)
    disc1 = freeze_pretrained_layers(disc1, freeze_until_layer=None)

    # Strategy 2: Freeze only early layers
    print("\n")
    disc2 = Discriminator(use_pretrained=True)
    disc2 = freeze_pretrained_layers(disc2, freeze_until_layer='layer2')

    # Strategy 3: Fine-tune everything
    print("\n" + "=" * 80)
    print("Strategy 3: Fine-tune All Layers (No Freezing)")
    print("=" * 80)
    disc3 = Discriminator(use_pretrained=True)
    total = sum(p.numel() for p in disc3.parameters())
    trainable = sum(p.numel() for p in disc3.parameters() if p.requires_grad)
    print(f"Total: {total:,} params | Trainable: {trainable:,} params (100%)")
