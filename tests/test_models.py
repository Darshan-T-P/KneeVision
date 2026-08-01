import torch

from kneevision.models.image_model import (
    AVAILABLE_MODELS,
    BACKBONE_REGISTRY,
    _infer_model_name,
    _infer_ordinal,
    ImprovedHead,
)


def test_available_models_registered():
    assert AVAILABLE_MODELS == set(BACKBONE_REGISTRY)
    assert {"densenet121", "efficientnet-b4", "vit_b_16", "swin_t"} <= AVAILABLE_MODELS


def test_improved_head_output_shape():
    head = ImprovedHead(1024, 5)
    out = head(torch.randn(4, 1024))
    assert out.shape == (4, 5)


def test_improved_head_ordinal_shape():
    head = ImprovedHead(1024, 4)
    assert len(head) == 6
    assert head[5].out_features == 4


def test_infer_model_name_from_filenames():
    assert _infer_model_name("best_densenet121.pt") == "densenet121"
    assert _infer_model_name("best_vit_b_16_ordinal.pt") == "vit_b_16"
    assert _infer_model_name("checkpoint_swin_t.pt") == "swin_t"


def test_infer_ordinal():
    head = ImprovedHead(1024, 4)
    sd = {"classifier.net.5.weight": head.net[5].weight.detach(), "classifier.net.5.bias": head.net[5].bias.detach()}
    assert _infer_ordinal(sd, num_classes=5) is True

    head = ImprovedHead(1024, 5)
    sd = {"classifier.net.5.weight": head.net[5].weight.detach(), "classifier.net.5.bias": head.net[5].bias.detach()}
    assert _infer_ordinal(sd, num_classes=5) is False
