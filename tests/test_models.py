import torch

from kneevision.models.image_model import (
    AVAILABLE_MODELS,
    BACKBONE_REGISTRY,
    KneeXRayClassifier,
    _infer_model_name,
    _infer_ordinal,
    _infer_aux_grades,
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


def test_classifier_without_aux_grades_has_no_aux_heads():
    model = KneeXRayClassifier("densenet121", num_classes=5)
    assert len(model.aux_heads) == 0
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)


def test_classifier_with_aux_grades_forward_with_aux():
    aux_grades = {"jsn_m": 4, "osteophyte_l": 4}
    model = KneeXRayClassifier("densenet121", num_classes=5, aux_grades=aux_grades)
    model.eval()
    assert set(model.aux_heads.keys()) == set(aux_grades)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        main_logits, aux_logits = model.forward_with_aux(x)
        # forward() (used at inference / by every other call site) is unaffected
        forward_only_logits = model(x)
    assert main_logits.shape == (2, 5)
    assert aux_logits["jsn_m"].shape == (2, 4)
    assert aux_logits["osteophyte_l"].shape == (2, 4)
    assert torch.allclose(forward_only_logits, main_logits)


def test_infer_aux_grades_detects_auxiliary_heads():
    state = {
        "aux_heads.jsn_m.weight": torch.randn(4, 1024),
        "aux_heads.jsn_m.bias": torch.randn(4),
        "aux_heads.osteophyte_l.weight": torch.randn(4, 1024),
        "aux_heads.osteophyte_l.bias": torch.randn(4),
        "classifier.net.5.weight": torch.randn(5, 1024),
    }
    assert _infer_aux_grades(state) == {"jsn_m": 4, "osteophyte_l": 4}


def test_infer_aux_grades_empty_for_standard_checkpoint():
    state = {"classifier.net.5.weight": torch.randn(5, 1024), "classifier.net.5.bias": torch.randn(5)}
    assert _infer_aux_grades(state) == {}
