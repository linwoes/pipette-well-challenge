"""
Smoke test for DINOv2Backbone and DualViewFusion models.
Tests shape contracts and basic forward pass functionality.
"""

import sys
sys.path.insert(0, 'src')

def test_models():
    """Run smoke tests for backbone and fusion modules."""
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch torchvision")
        return False
    
    try:
        print("=" * 70)
        print("SMOKE TEST: ML Backbone and Fusion Modules")
        print("=" * 70)
        print()

        # Test 1: DINOv2Backbone
        print("TEST 1: DINOv2Backbone with LoRA")
        print("-" * 70)
        from models.backbone import DINOv2Backbone

        try:
            backbone = DINOv2Backbone(use_lora=True, lora_rank=8)
            x = torch.randn(2, 3, 224, 224)
            out = backbone(x)
            
            assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"
            trainable_count = backbone.trainable_parameters()
            print(f"✓ DINOv2Backbone forward pass: input {x.shape} → output {out.shape}")
            print(f"✓ Trainable parameters (LoRA only): {trainable_count:,}")
            print()
        except Exception as e:
            print(f"✗ DINOv2Backbone test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test 2: TemporalAttention
        print("TEST 2: TemporalAttention")
        print("-" * 70)
        from models.fusion import TemporalAttention

        try:
            temporal = TemporalAttention(d_model=768, nhead=8, num_layers=2)
            x = torch.randn(2, 8, 768)  # (B, N, D)
            out = temporal(x)
            
            assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"
            print(f"✓ TemporalAttention forward pass: input {x.shape} → output {out.shape}")
            print()
        except Exception as e:
            print(f"✗ TemporalAttention test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test 3: DualViewFusion
        print("TEST 3: DualViewFusion (Full Model)")
        print("-" * 70)
        from models.fusion import DualViewFusion

        try:
            model = DualViewFusion(num_rows=8, num_columns=12, shared_backbone=True)
            fpv = torch.randn(2, 8, 3, 224, 224)
            tv = torch.randn(2, 8, 3, 224, 224)
            row_logits, col_logits = model(fpv, tv)
            
            assert row_logits.shape == (2, 8), f"Expected row (2, 8), got {row_logits.shape}"
            assert col_logits.shape == (2, 12), f"Expected col (2, 12), got {col_logits.shape}"
            print(f"✓ FPV input: {fpv.shape} → row logits {row_logits.shape}")
            print(f"✓ TopView input: {tv.shape} → col logits {col_logits.shape}")
            print(f"✓ Raw logits (no sigmoid applied): row min/max = [{row_logits.min():.3f}, {row_logits.max():.3f}]")
            print()
        except Exception as e:
            print(f"✗ DualViewFusion test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test 4: WellDetectionLoss
        print("TEST 4: WellDetectionLoss (Focal Loss)")
        print("-" * 70)
        from models.fusion import WellDetectionLoss

        try:
            loss_fn = WellDetectionLoss(gamma=2.0, alpha=0.25)
            row_logits = torch.randn(2, 8)
            col_logits = torch.randn(2, 12)
            row_targets = torch.randint(0, 2, (2, 8)).float()
            col_targets = torch.randint(0, 2, (2, 12)).float()
            
            loss = loss_fn(row_logits, col_logits, row_targets, col_targets)
            assert loss.item() > 0, f"Loss should be > 0, got {loss.item()}"
            
            print(f"✓ Focal loss computation: {loss.item():.4f}")
            print(f"✓ Row targets: {row_targets.sum(dim=1).int()} wells per row")
            print(f"✓ Col targets: {col_targets.sum(dim=1).int()} wells per col")
            print()
        except Exception as e:
            print(f"✗ WellDetectionLoss test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Summary
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • DINOv2Backbone: (B,3,224,224) → (B,768) with LoRA fine-tuning")
        print("  • TemporalAttention: (B,N,768) → (B,768) via Transformer pooling")
        print("  • DualViewFusion: dual (B,N,3,224,224) → (B,8) + (B,12) logits")
        print("  • WellDetectionLoss: Focal loss for multi-label well classification")
        print()
        return True

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_models()
    sys.exit(0 if success else 1)
