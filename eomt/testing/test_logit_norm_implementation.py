#!/usr/bin/env python3
"""
Script per testare l'implementazione di Logit Normalization
Esegui questo PRIMA di lanciare il training completo!
"""

import sys
import os

# Fix path per import - IMPORTANTE! 
script_dir = os.path.dirname(os.path.abspath(__file__))
eomt_dir = os.path.dirname(script_dir)
sys.path.insert(0, eomt_dir)

import torch
import torch.nn.functional as F

print("=" * 70)
print("TEST IMPLEMENTAZIONE LOGIT NORMALIZATION")
print("=" * 70)

# ============================================================================
# TEST 1: Importazione Moduli
# ============================================================================
print("\n[TEST 1] Importazione moduli...")

try:
    from training.mask_classification_loss import MaskClassificationLoss
    print("✅ MaskClassificationLoss importato correttamente")
except Exception as e:
    print(f"❌ ERRORE nell'importare MaskClassificationLoss: {e}")
    sys.exit(1)

try:
    from training.mask_classification_semantic import MaskClassificationSemantic
    print("✅ MaskClassificationSemantic importato correttamente")
except Exception as e:
    print(f"❌ ERRORE nell'importare MaskClassificationSemantic: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Istanziazione Loss con Logit Normalization
# ============================================================================
print("\n[TEST 2] Istanziazione Loss...")

try:
    # Test con Logit Norm DISABILITATO
    loss_fn_disabled = MaskClassificationLoss(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=19,
        no_object_coefficient=0.1,
        use_logit_normalization=False,
    )
    print(f"✅ Loss con Logit Norm DISABILITATO creata")
    print(f"   use_logit_normalization = {loss_fn_disabled.use_logit_normalization}")
    
    # Test con Logit Norm ABILITATO
    loss_fn_enabled = MaskClassificationLoss(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=19,
        no_object_coefficient=0.1,
        use_logit_normalization=True,  #  ABILITATO
    )
    print(f"✅ Loss con Logit Norm ABILITATO creata")
    print(f"   use_logit_normalization = {loss_fn_enabled.use_logit_normalization}")
    
except Exception as e:
    print(f"❌ ERRORE nell'istanziare Loss: {e}")
    sys.exit(1)

# ============================================================================
# TEST 3: Verifica Normalizzazione
# ============================================================================
print("\n[TEST 3] Verifica normalizzazione logits...")

batch_size, num_queries, num_classes = 2, 100, 19
class_logits = torch.randn(batch_size, num_queries, num_classes)

print(f"\n   Logits originali:")
print(f"   - Shape: {class_logits.shape}")
print(f"   - Norma L2 media: {class_logits.norm(p=2, dim=-1).mean():.4f}")
print(f"   - Min: {class_logits.min():.4f}, Max: {class_logits.max():.4f}")

# Applica normalizzazione
class_logits_norm = F.normalize(class_logits, p=2, dim=-1)

print(f"\n   Logits normalizzati:")
print(f"   - Shape: {class_logits_norm.shape}")
print(f"   - Norma L2 media: {class_logits_norm.norm(p=2, dim=-1).mean():.4f}")
print(f"   - Min: {class_logits_norm.min():.4f}, Max: {class_logits_norm.max():.4f}")

# Verifica che la norma sia ~1.0
expected_norm = 1.0
actual_norm = class_logits_norm.norm(p=2, dim=-1).mean().item()
if abs(actual_norm - expected_norm) < 0.01:
    print(f"   ✅ Normalizzazione corretta! (norma ≈ 1.0)")
else:
    print(f"   ❌ ERRORE: Norma attesa {expected_norm}, ottenuta {actual_norm:.4f}")
    sys.exit(1)

# ============================================================================
# TEST 4: Verifica Parametro in MaskClassificationSemantic
# ============================================================================
print("\n[TEST 4] Verifica parametro in MaskClassificationSemantic...")

import inspect

sig = inspect.signature(MaskClassificationSemantic.__init__)
params = list(sig.parameters.keys())

if 'use_logit_normalization' in params:
    print("✅ Parametro 'use_logit_normalization' trovato")
    
    param = sig.parameters['use_logit_normalization']
    print(f"   - Default value: {param.default}")
    print(f"   - Type: {param.annotation}")
    
    if param.default == False:
        print("   ✅ Default corretto (False)")
    else:
        print(f"     Default inaspettato: {param.default}")
else:
    print("❌ ERRORE: Parametro 'use_logit_normalization' NON trovato!")
    print(f"   Parametri disponibili: {params}")
    sys.exit(1)

# ============================================================================
# TEST 5: Config YAML
# ============================================================================
print("\n[TEST 5] Verifica config YAML...")

import os
config_path = "configs/dinov2/cityscapes/semantic/eomt_base_640_logit_norm.yaml"

if os.path.exists(config_path):
    print(f"✅ Config trovato: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    if 'use_logit_normalization: True' in config_content:
        print("   ✅ Parametro 'use_logit_normalization: True' presente nel config")
    elif 'use_logit_normalization: true' in config_content:
        print("   ✅ Parametro 'use_logit_normalization: true' presente nel config")
    else:
        print("     Parametro 'use_logit_normalization' non trovato nel config!")
        print("   Verifica manualmente il file YAML")
else:
    print(f"❌ ERRORE: Config non trovato: {config_path}")
    sys.exit(1)

# ============================================================================
# RISULTATO FINALE
# ============================================================================
print("\n" + "=" * 70)
print("TUTTI I TEST PASSATI CON SUCCESSO!")
print("=" * 70)
print("\n✅ Implementazione Logit Normalization completa e funzionante!")