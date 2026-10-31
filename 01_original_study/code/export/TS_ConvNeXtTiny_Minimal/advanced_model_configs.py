def get_advanced_model_config(name: str):
    """
    Returns the configuration dictionary for a given advanced model name.
    """
    all_configs = {
        # --- Idea 1: Deformable Conv for Lesion-Aware Tokenization ---
        "Adv_Idea_1_DeformableToken": {
            "name": "Adv_Idea_1_DeformableToken",
            "description": "Uses Deformable Convolutions in the CNN neck to create lesion-aware tokens for the ViT.",
            "use_deformable_conv": True,
            "fusion_method": "concat",
        },
        # --- Idea 2: CNN-ViT Feature Alignment with Contrastive Loss ---
        "Adv_Idea_2_ContrastiveAlign": {
            "name": "Adv_Idea_2_ContrastiveAlign",
            "description": "Forces CNN and ViT feature branches to align in a shared projection space using a contrastive loss.",
            "use_contrastive_loss": True,
            "fusion_method": "concat",
            "auxiliary_losses": {
                "contrastive": {"weight": 0.3, "type": "contrastive"}
            }
        },
        # --- Idea 3: Prototype Learning with Uncertainty ---
        # Note: Uncertainty part needs to be integrated in the trainer logic, here we just define the prototype head
        "Adv_Idea_3_PrototypeHead": {
            "name": "Adv_Idea_3_PrototypeHead",
            "description": "Replaces the final linear classifier with a learnable prototype for each class.",
            "use_prototype_head": True,
            "fusion_method": "concat",
            "auxiliary_losses": {
                "prototypical": {"weight": 0.5, "type": "prototypical"}
            }
        },
        # --- Idea 4: Hyper-parameter-free Attention Fusion (FiLM) ---
        "Adv_Idea_4_FiLMFusion": {
            "name": "Adv_Idea_4_FiLMFusion",
            "description": "Fuses CNN and ViT features using a Feature-wise Linear Modulation (FiLM) layer.",
            "fusion_method": "film",
        },
        # --- Idea 5: Ordinal Relation Preservation via Spectral Norm ---
        "Adv_Idea_5_SpectralNorm": {
            "name": "Adv_Idea_5_SpectralNorm",
            "description": "Applies Spectral Normalization to the classifier to encourage smoother, ordinally-aware predictions.",
            "use_spectral_norm": True,
            "fusion_method": "concat",
        },
    }
    
    if name not in all_configs:
        raise ValueError(f"Model config '{name}' not found. Available models: {list(all_configs.keys())}")
        
    # Add default values for keys that might be missing
    config = all_configs[name]
    defaults = {
        "use_deformable_conv": False,
        "use_contrastive_loss": False,
        "use_prototype_head": False,
        "use_spectral_norm": False,
        "fusion_method": "concat",
        "auxiliary_losses": {}
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            
    return config

def list_advanced_models():
    """Returns a list of all available advanced model names."""
    return [
        "Adv_Idea_1_DeformableToken",
        "Adv_Idea_2_ContrastiveAlign",
        "Adv_Idea_3_PrototypeHead",
        "Adv_Idea_4_FiLMFusion",
        "Adv_Idea_5_SpectralNorm",
    ]
