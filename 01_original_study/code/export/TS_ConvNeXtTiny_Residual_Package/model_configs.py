"""
Configuration profiles for 10 CNN+ViT research ideas.

Each configuration enables/disables specific components and hyperparameters
corresponding to the novel architectural modifications.
"""

MODEL_CONFIGS = {
    
    "Idea_1_ConfidenceGatedFusion": {
        "name": "Confidence-Gated Local-Global Fusion",
        "description": "Add confidence-gated fusion block that learns per-sample weights",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 1.0, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"entropy_reg": 0.01},
    },
    
    "Idea_2_MultiScaleLesionAttention": {
        "name": "Lesion-Scale Spatial-Channel Co-Attention Pyramid",
        "description": "Multi-scale co-attention combining channel and spatial attention",
        "fusion_method": "concat",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 1.0, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"scale_attention_weight": 0.3},
    },
    
    "Idea_3_UncertaintyTokenRefinement": {
        "name": "Uncertainty-Driven Token Refinement Transformer",
        "description": "Iteratively prune/reweight uncertain tokens in ViT",
        "fusion_method": "gate",
        "use_uncertainty_refinement": True,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 1.0, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"token_beta": 1.5, "uncertainty_weight": 0.1},
    },
    
    "Idea_4_OrdinalQWKOptimization": {
        "name": "Ordinal-Distribution Aware QWK Optimization",
        "description": "Hybrid objective with ordinal dist., QWK surrogate, and ordinal labels",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": True,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 0.6, "ordinal": 0.2, "qwk": 0.2, "proto": 0.0},
        "hyperparam": {"ordinal_weight": 0.2, "qwk_weight": 0.2, "class_margin": 0.1},
    },
    
    "Idea_5_DualStreamCrossAttention": {
        "name": "Dual-Stream Cross-Attention with Lesion Prototype Memory",
        "description": "Forces both branches to align through cross-attention to prototypes",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": True,
        "use_dual_attention": True,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 0.7, "ordinal": 0.0, "qwk": 0.0, "proto": 0.3},
        "hyperparam": {"prototype_momentum": 0.99, "proto_compact_weight": 0.5},
    },
    
    "Idea_6_TopologyGraphTransformer": {
        "name": "Topology-Aware Retinal Graph Transformer (Structural)",
        "description": "Graph attention network over lesion candidates and anatomical regions",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 1.0, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"graph_weight": 0.2},
    },
    
    "Idea_7_FrequencySpatialDual": {
        "name": "Frequency-Spatial Dual Domain Hybrid Encoder",
        "description": "Parallel frequency branch with cross-domain attention",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": True,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 1.0, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"frequency_weight": 0.3},
    },
    
    "Idea_8_MixtureOfExpertsSeverity": {
        "name": "Mixture-of-Experts Severity Router (Complex)",
        "description": "Expert sub-networks specialized for low/mid/high severity",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": True,
        "use_self_distillation": False,
        "loss_weights": {"ce": 0.8, "ordinal": 0.0, "qwk": 0.1, "proto": 0.1},
        "hyperparam": {"num_experts": 3, "load_balance_weight": 0.01},
    },
    
    "Idea_9_CausalCounterfactual": {
        "name": "Causal Counterfactual Lesion Consistency",
        "description": "Lesion-focused counterfactual consistency with causal loss",
        "fusion_method": "gate",
        "use_uncertainty_refinement": False,
        "use_ordinal_head": False,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": False,
        "loss_weights": {"ce": 0.7, "ordinal": 0.0, "qwk": 0.0, "proto": 0.0},
        "hyperparam": {"causal_consistency_weight": 0.3},
    },
    
    "Idea_10_TriLevelSelfDistillation": {
        "name": "Tri-Level Uncertainty-Calibrated Self-Distillation (Publication-Level)",
        "description": "Multi-level distillation with uncertainty weighting and ordinal calibration",
        "fusion_method": "gate",
        "use_uncertainty_refinement": True,
        "use_ordinal_head": True,
        "use_prototype_memory": False,
        "use_dual_attention": False,
        "use_frequency_branch": False,
        "use_moe_router": False,
        "use_self_distillation": True,
        "loss_weights": {"ce": 0.5, "ordinal": 0.2, "qwk": 0.1, "proto": 0.0},
        "hyperparam": {
            "distill_weight": 0.5,
            "distill_temp": 3.0,
            "ema_momentum": 0.999,
            "calibration_weight": 0.2,
        },
    },
}


def get_model_config(idea_name: str) -> dict:
    """Get configuration for a specific idea."""
    if idea_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown idea: {idea_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[idea_name]


def list_all_models() -> list:
    """List all available model ideas."""
    return list(MODEL_CONFIGS.keys())


def get_model_description(idea_name: str) -> str:
    """Get human-readable description."""
    config = get_model_config(idea_name)
    return f"{config['name']}: {config['description']}"
