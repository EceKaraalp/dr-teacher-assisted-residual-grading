import os

class CFG:
    seed = 42
    num_classes = 5
    img_size = 224
    batch_size = 16
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    patience = 10
    
    # Paths
    data_dir = "APTOS2019"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "valid.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    img_dir = os.path.join(data_dir, "train_images")
    
    # Model
    cnn_backbone = "efficientnet_b4"
    trans_backbone = "vit_b_16"
    embed_dim = 512
