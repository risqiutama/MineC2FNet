import argparse
from train import *

def argparser():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--batch_size", default=32, help="Batch size")
    parser.add_argument("--channel", help="RGB, RGB+NIR, RGB+NDVI, RGB+NIR+NDVI, NIR+NDVI, NIR+NDVI")
    parser.add_argument("--model", help="Select Model: FPN, FPNRAW, UNET, DeepLabV3+")
    parser.add_argument("--set_trainable", default=True, help="True for Finetuning, False for Freeze")
    parser.add_argument("--useaug", default=True, help="Setting for Augmentation")
    parser.add_argument("--num_channel", default=4, help="Number of channel")
    parser.add_argument("--class_weight", default="yes", help="Use class weight?")
    parser.add_argument("--activation_output", help="Activation output")
    parser.add_argument("--n_epochs", default=20, help="Number of Epochs")
    parser.add_argument("--seed", default=42, help="Seed")
    parser.add_argument("--n_classes", default=9, help="Number of classes")
    parser.add_argument("--backbone", help="Backbone Model")
    parser.add_argument("--lr", default=0.0001, help="Learning rate")
    parser.add_argument("--name_file_weight", default="Train_Weight_", help="Weight file name")
    parser.add_argument("--name_file_csv", default="Train_CSV_", help="CSV file name")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argparser()
    train_model(args)

