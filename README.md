# Coarse-to-Fine Domain Incremental Learning with Attentive Distillation for Mining Footprint Segmentation in Multispectral Imagery

Alif Tri Handoyo<sup>1,*</sup>, Vincent C.S. Lee<sup>2</sup>, Rizka Widyarini Purwanto<sup>1</sup>, Alex M. Lechner<sup>1,3</sup>, Deanna Kemp<sup>4</sup>, and Muhamad Risqi U. Saputra<sup>1,*</sup>. Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026), AI and Social Good Track. [[Paper]](https://www.ijcai.org/proceedings/2026/800)

## Abstract

Automatically mapping and segmenting global mining footprints using remote sensing and deep learning is critical for monitoring the socio-environmental risks and impacts of mining, yet its progress is hindered by the scarcity of fine-grained annotated data. Although large-scale datasets with coarse boundaries are widely available, leveraging them to improve fine-grained segmentation is challenging due to significant domain shift. To address this, we propose MineC2FNet, a coarse-to-fine domain incremental learning framework that exploits abundant coarse data to enhance fine-grained mining footprint segmentation. MineC2FNet adopts a teacher–student architecture with attentive distillation at both the feature and prediction levels, selectively transferring generalized knowledge from the coarse domain while enabling boundary refinement using limited fine-grained data (fine domain). We further introduce an expertly validated dataset of 219 images with precise boundary annotations across diverse geographies and commodities. Extensive experiments against state-of-the-art approaches, including domain adaptation and domain incremental learning methods, demonstrate that MineC2FNet achieves superior performance while effectively handling domain shift.

## Repository structure

The following structure reflects the files in the release archive:

```text
README.md                            Project overview, setup and usage
LICENSE                              CC BY-NC 4.0 license
NOTICE                               Author and third-party attribution notes
config/
  config_RGBNIRSWIR_COARSE.yml       Coarse-domain data and output paths
  config_RGBNIRSWIR_FINE.yml         Fine-domain data and output paths
  config_RGBNIRSWIR_DISTIL_FINE.yml  Fine-domain paths and teacher checkpoint
Modules/
  Models/                            FPN, dual FPN, U-Net and DeepLabV3+ models
  custom_layers/                     Multispectral pretrained layers, blocks and attention
dataloader_dual.py                   Fine-domain data pipeline with teacher predictions/features
distiller_dual_phase.py              Teacher–student distillation logic
import_config_kickoff.py             Selects a YAML config using --channel
loss_new.py                          Loss functions and segmentation metrics
main.py                              Standard segmentation training entry point
main_dualphase.py                    MineC2FNet distillation training entry point
train.py                             Standard segmentation training implementation
train_dualphase.py                   Distillation training implementation
Evaluation_QADTAV2_RGBNIRSWIR.ipynb  Evaluation and inference notebook
environment.yml                      Conda environment specification
Saved_CSV/                           Destination for training logs
Saved_Weights/                       Destination for model checkpoints
```

## Dataset

The paper uses a coarse dataset and an expert-validated fine dataset of 219 images. The model is trained here with six-channel RGB+NIR+SWIR1+SWIR2 imagery and binary **Mining** / **Non-Mining** masks.

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1_sa_bhfqAy-0OYPH9v9urQ-jSDNlFUgP).

## Installation

Create the Conda environment from the provided file. The environment was prepared using Anaconda3 2022.05 on Linux.

```bash
conda env create -f environment.yml
conda activate monashenv
```

## Configure dataset paths

Edit the YAML file for the dataset and task under `config/`. The release configs use empty path values as placeholders. Set `train_img_dir`, `train_mask_dir`, `val_img_dir`, and `val_mask_dir` to your local split directories. For distillation, also set `teacher_model_weights` to the teacher model weights. If you add a new config, add its channel name and file mapping to `import_config_kickoff.py`.

## Training

Run the commands from the repository root. The examples below follow the six-channel FPN and DenseNet-121 setup used for the released experiments.

Train the teacher model:

```bash
python main.py \
  --channel RGBNIRSWIRCOARSE \
  --num_channel 6 \
  --class_weight no \
  --model FPN \
  --backbone densenet121 \
  --batch_size 8 \
  --activation_output sigmoid \
  --n_epochs 100 \
  --n_classes 1 \
  --lr 0.001 \
  --useaug True \
  --name_file_weight MODIFIEDBOSSIMR256SWR_WEIGHT \
  --name_file_csv MODIFIEDBOSSIMR256SWR_CSV
```

Train MineC2FNet with attentive distillation:

```bash
python main_dualphase.py \
  --channel RGBNIRSWIRDISTILFINE \
  --num_channel 6 \
  --class_weight no \
  --model FPN \
  --backbone densenet121 \
  --batch_size 8 \
  --activation_output sigmoid \
  --n_epochs 100 \
  --n_classes 1 \
  --lr 0.001 \
  --useaug True \
  --name_file_weight QADTAV2_WEIGHT \
  --name_file_csv QADTAV2_CSV
```

The channel names select configs through `import_config_kickoff.py`. The distillation run loads its teacher from `teacher_model_weights` in `config_RGBNIRSWIR_DISTIL_FINE.yml` and saves checkpoints and logs to the configured output directories.

## Trained model and evaluation

Download the trained MineC2FNet checkpoint from [Google Drive](https://drive.google.com/drive/folders/12lp2C8iwRaILzGowLKHJmr5a11kRdYa4).

After downloading, extract the checkpoint archive under `Saved_Weights/` to obtain the TensorFlow checkpoint directory. Set the paths in `Evaluation_QADTAV2_RGBNIRSWIR.ipynb` for your test images and masks, then run the notebook to generate predictions and evaluation results.

## Citation

```bibtex
@inproceedings{ijcai2026p800,
  title     = {Coarse-to-Fine Domain Incremental Learning with Attentive Distillation for Mining Footprint Segmentation in Multispectral Imagery},
  author    = {Handoyo, Alif Tri and Lee, Vincent C.S. and Purwanto, Rizka Widyarini and Lechner, Alex M. and Kemp, Deanna and Saputra, Muhamad Risqi U.},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {7192--7200},
  year      = {2026},
  month     = {8},
  note      = {AI and Social Good},
  doi       = {10.24963/ijcai.2026/800},
  url       = {https://doi.org/10.24963/ijcai.2026/800},
}
```

## License

This repository is released under the [CC BY-NC 4.0](LICENSE) license. Pretrained models, datasets and third-party software are subject to their own licenses; see [NOTICE](NOTICE).
