# VMambaDA: Visual State Space Model with Unsupervised Domain Adaptation in Cervical Cancer Detection

## Introduction

VMambaDA is a deep learning project designed for cervical cancer detection using unsupervised domain adaptation techniques. It leverages a Visual State Space Model (VMamba) for feature extraction, coupled with Maximum Classifier Discrepancy (MCD) and Sliced Wasserstein Distance (SWD) for minimizing the domain gap between public and private datasets.

This project is structured to perform training, validation, and testing on both source and target datasets, utilizing various loss functions and model architectures.

## Features

- **VMamba Model**: For extracting visual features from images.
- **MCD**: Helps with unsupervised domain adaptation.
- **Sliced Wasserstein Distance (SWD)**: Reduces domain discrepancy.
- **Data Split**: Training, validation, and testing for both source and target datasets.

## Dataset

The dataset is split into source and target domains, with separate folders for training, validation, and testing:

```plaintext
dataset/
├── source/                # Source dataset (public)
│   ├── test/
│   ├── train/
│   └── val/
└── target/                # Target dataset (private)
    ├── test/
    ├── train/
    └── val/
```

## Project Structure

Here is a high-level overview of the project directory:

```plaintext
VMambaDA/
├── configs/               # Configuration files for model and training
├── data/                  # Raw data files
├── dataset/               # Preprocessed data, split into source and target datasets
│   ├── source/
│   └── target/
├── kernels/               # Kernel functions or operations
├── loss/                  # Custom loss functions
├── models/                # Model architectures (VMamba, MCD, SWD)
├── pretrain/              # Pretrained model weights
├── utils/                 # Utility scripts for data preprocessing and evaluation
├── vscode-root/           # VSCode workspace files
├── config.py              # Main configuration file
├── main.py                # Entry point for running the model
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── solver.py              # Model solver logic (training/evaluation loop)
```

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed, along with `pip` to manage dependencies.

### Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/NguyenVH01/VMambaDA.git
   cd VMambaDA
   ```

2. Enviroment setup:
   VMambaDA recommends creating a conda environment and installing dependencies via pip. Use the following commands to set up your environment:  
   We also recommend using pytorch>=2.0 and cuda>=11.8, though lower versions of PyTorch and CUDA are supported.

***Create and activate a new conda environment***

```bash
conda create -n vmambada
conda activate vmambada
```

***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
***Check Selective Scan (optional)***

* If you want to compare the modules with `mamba_ssm`, install [`mamba_ssm`](https://github.com/state-spaces/mamba) first!

* To verify if our implementation of `selective scan` in VMambaDA matches `mamba_ssm`, you can use `selective_scan/test_selective_scan.py`. Change the value of `MODE = "mamba_ssm_sscore"` in `selective_scan/test_selective_scan.py` and run `pytest selective_scan/test_selective_scan.py`.

* If you want to check whether VMambaDA's `selective scan` implementation matches the reference code (`selective_scan_ref`), change the value of `MODE = "sscore"` in `selective_scan/test_selective_scan.py`, and run `pytest selective_scan/test_selective_scan.py`.

* `MODE = "mamba_ssm"` checks whether the results of `mamba_ssm` are close to `selective_scan_ref`, while `"sstest"` is reserved for further development.

* If you find that `mamba_ssm` (`selective_scan_cuda`) or `selective_scan` (`selective_scan_cuda_core`) are not close enough to `selective_scan_ref`, and the test fails, don't worry. Check if `mamba_ssm` and `selective_scan` results are close enough [here](https://github.com/state-spaces/mamba/pull/161).

* ***If you're interested in selective scan, feel free to explore [mamba](https://github.com/state-spaces/mamba), [mamba-mini](https://github.com/MzeroMiko/mamba-mini), [mamba.py](https://github.com/alxndrTL/mamba.py), and [mamba-minimal](https://github.com/johnma2006/mamba-minimal) for more information.***

***Dependencies for `Detection` and `Segmentation` (optional)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

---

This version uses **VMambaDA** consistently while maintaining clarity and context.

### Data Preparation

Place your datasets into the `dataset/source` and `dataset/target` directories, ensuring the correct split between training, validation, and testing sets.

## Usage

### Training the Model

To train the model, use the following command:

```bash
python main.py --config configs/train_config.yaml
```

### Configuration

Modify training and model configurations in `configs/train_config.yaml` to adjust parameters like:

- Learning rate
- Batch size
- Model architecture

## Results

The model's training logs, checkpoints, and final evaluation metrics will be saved in the `results/` directory.

### Example Results

- **Accuracy**: 92.5%
- **F1-Score**: 0.88
- **Precision**: 0.90
- **Recall**: 0.87

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite it as follows:

```
@article{VMambaDA,
  title={VMambaDA: Visual State Space Model with Unsupervised Domain Adaptation in Cervical Cancer Detection},
  author={Nguyen Vu and Truc Nguyen et al.},
  journal={},
  year={2024}
}
```

## Contact

For questions or collaboration, please contact [nguyenvuhoangwork@gmail.com](mailto:nguyenvuhoangwork@gmail.com).