# Abstract
Internal ribosome entry sites (IRESs) in circular RNAs (circRNAs) are key elements that drive cap-independent translation, and their accurate identification is crucial for understanding circRNA function. Currently, most IRES prediction models are designed for linear RNA sequences, and models specifically for circRNAs still show suboptimal performance. To address this, we proposed a dual-attenuation fusion framework, circIRES-DAF, specifically designed for predicting circRNA IRESs. The model captures sequence features and local contextual information through a dual-channel sequence feature extraction module; meanwhile, it integrates a graph neural network to model RNA secondary structure. Furthermore, through local feature quality evaluation and global modality importance weighting, the model introduces a dual-attenuation fusion strategy to suppress noise and enhance discriminative capability. Ablation experiments demonstrate that sequence and structural features provide complementary information, and the dual-attenuation fusion mechanism significantly improves model performance. On independent test sets, circIRES-DAF outperforms existing mainstream methods. Furthermore, interpretability analysis reveals several potential consensus motifs associated with IRES elements, providing a powerful computational tool and biological insights for circRNA research.

## Features

- **Multi-channel sequence encoding**: Extracting sequence features
- **Graph neural network**: Processes RNA secondary structure as a graph
- **Dual attenuation fusion**: Intelligent feature fusion strategy
- **Ablation study support**: Easy comparison of different model configurations

## Project Structure

```
├── src/
│   ├── encoders/           # Feature encoders
│   ├── data/               # Dataset handling
│   ├── models/             # Model components
│   │   ├── seq_processor.py
│   │   ├── graph_processor.py
│   │   ├── attention.py
│   │   ├── fusion.py
│   │   └── unified_model.py
│   ├── training/           # Training & evaluation
│   └── utils/              # Utilities
├── main.py                 # Main entry point
├── requirements.txt
└── README.md
```

## dependencies
```
# Python 3.8.8

# Core dependencies
torch==2.4.1
torch-geometric==2.6.1
numpy==1.22.4
pandas==1.2.4
scikit-learn==0.24.1
matplotlib==3.3.4
seaborn==0.11.1
tqdm==4.59.0

# RNA secondary structure prediction
ViennaRNA==2.7.0
```



## Usage

### Run Full Ablation Study

```bash
python main.py \
    --train_fasta data/train.fa \
    --train_labels data/train.npy \
    --test_fasta data/test.fa \
    --test_labels data/test.npy \
    --output_dir results \
    --ablation
```

### Train Our Model

```bash
python main.py \
    --train_fasta data/train.fa \
    --train_labels data/train.npy \
    --test_fasta data/test.fa \
    --test_labels data/test.npy \
    --config dual_attenuation
```

## Data Format

- **FASTA file**: Standard FASTA format with RNA sequences
- **Labels**: NumPy array (.npy) with binary labels (0/1)

