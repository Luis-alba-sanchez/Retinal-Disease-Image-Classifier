# Eye Disease Classifier

A PyTorch-based deep learning project for automatic classification of ocular diseases from retinal fundus images. This project explores multiple CNN architectures and transfer learning approaches for binary disease risk classification and multi label classification.

## 🎯 Project Goal

This is a learning-focused project designed to:
- Master **PyTorch** for computer vision and image classification
- Explore and compare multiple model architectures (Simple CNN, DenseNet121, ResNet50)
- Implement transfer learning and fine-tuning techniques
- Build a strong portfolio project for roles in ML/AI and bioinformatics
- Implement and train Vision Transformer models for medical image classification

## 📊 Dataset

This project uses the **RFMiD (Retinal Fundus Multi-disease Image Dataset)**:
- **3,200** color fundus images
- Captured using **3 different fundus cameras**
- **46 conditions** annotated with consensus from senior retinal experts
- **License**: CC-BY 4.0
- **Source**: https://www.mdpi.com/2306-5729/6/2/14 ; https://riadd.grand-challenge.org/download-all-classes/ 

### Diseases Targeted
Currently developing a **binary classification model** to predict disease risk presence:
- **DR** (Diabetic Retinopathy)
- **MH** (Media Haze)
- **ODC** (Optic Disc Cupping)

## 🛠️ Models & Architectures

The project includes implementations of multiple approaches:

| Model | Type | Status |
|-------|------|--------|
| Simple CNN | Custom CNN | Trained |
| ResNet50 | Transfer Learning (Fine-tuned) | Trained |
| DenseNet121 | Transfer Learning (Fine-tuned) | Trained |
| Vision Transformer | ViT-based classifier | Trained |
| Swin Transformer | Fine-tuned Swin-T | Trained |

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- GPU with CUDA support (tested on RTX 5070 Ti)
- At least 16GB RAM for model training

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Eye-Disease-Classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install numpy pillow matplotlib pandas tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install torchmetrics torchinfo
```

### Download Dataset
- Download RFMiD dataset from: https://riadd.grand-challenge.org/download-all-classes/
- Extract to the `data/source/` directory

## 📁 Project Structure

```
├── classes/                          # Model definitions
│   ├── ViT.py                        # Vision Transformer implementation
│   ├── RetinaSimpleCNN.py            # Custom CNN architecture
│   ├── CNNBinaryClassif.py           # Binary classification wrapper
│   ├── CNNMultiClassMultiLabeling.py # Multi-label approach
│   └── RetinaDataset.py              # PyTorch Dataset class
│
├── model-training/                   # Training pipelines
│   ├── Training-Notebooks/           # Jupyter notebooks for training
│   │   ├── Simple_CNN_BC.ipynb
│   │   ├── Simple_CNN_MCMLC.ipynb
│   │   ├── FT_ResNet50_BC.ipynb
│   │   ├── FT_ResNet50_MCMLC.ipynb
│   │   ├── FT_DenseNet121_BC.ipynb
│   │   ├── FT_DenseNet121_MCMLC.ipynb
│   │   ├── FT_swin_t_BC.ipynb        # NEW
│   │   ├── FT_swin_t_MCMLC.ipynb     # NEW
│   │   └── Vision-Transformer-pytorch.ipynb  # NEW
│   ├── Models/                       # Trained model checkpoints
│   ├── Training-Statistics/          # CSV logs of training metrics
│   └── Training-Evolution-images/    # NEW - Training curves
│
├── data/                             # Dataset management
│   ├── Training-Set/                 # Training images & labels
│   ├── Test-Set/                     # Test images & labels
│   ├── Evaluation-Set/               # Validation images & labels
│   ├── source/                       # zip data from paper
│   └── mean-std/                     # Normalization statistics
│
├── tools/                            # Utility modules
│   ├── data_tools.py                # Data loading & preprocessing
│   ├── model_tools.py               # Model utilities
│   └── visualization_tools.py       # Plotting & visualization
│
└── EDA/                              # Exploratory Data Analysis
    ├── calculate_normalisation_statistics.py
    └── DataSet_Analisys.ipynb

```

## 📖 Usage

### Training a Model

Refer to the Jupyter notebooks in `model-training/Training-Notebooks/`:

1. **Simple CNN Binary Classification**:
   - Open `Simple_CNN_BC.ipynb`
   - Follow the notebook to train a custom CNN from scratch

2. **Fine-tuned Transfer Learning**:
   - Open `FT_ResNet50_MCMLC.ipynb` or `FT_DenseNet121_MCMLC.ipynb`
   - Learn how to fine-tune pre-trained models

### Data Exploration

Open `EDA/DataSet_Analisys.ipynb` to explore dataset statistics and distribution.

## 🔍 Key Features

- **PyTorch Implementation**: Full PyTorch pipeline for training, validation, and testing
- **Transfer Learning**: Fine-tuning of pre-trained models (ResNet50, DenseNet121)
- **Data Augmentation**: Image preprocessing and normalization
- **Multi-scale Image Processing**: Support for different input sizes (256×256, 516×516, 1024×1024)
- **Training Tracking**: CSV-based logging of training metrics and evolution
- **Modular Design**: Reusable classes for easy experimentation
- **Vision Transformer**: ViT architecture for advanced image understanding
- **Swin Transformer**: Fine-tuned Swin-T for efficient classification
- **Mixed Precision Training**: GPU optimization with torch.amp.GradScaler
- **Loss Balancing**: BCEWithLogitsLoss with class weight coefficients

## 📈 Current Status

- ✅ Data loading and preprocessing pipelines
- ✅ Multiple model architectures implemented (CNN, ResNet50, DenseNet121)
- ✅ Vision Transformer implementation completed
- ✅ Swin Transformer fine-tuning in progress
- 🔄 **Currently**: Training and optimizing ViT and Swin-T models (256×256 resolution)
- ⏳ Performance evaluation and benchmarking coming soon

## 🤖 Hardware

Developed and tested on:
- **CPU**: AMD Ryzen 7 7800X3D
- **GPU**: NVIDIA RTX 5070 Ti
- **RAM**: 16GB+

## 📚 Learning Resources

This project demonstrates:
- PyTorch fundamentals (tensors, autograd, nn.Module)
- Custom Dataset and DataLoader implementation
- Transfer learning and fine-tuning
- Model training loops with validation
- Checkpoint saving and loading
- GPU acceleration with CUDA

## 💡 How to Use This Project

If you're learning PyTorch and computer vision:
1. Explore the model definitions in `classes/`
2. Study the training notebooks for implementation patterns
3. Modify and experiment with hyperparameters
4. Use the code as a foundation for your own projects

## 📄 Dataset Citation

If you use the RFMiD dataset, please cite:
- Pachade, R.R.; Porwal, P.; Kokil, P.; et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Classification of Retinal Fundus Images Using Conventional Machine Learning and Deep Learning. Data 2021, 6, 14. https://doi.org/10.3390/data6020014

## 🌟 Future Work

- Ensemble methods combining multiple architectures
- Explainability analysis (Grad-CAM, attention maps)
- Web interface for inference
- Performance benchmarking and optimization

## 👤 Author

Created as a portfolio project for roles in Machine Learning/AI and Bioinformatics.
LinkedIn : https://www.linkedin.com/in/luis-alexandre-alba-sanchez/ 

---

For questions or suggestions, feel free to open an issue or reach out!