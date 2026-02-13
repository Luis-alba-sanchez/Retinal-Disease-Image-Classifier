# Eye Disease Classifier

A PyTorch-based deep learning project for automatic classification of ocular diseases from retinal fundus images. This project explores multiple CNN architectures and transfer learning approaches for binary disease risk classification.

## 🎯 Project Goal

This is a learning-focused project designed to:
- Master **PyTorch** for computer vision and image classification
- Explore and compare multiple model architectures (Simple CNN, DenseNet121, ResNet50)
- Implement transfer learning and fine-tuning techniques
- Build a strong portfolio project for roles in ML/AI and bioinformatics
- Serve as a foundation for future Vision Transformer implementations

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
| Simple CNN | Custom CNN | Training |
| ResNet50 | Transfer Learning (Fine-tuned) | Training |
| DenseNet121 | Transfer Learning (Fine-tuned) | Training |
| Vision Transformer | ViT-based classifier | Planned |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
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
pip install torchmetrics
```

### Download Dataset
- Download RFMiD dataset from: https://riadd.grand-challenge.org/download-all-classes/
- Extract to the `data/source/` directory

## 📁 Project Structure

```
├── classes/                          # Model definitions
│   ├── RetinaSimpleCNN.py           # Custom CNN architecture
│   ├── CNNBinaryClassif.py          # Binary classification wrapper
│   ├── CNNMultiClassMultiLabeling.py # Multi-label approach
│   └── RetinaDataset.py             # PyTorch Dataset class
│
├── model-training/                   # Training pipelines
│   ├── Training-Notebooks/          # Jupyter notebooks for training
│   │   ├── Simple_CNN_BC.ipynb
│   │   ├── FT_ResNet50_MCMLC.ipynb
│   │   └── FT_DenseNet121_MCMLC.ipynb
│   ├── Models/                      # Trained model checkpoints
│   └── Training-Statistics/         # CSV logs of training metrics
│
├── data/                             # Dataset management
│   ├── organizer.py                 # Dataset preparation scripts
│   ├── Training-Set/                # Training images & labels
│   ├── Evaluation-Set/              # Validation images & labels
│   └── mean-std/                    # Normalization statistics
│
├── tools/                            # Utility modules
│   ├── data_tools.py                # Data loading & preprocessing
│   ├── model_tools.py               # Model utilities
│   └── visualization_tools.py       # Plotting & visualization
│
├── EDA/                              # Exploratory Data Analysis
│   ├── calculate_normalisation_statistics.py
│   └── DataSet_Analisys.ipynb
│
└── test.ipynb                        # Testing & inference examples
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

### Running Inference

Use `test.ipynb` to load a trained model and make predictions on new retinal images.

### Data Exploration

Run `EDA/DataSet_Analisys.ipynb` to explore dataset statistics and distribution.

## 🔍 Key Features

- **PyTorch Implementation**: Full PyTorch pipeline for training, validation, and testing
- **Transfer Learning**: Fine-tuning of pre-trained models (ResNet50, DenseNet121)
- **Data Augmentation**: Image preprocessing and normalization
- **Multi-scale Image Processing**: Support for different input sizes (256×256, 516×516, 1024×1024)
- **Training Tracking**: CSV-based logging of training metrics and evolution
- **Modular Design**: Reusable classes for easy experimentation

## 📈 Current Status

- ✅ Data loading and preprocessing pipelines
- ✅ Multiple model architectures implemented
- ✅ Training frameworks set up
- 🔄 **Currently**: Finding optimal models for high-resolution images (1024×1024)
- ⏳ Model training and evaluation coming soon
- 🎯 Vision Transformer implementation planned for future

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

## 📝 License

This project uses the RFMiD dataset under **CC-BY 4.0** license.

## 🌟 Future Work

- Vision Transformer (ViT) implementation for retinal image classification
- Ensemble methods combining multiple architectures
- Explainability analysis (Grad-CAM, attention maps)
- Web interface for inference
- Performance benchmarking and optimization

## 👤 Author

Created as a portfolio project for roles in Machine Learning/AI and Bioinformatics.

---

For questions or suggestions, feel free to open an issue or reach out!