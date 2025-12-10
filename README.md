# 🗑️ Trash-Buddy

### Your smart AI companion for waste sorting ♻️

**Trash-Buddy** is an AI-powered waste classification system that helps identify waste items across **18 subcategories** organized into 4 main categories: **Hazardous**, **Non-Recyclable**, **Organic**, and **Recyclable**. Built using PyTorch and deep learning, Trash-Buddy combines **computer vision** and **machine learning** to make waste disposal smarter, faster, and more sustainable.

---

## 🌍 Features

* 📸 **Multi-class Classification** – Classifies waste into 18 specific subcategories across 4 main categories
* ⚡ **Real-time Detection** – Live webcam integration for instant waste identification
* 🌐 **Web API** – Flask-based REST API for easy integration
* 🧠 **High Accuracy** – Trained on thousands of labeled waste images using transfer learning
* 📊 **Comprehensive Pipeline** – Complete ML pipeline from data analysis to deployment
* 🎯 **Model Optimization** – Quantization and pruning for efficient deployment
* 🔍 **Interpretability** – Grad-CAM visualizations to understand model decisions

---

## 🧩 Tech Stack

* **Python 3.8+** for development
* **PyTorch** for deep learning model development
* **Flask** for web API
* **OpenCV** for image processing and camera integration
* **Jupyter Notebooks** for interactive development
* **scikit-learn** for data preprocessing and evaluation
* **Optuna** for hyperparameter tuning
* **SHAP** for model interpretability

---

## 📁 Project Structure

```
Trash-Buddy/
│
├── Data/                          # Dataset directory
│   ├── Hazardous/                # 4 subcategories
│   ├── Non-Recyclable/           # 5 subcategories
│   ├── Organic/                  # 5 subcategories
│   └── Recyclable/               # 4 subcategories
│
├── processed_data/                # Preprocessed data splits
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   ├── label_classes.npy
│   └── class_weights.json
│
├── models/                       # Trained models
│   ├── best_model_*.pth
│   ├── training_history.csv
│   └── evaluation_results.json
│
├── optimized_models/             # Optimized model variants
│   ├── *_fp16.pth
│   ├── *_int8.pth
│   ├── *_pruned_*.pth
│   └── optimization_comparison.csv
│
├── 1. Trash_Buddy_Dataset_Analysis.ipynb
├── 2. Trash_Buddy_Data_Preprocessing.ipynb
├── 3. Trash_Buddy_Model_Training.ipynb
├── 4. Trash_Buddy_Model_Evaluation.ipynb
├── 5. Trash_Buddy_Inference.ipynb
├── 6. Trash_Buddy_Model_Optimization_and_Compression.ipynb
├── 7. Trash_Buddy_Model_Interpretability_and_Explainability.ipynb
├── 8. Trash_Buddy_Hyperparameter_Tuning.ipynb
├── 9. Trash_Buddy_Ensemble_Methods.ipynb
├── flask_api_demo.py
├── camera_demo.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 📚 Notebook Pipeline

The project follows a comprehensive 9-step pipeline, each implemented as a separate Jupyter notebook:

### 1. Dataset Analysis (`1. Trash_Buddy_Dataset_Analysis.ipynb`)
**Purpose**: Exploratory Data Analysis (EDA) and dataset understanding

**Features**:
- Dataset summary and statistics
- Category and subcategory distribution analysis
- Image property analysis (size, format, dimensions)
- Class balance visualization
- Data quality assessment

**Outputs**: Dataset insights, visualizations, and statistics

---

### 2. Data Preprocessing (`2. Trash_Buddy_Data_Preprocessing.ipynb`)
**Purpose**: Prepare data for model training

**Features**:
- Image preprocessing and normalization
- Data augmentation strategies (standard and aggressive)
- Stratified train/validation/test splits (70/15/15)
- Class weight calculation for imbalanced data
- Label encoding

**Outputs**: 
- Preprocessed data splits (CSV files)
- Label classes and encoders
- Class weights for loss function

---

### 3. Model Training (`3. Trash_Buddy_Model_Training.ipynb`)
**Purpose**: Train deep learning models using transfer learning

**Features**:
- Support for multiple architectures (ResNet50, EfficientNet-B0, MobileNet-V2)
- Transfer learning with pretrained ImageNet weights
- Custom dataset class with aggressive augmentation for minority classes
- Training loop with progress tracking
- Learning rate scheduling
- Early stopping mechanism
- Model checkpointing

**Outputs**:
- Trained model checkpoints
- Training history (loss, accuracy, F1-scores)
- Best model saved for inference

---

### 4. Model Evaluation (`4. Trash_Buddy_Model_Evaluation.ipynb`)
**Purpose**: Comprehensive model performance analysis

**Features**:
- Overall performance metrics (accuracy, F1-scores)
- Per-class performance analysis
- Confusion matrix visualization
- Error analysis and misclassification patterns
- Category-level performance comparison
- Training history visualization
- Recommendations for improvement

**Outputs**:
- Evaluation results (JSON)
- Performance visualizations
- Error analysis reports

---

### 5. Inference Pipeline (`5. Trash_Buddy_Inference.ipynb`)
**Purpose**: Model inference and prediction utilities

**Features**:
- Single image prediction
- Batch image prediction
- Top-k predictions with confidence scores
- Prediction visualization
- Exportable inference script (`inference.py`)

**Outputs**:
- Standalone `inference.py` script for command-line usage

---

### 6. Model Optimization & Compression (`6. Trash_Buddy_Model_Optimization_and_Compression.ipynb`)
**Purpose**: Optimize models for deployment

**Features**:
- FP16 quantization (half precision for GPU)
- INT8 quantization (post-training for CPU/edge devices)
- Model pruning (L1 unstructured, 20% weight removal)
- Performance comparison (accuracy, speed, size)
- Model size reduction analysis
- Deployment recommendations

**Outputs**:
- Optimized model variants (FP16, INT8, pruned)
- Optimization comparison report
- Size and speed benchmarks

---

### 7. Model Interpretability & Explainability (`7. Trash_Buddy_Model_Interpretability_and_Explainability.ipynb`)
**Purpose**: Understand model decision-making

**Features**:
- Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
- Feature importance analysis
- Attention map generation
- Misclassification analysis with visual explanations
- Model behavior validation

**Outputs**:
- Grad-CAM heatmaps showing model focus areas
- Interpretability visualizations

---

### 8. Hyperparameter Tuning (`8. Trash_Buddy_Hyperparameter_Tuning.ipynb`)
**Purpose**: Optimize hyperparameters for better performance

**Features**:
- Automated hyperparameter search (Optuna, Ray Tune)
- Grid search and random search
- Bayesian optimization
- Learning rate finder
- Architecture search

**Outputs**:
- Optimized hyperparameters
- Search results and comparisons

---

### 9. Ensemble Methods (`9. Trash_Buddy_Ensemble_Methods.ipynb`)
**Purpose**: Combine multiple models for improved accuracy

**Features**:
- Voting ensemble (hard and soft voting)
- Stacking ensemble
- Model averaging
- Test-time augmentation
- Cross-validation ensemble

**Outputs**:
- Ensemble model predictions
- Performance comparison with individual models

---

## 🚀 Demo Applications

### Flask API Demo (`flask_api_demo.py`)

A web-based application for waste classification via file upload.

**Features**:
- Web interface for image upload
- REST API endpoint (`/predict`) for programmatic access
- Top-5 predictions with confidence scores
- Image preview and results visualization
- Support for PNG, JPG, JPEG, GIF, BMP formats

**Usage**:
```bash
python flask_api_demo.py
```

Then open `http://localhost:5000` in your browser.

**API Endpoint**:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

---

### Real-Time Camera Demo (`camera_demo.py`)

Live webcam integration for real-time waste detection.

**Features**:
- Real-time video feed from webcam
- Live predictions overlaid on video
- FPS counter
- Prediction smoothing using history
- Confidence-based color coding

**Controls**:
- `q` - Quit application
- `s` - Save current frame
- `r` - Reset predictions

**Usage**:
```bash
python camera_demo.py
```

**Requirements**: Webcam connected and accessible

---

### Standalone Inference Script (`inference.py`)

Command-line tool for batch image classification.

**Usage**:
```bash
python inference.py <image_path>
python inference.py <image_path> --top-k 5
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- Webcam (for camera demo)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/<your-username>/Trash-Buddy.git
cd Trash-Buddy
```

2. **Create virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. **Dataset Analysis** (Step 1):
   - Open `1. Trash_Buddy_Dataset_Analysis.ipynb`
   - Run all cells to analyze the dataset

2. **Data Preprocessing** (Step 2):
   - Open `2. Trash_Buddy_Data_Preprocessing.ipynb`
   - Run all cells to preprocess and split the data

3. **Model Training** (Step 3):
   - Open `3. Trash_Buddy_Model_Training.ipynb`
   - Configure training parameters
   - Run training cells (this may take several hours)

4. **Model Evaluation** (Step 4):
   - Open `4. Trash_Buddy_Model_Evaluation.ipynb`
   - Run all cells to evaluate model performance

5. **Inference** (Step 5):
   - Open `5. Trash_Buddy_Inference.ipynb`
   - Test predictions on sample images

6. **Optimization** (Step 6):
   - Open `6. Trash_Buddy_Model_Optimization_and_Compression.ipynb`
   - Generate optimized model variants

7. **Interpretability** (Step 7):
   - Open `7. Trash_Buddy_Model_Interpretability_and_Explainability.ipynb`
   - Generate Grad-CAM visualizations

8. **Hyperparameter Tuning** (Step 8):
   - Open `8. Trash_Buddy_Hyperparameter_Tuning.ipynb`
   - Optimize hyperparameters (optional)

9. **Ensemble Methods** (Step 9):
   - Open `9. Trash_Buddy_Ensemble_Methods.ipynb`
   - Create ensemble models (optional)

### Running Demos

**Flask API**:
```bash
python flask_api_demo.py
# Open http://localhost:5000 in browser
```

**Camera Demo**:
```bash
python camera_demo.py
# Press 'q' to quit
```

---

## 📊 Dataset

### Dataset Source
[Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification/data) on Kaggle

### Dataset Structure

The dataset contains **18 subcategories** organized into **4 main categories**:

```
Data/
│
├── Hazardous/              (4 subcategories)
│   ├── batteries/          (~114 images)
│   ├── e-waste/            (~544 images)
│   ├── paints/             (~153 images)
│   └── pesticides/         (~139 images)
│
├── Non-Recyclable/         (5 subcategories)
│   ├── ceramic_product/   (~139 images)
│   ├── diapers/            (~145 images)
│   ├── platics_bags_wrappers/ (~135 images)
│   ├── sanitary_napkin/    (~110 images)
│   └── stroform_product/   (~118 images)
│
├── Organic/                (5 subcategories)
│   ├── coffee_tea_bags/    (~157 images)
│   ├── egg_shells/         (~125 images)
│   ├── food_scraps/        (~147 images)
│   ├── kitchen_waste/      (~117 images)
│   └── yard_trimmings/     (~131 images)
│
└── Recyclable/             (4 subcategories)
    ├── cans_all_type/      (~272 images)
    ├── glass_containers/   (~142 images)
    ├── paper_products/    (~121 images)
    └── plastic_bottles/    (~130 images)
```

**Total**: ~2,500+ images across 18 waste subcategories

### Category Descriptions

| **Category**       | **Subcategories**                                                               | **Description**                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Hazardous**      | Batteries, e-waste, paints, pesticides                                          | Contains harmful materials requiring special disposal (e.g., batteries, chemical containers, electronics, paint cans). |
| **Non-Recyclable** | Ceramic products, diapers, plastic bags & wrappers, sanitary napkins, styrofoam | Items that cannot be recycled or composted; often end up in landfills.                                                 |
| **Organic**        | Coffee/tea bags, egg shells, food scraps, kitchen waste, yard trimmings         | Biodegradable waste suitable for composting or organic recycling.                                                      |
| **Recyclable**     | Cans, glass containers, paper products, plastic bottles                         | Materials that can be processed and reused through recycling streams.                                                  |

---

## 📦 Dependencies

Key dependencies (see `requirements.txt` for complete list):

- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `scikit-learn>=1.3.0` - Machine learning utilities
- `flask>=2.3.0` - Web framework
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image handling
- `optuna>=3.0.0` - Hyperparameter optimization
- `shap>=0.42.0` - Model interpretability
- `jupyter>=1.0.0` - Notebook environment

---

## 🎯 Model Performance

The trained model achieves:
- **Test Accuracy**: ~75-80% (varies by model architecture)
- **Weighted F1-Score**: ~0.74-0.78
- **Macro F1-Score**: ~0.70-0.75

Performance varies across classes, with better accuracy on classes with more training data.

---

## 🔧 Configuration

### Training Configuration

Key hyperparameters (configurable in Step 3):
- **Image Size**: 224x224
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss with class weights

### Model Architectures

Supported architectures:
- **ResNet50** (default)
- **EfficientNet-B0**
- **MobileNet-V2**

---

## 💡 Future Improvements

* 📱 Mobile app version (TensorFlow Lite / ONNX)
* 🌐 Cloud deployment (AWS, GCP, Azure)
* 🔄 Continuous learning pipeline
* 📊 Advanced analytics dashboard
* 🎮 Gamified eco-challenges
* 🌍 Multi-language support
* 🔔 Recycling center locator integration

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

* Dataset: [Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification/data) on Kaggle
* PyTorch team for the excellent deep learning framework
* OpenCV community for computer vision tools

---

## 🧠 Inspiration

Trash-Buddy was created to make sustainability effortless — because even small, smart actions can make a big difference for the planet. 🌎

**Let's work together to reduce waste and protect our environment!** ♻️

---

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

**Made with ❤️ for a sustainable future**
