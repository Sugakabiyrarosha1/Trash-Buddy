
## 🗑️ Trash-Buddy

### Your smart AI companion for waste sorting ♻️

**Trash-Buddy** is an AI-powered waste classification app that helps users identify whether an item is **organic** or **recyclable** — simply by taking a picture. Built using the [Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification/data), Trash-Buddy combines **computer vision** and **machine learning** to make waste disposal smarter, faster, and more sustainable.

---

### 🌍 Features

* 📸 **Image recognition** – Take or upload a photo of an item; Trash-Buddy predicts if it’s *organic* or *recyclable*.
* ⚡ **Fast & lightweight** – Uses a deep learning model optimized for real-time use (TensorFlow / PyTorch).
* 🧠 **AI-powered accuracy** – Trained on thousands of labeled waste images.
* 🪄 **Simple UI** – Clean and user-friendly interface for all ages.
* 🌱 **Eco impact** – Encourages better waste habits and reduces landfill contamination.

---

### 🧩 Tech Stack

* **Python** for training and preprocessing
* **TensorFlow / Keras** or **PyTorch** for model development
* **Streamlit / Flask** for the demo web app
* **OpenCV** for image handling
* **Kaggle Waste Classification Dataset** for training

---

### 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/Trash-Buddy.git
cd Trash-Buddy

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py
```

---

### 📊 Dataset

Dataset: [Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification/data)


#### 📁 Dataset Directory Structure

The dataset is organized into four main waste categories — **Hazardous**, **Non-Recyclable**, **Organic**, and **Recyclable** — each with detailed subcategories of images.

```text
Data/
│
├── Hazardous/
│   ├── batteries/
│   ├── e-waste/
│   ├── paints/
│   └── pesticides/
│
├── Non-Recyclable/
│   ├── ceramic_product/
│   ├── diapers/
│   ├── plastics_bags_wrappers/
│   ├── sanitary_napkin/
│   └── stroform_product/
│
├── Organic/
│   ├── coffee_tea_bags/
│   ├── egg_shells/
│   ├── food_scraps/
│   ├── kitchen_waste/
│   └── yard_trimmings/
│
└── Recyclable/
    ├── cans_all_type/
    ├── glass_containers/
    ├── paper_products/
    └── plastic_bottles/
```

### 🏷️ Category Descriptions

| **Category**       | **Subcategories**                                                               | **Description**                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Hazardous**      | Batteries, e-waste, paints, pesticides                                          | Contains harmful materials requiring special disposal (e.g., batteries, chemical containers, electronics, paint cans). |
| **Non-Recyclable** | Ceramic products, diapers, plastic bags & wrappers, sanitary napkins, styrofoam | Items that cannot be recycled or composted; often end up in landfills.                                                 |
| **Organic**        | Coffee/tea bags, egg shells, food scraps, kitchen waste, yard trimmings         | Biodegradable waste suitable for composting or organic recycling.                                                      |
| **Recyclable**     | Cans, glass containers, paper products, plastic bottles                         | Materials that can be processed and reused through recycling streams.                                                  |

Each subfolder contains **hundreds of labeled images**, making this dataset ideal for **multi-class waste classification** and **AI-powered recycling applications**.

---

### 💡 Future Improvements

* Mobile app version (TensorFlow Lite / ONNX)
* Real-time camera integration
* Gamified eco-challenges

---

### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

### 🧠 Inspiration

Trash-Buddy was created to make sustainability effortless — because even small, smart actions can make a big difference for the planet. 🌎


