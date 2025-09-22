## 📄 README.md

## Output
<img width="712" height="467" alt="image" src="https://github.com/user-attachments/assets/addebaaf-1ce4-4823-8dd2-c953ec98e427" />

## Output ```=> alexnet : 0.687, mobilenet_v2: 0.313``` 
## Answer is yet unclear but will be either of this two class is gauranteed I am processing data again


````markdown
# 🧩 Victim Model Identification (Bit-by-Bit Hackathon)

This repository contains the full pipeline to **identify CNN victim models**  
(such as AlexNet vs MobileNet) using **hardware performance counter traces**  
collected via Linux `perf stat`.

The workflow includes:
- Trace collection
- Cleaning and preprocessing
- Feature extraction
- Model training & evaluation
- Victim prediction

---

## ⚙️ Setup

Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/hackathon-victim-identification.git
cd hackathon-victim-identification
````

Install requirements:

```bash
pip3 install -r requirements.txt
```

---

## 📂 Repo Structure

```
.
├── clean_all.py         # Cleans raw perf CSV traces into structured CSV
├── collect_traces.sh    # Collects traces using perf
├── extract_features.py  # Extracts statistical + time-series features
├── predict_victim.py    # Predicts victim model (e.g., AlexNet vs MobileNet)
├── train_model.py       # Trains classifier and saves model artifacts
├── utils.py             # Common helper functions
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Ignore generated data
```

---

## 🚀 Usage

### 1. Collect traces

Collect raw hardware performance counter traces for each model:

```bash
./collect_traces.sh mobilenet 15
./collect_traces.sh alexnet 15
```

This creates CSV logs in `traces/`.

### 2. Clean traces

Convert raw logs into structured CSV:

```bash
python3 clean_all.py
```

### 3. Extract features

Generate features and time-series vectors:

```bash
python3 extract_features.py
```

### 4. Train model

Train RandomForest classifier:

```bash
python3 train_model.py
```

### 5. Predict victim

For unknown victim traces:

```bash
python3 predict_victim.py
```

---

## 🛠 Requirements

* Python 3.8+
* Linux with `perf` installed
* Python libraries in `requirements.txt`:

  * numpy
  * pandas
  * scipy
  * scikit-learn
  * joblib
  * matplotlib

---



