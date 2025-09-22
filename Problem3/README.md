# 🔐 Side-Channel Analysis (AES Key Recovery with Deep Learning)

This project demonstrates a **Side-Channel Analysis (SCA) attack** on AES using **power/EM traces** and a **neural network**.  
The pipeline is implemented in **Google Colab (Python + TensorFlow)**.

---

## 📂 Project Structure

```

.
├── sca\_attack\_colab.py     # Colab-ready script (training + attack + plots)
├── datasetB/               # Profiling (training) traces and labels
│   ├── trace.npy
│   ├── plaintext.npy
│   └── key.npy
├── datasetA/               # Attack traces and plaintexts
│   ├── trace.npy
│   └── plaintext.npy
└── trained\_sca\_model.h5    # Saved trained model

```

---

## 🚀 Steps

### 1. Training (Dataset B)

- Train a neural network on profiling traces from **Device B**.
- The NN learns to classify the **Hamming Weight of the AES S-box output**.
- Architecture:
  - Input layer: length = trace length
  - Dense(64, ReLU)
  - Dense(64, ReLU)
  - Dense(9, softmax)

**Training logs (sample run):**

```

Epoch 18/20
98/98 - 0s - 2ms/step - accuracy: 0.4291 - loss: 1.3495
Epoch 19/20
98/98 - 0s - 3ms/step - accuracy: 0.4344 - loss: 1.3408
Epoch 20/20
98/98 - 0s - 2ms/step - accuracy: 0.4306 - loss: 1.3381

✅ Model saved to /content/drive/MyDrive/trained\_sca\_model.h5

```

---

### 2. Attack (Dataset A)

- Use the trained model to classify **attack traces** from **Device A**.
- Compute log-likelihoods for all **256 key guesses**.
- Rank candidates.

**Top 5 most likely key bytes (example run):**

```

\--- Attack (Dataset A) ---

Top 5 most likely key bytes:
Rank 1: 0xfa (Log-Likelihood: -35.19)
Rank 2: 0x64 (Log-Likelihood: -39.51)
Rank 3: 0x60 (Log-Likelihood: -39.65)
Rank 4: 0xdc (Log-Likelihood: -41.21)
Rank 5: 0xba (Log-Likelihood: -41.52)

```

---

## 📊 Visualizations

The script also generates plots:
- Training **accuracy** and **loss** curves
- Distribution of **log-likelihoods across all 256 keys**
- Bar chart of **Top-10 key candidates**

(Example visualization)

<img width="1018" height="470" alt="image" src="https://github.com/user-attachments/assets/cf089a91-2537-467d-9c7a-5011e837ddfd" />
<img width="1017" height="547" alt="image" src="https://github.com/user-attachments/assets/4e4a62d9-1f9e-4118-89ce-6a1735a289d1" />
<img width="853" height="451" alt="image" src="https://github.com/user-attachments/assets/e70a6955-7051-48eb-b4ec-d7c1275b6a60" />



---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Google Colab (for Drive mounting, optional)

---
