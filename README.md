# 🚀 Bit-by-Bit Hackathon – Side-Channel Analysis Project

This repository contains our solution for the **Bit-by-Bit Hackathon**.  
We implemented **Side-Channel Analysis (SCA)** techniques to recover AES secret keys using power/EM traces.  
The project includes scripts for **trace collection, training, prediction, evaluation, and attack execution** using deep learning models (MLP, AlexNet, MobileNet, etc.).

---

## 📂 Solution for Problem 1
<img width="1060" height="705" alt="image" src="https://github.com/user-attachments/assets/d13e9af2-21eb-4f90-bb76-5b0d2af9f72e" />

### Final Answer => 110011011110001 (based on known data in given problem statement MSB = 1 and LSB = 0001


## 📂 Solution for Problem 2
<img width="607" height="466" alt="image" src="https://github.com/user-attachments/assets/da9aacac-aab0-4769-bd4d-afc4dd96ce1b" />

### Final Answer => we are a bit unsure because of class probablity but we can definetly gaurantee that the answer is either "alexnet" or ""mobilenet_v2


## 📂 Solution for Problem 3
<img width="853" height="451" alt="image" src="https://github.com/user-attachments/assets/dab92c87-7db6-483c-8886-6a7bd78c0195" />

### Final Answer => Top 5 most likely key bytes:
```
  Rank 1: 0xfa (Log-Likelihood: -35.19)
  Rank 2: 0x64 (Log-Likelihood: -39.51)
  Rank 3: 0x60 (Log-Likelihood: -39.65)
  Rank 4: 0xdc (Log-Likelihood: -41.21)
  Rank 5: 0xba (Log-Likelihood: -41.52)
```
