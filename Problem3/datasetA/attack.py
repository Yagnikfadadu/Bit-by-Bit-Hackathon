import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# AES S-box and Hamming Weight lookup table (from the provided script)
sbox = (
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16)
hw = [bin(x).count("1") for x in range(256)]

def create_nn_model(input_shape, num_classes):
    """Defines the neural network architecture."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for probability distribution output
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- STEP 1: TRAINING PHASE (Using Dataset B) ---
print("--- Starting Step 1: Training the Neural Network ---")

# 1.1: Load the profiling data from Device B
try:
    prof_traces = np.load('datasetB/trace.npy')
    prof_plaintexts = np.load('datasetB/plaintext.npy')
    prof_keys = np.load('datasetB/key.npy')
except FileNotFoundError:
    print("Error: Dataset B files not found. Ensure trace.npy, plaintext.npy, and key.npy are in the 'datasetB' folder.")
    exit()

# 1.2: Calculate the intermediate value (labels) for each trace
# We use Z = HW(Sbox(P ⊕ K)) as the sensitive variable
num_prof_traces = len(prof_traces)
labels = np.zeros(num_prof_traces)
for i in range(num_prof_traces):
    sbox_out = sbox[prof_plaintexts[i][0] ^ prof_keys[i][0]]
    labels[i] = hw[sbox_out]

# Convert labels to one-hot encoding for the neural network
num_classes = 9 # HW can be from 0 to 8
labels_categorical = to_categorical(labels, num_classes=num_classes)

# 1.3: Create and train the model
trace_len = len(prof_traces[0])
model = create_nn_model(trace_len, num_classes)

print("\nModel Architecture:")
model.summary()

print("\nTraining the model...")
model.fit(prof_traces, labels_categorical, epochs=20, batch_size=256, verbose=2)

# 1.4: Save the trained model (a required deliverable)
model.save('trained_sca_model.h5')
print("\n✅ Training complete. Model saved as 'trained_sca_model.h5'.")


# --- STEP 2: KEY RECOVERY PHASE (Attacking Device A) ---
print("\n--- Starting Step 2: Key Recovery Attack ---")

# 2.1: Load the attack data from Device A
try:
    attack_traces = np.load('datasetA/trace.npy')
    attack_plaintexts = np.load('datasetA/plaintext.npy')
except FileNotFoundError:
    print("Error: Dataset A files not found. Ensure trace.npy and plaintext.npy are in the 'datasetA' folder.")
    exit()

num_attack_traces = len(attack_traces)
print(f"Loaded {num_attack_traces} traces from the target device for the attack.")

# 2.2: Use the trained model to get probability distributions for attack traces
print("Predicting probabilities for attack traces...")
# The model.predict() function returns a matrix where each row is the probability
# distribution for the corresponding trace.
predicted_probabilities = model.predict(attack_traces)

# 2.3: Compute log-likelihood score for each key guess
log_likelihoods = np.zeros(256)

for k_guess in range(256): # Iterate through all 256 possible key bytes
    for i in range(num_attack_traces):
        plaintext_byte = attack_plaintexts[i][0]
        
        # Calculate the hypothetical intermediate value for this key guess
        sbox_out = sbox[plaintext_byte ^ k_guess]
        hypothetical_hw = hw[sbox_out]
        
        # Get the model's predicted probability for this hypothetical HW
        prob_of_hw = predicted_probabilities[i][hypothetical_hw]
        
        # Add its logarithm to the total score for this key guess
        # Add a small epsilon to prevent log(0)
        log_likelihoods[k_guess] += np.log(prob_of_hw + 1e-30)

# 2.4: Rank the keys based on the final scores
# argsort() gives the indices that would sort the array in ascending order.
# We use [::-1] to reverse it for descending order (highest score first).
ranked_keys = np.argsort(log_likelihoods)[::-1]

# --- FINAL RESULTS ---
print("\n--- Attack Results ---")
print("✅ Key recovery complete. Displaying the most likely keys.")
print("Sorted list of possible keys (most likely first):")
# Convert keys to hexadecimal for readability
ranked_keys_hex = [f"0x{key:02x}" for key in ranked_keys]
print(ranked_keys_hex)

print("\nTop 5 most likely key bytes:")
for i in range(5):
    key_val = ranked_keys[i]
    score = log_likelihoods[key_val]
    print(f"  Rank {i+1}: Key = 0x{key_val:02x} (Log-Likelihood: {score:.2f})")