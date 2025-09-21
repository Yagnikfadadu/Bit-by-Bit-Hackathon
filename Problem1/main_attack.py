#!/usr/bin/env python3
"""
CPA attack script — LEFT-TO-RIGHT (square then multiply) only.

Plots for each recovered bit show:
 - correlation trace for guess=0 (blue) and guess=1 (red)
 - peak point for each guess marked and labeled
 - horizontal line at y=0 and vertical center marker
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Optional imports (graceful fallback)
try:
    from scipy.signal import butter, filtfilt, medfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# --- 1. CONFIGURATION ---
TRACE_FOLDER = "./"              # Folder containing your CSV
TRACE_FILENAME = "single_trace.csv"
RSA_N = 64507                    # Modulus used in your simulation
KEY_BITS = 15                    # Number of bits to recover
SPIKE_WIDTH = 500                # Width around detected spike to select window
verbose_plot = True              # True => show per-bit plots
BANDPASS_ON = False              # True => apply bandpass filter (if scipy available)
BP_LOW = 0.001
BP_HIGH = 0.4
BP_ORDER = 4

# --- 2. HELPERS ---
def hamming_weight(n: int) -> int:
    return bin(int(n) & 0xFFFFFFFFFFFFFFFF).count("1")

def load_traces(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ciphertexts, traces = [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        first = next(reader)
        # Heuristic to detect header
        try:
            int(first[0], 16)
            row0 = first
        except Exception:
            row0 = next(reader)
        ciphertexts.append(int(row0[0], 16))
        traces.append(np.array([float(x) for x in row0[1:]], dtype=np.float64))
        for row in reader:
            if not row:
                continue
            ciphertexts.append(int(row[0], 16))
            traces.append(np.array([float(x) for x in row[1:]], dtype=np.float64))
    return np.array(ciphertexts, dtype=np.int64), np.vstack(traces)

def baseline_remove(traces: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    if SCIPY_AVAILABLE:
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        out = np.empty_like(traces)
        for i in range(traces.shape[0]):
            out[i] = traces[i] - medfilt(traces[i], kernel_size=k)
        return out
    else:
        return traces - np.mean(traces, axis=1, keepdims=True)

def bandpass_filter(traces: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        print("scipy not available: skipping bandpass filter.")
        return traces
    b, a = butter(order, [low, high], btype='band')
    filtered = np.empty_like(traces)
    for i in range(traces.shape[0]):
        filtered[i] = filtfilt(b, a, traces[i])
    return filtered

def detect_spike(traces: np.ndarray, width: int) -> Tuple[int, int, int]:
    avg_trace = np.mean(traces, axis=0)
    spike_index = int(np.argmax(np.abs(avg_trace - np.mean(avg_trace))))
    half = width // 2
    start = max(0, spike_index - half)
    end = min(traces.shape[1], spike_index + half)
    return start, end, spike_index

def align_traces(traces: np.ndarray, ref_index: int = 0, search_radius: int = 50,
                 spike_index: int = None) -> np.ndarray:
    n_traces, n_samples = traces.shape
    if spike_index is None:
        avg = np.mean(traces, axis=0)
        spike_index = int(np.argmax(np.abs(avg - np.mean(avg))))
    ref_trace = traces[ref_index]
    left = max(0, spike_index - search_radius)
    right = min(n_samples, spike_index + search_radius)
    ref_peak = int(np.argmax(np.abs(ref_trace[left:right])) + left)
    aligned = np.zeros_like(traces)
    for i in range(n_traces):
        tr = traces[i]
        left_i = max(0, spike_index - search_radius)
        right_i = min(n_samples, spike_index + search_radius)
        local_peak = int(np.argmax(np.abs(tr[left_i:right_i])) + left_i)
        shift = ref_peak - local_peak
        if shift == 0:
            aligned[i] = tr
        elif shift > 0:
            aligned[i, shift:] = tr[:n_samples - shift]
        else:
            s = -shift
            aligned[i, :n_samples - s] = tr[s:]
    return aligned

def compute_samplewise_correlation(hyp: np.ndarray, traces_window: np.ndarray) -> np.ndarray:
    hyp = hyp.astype(np.float64)
    N = hyp.shape[0]
    if N < 2:
        return np.zeros(traces_window.shape[1])
    hyp_mean = hyp.mean()
    hyp_std = hyp.std(ddof=0) + 1e-12
    hyp_norm = (hyp - hyp_mean) / hyp_std
    tw_mean = np.mean(traces_window, axis=0)
    tw_std = np.std(traces_window, axis=0) + 1e-12
    tw_norm = (traces_window - tw_mean) / tw_std
    return np.dot(hyp_norm, tw_norm) / float(N)

# --- 3. LTR hypothesis builder (square then multiply) ---
def build_hypothesis_for_guess_ltr(bit_index: int, guess: int, ciphertexts: np.ndarray,
                                   recovered_bits: list, rsa_n: int) -> np.ndarray:
    """
    Left-to-right (MSB-first) : square then multiply.
    recovered_bits is assumed to store bits in device order (0..bit_index-1).
    We simulate state AFTER processing the current bit (include square+possible multiply for current guess).
    """
    hyp = []
    for c in ciphertexts:
        S = 1
        for j in range(bit_index):
            S = (S * S) % rsa_n
            if recovered_bits[j] == '1':
                S = (S * c) % rsa_n
        # now process current bit (simulate the operation that produces leakage)
        S = (S * S) % rsa_n
        if guess == 1:
            S = (S * c) % rsa_n
        hyp.append(hamming_weight(S))
    return np.array(hyp, dtype=np.float64)

# high-level attack runner (LTR only)
def run_cpa_ltr(ciphertexts: np.ndarray, trace_window: np.ndarray, rsa_n: int,
                key_bits: int, verbose_plot: bool = True):
    recovered_bits = []
    cumulative_peak = 0.0
    per_bit_peaks = []

    for bit_index in range(key_bits):
        best_guess = 0
        best_peak = -np.inf
        best_corr_vector = None

        # compute both guesses and compare absolute peak magnitudes
        guess_corrs = {}
        guess_peaks = {}
        for guess in (0, 1):
            hyp = build_hypothesis_for_guess_ltr(bit_index, guess, ciphertexts, recovered_bits, rsa_n)
            corr = compute_samplewise_correlation(hyp, trace_window)
            peak = float(np.max(np.abs(corr)))
            guess_corrs[guess] = corr
            guess_peaks[guess] = peak

            if peak > best_peak:
                best_peak = peak
                best_guess = guess
                best_corr_vector = corr.copy()

        recovered_bits.append(str(best_guess))
        cumulative_peak += best_peak
        per_bit_peaks.append(best_peak)

        # print per-guess peaks for clarity
        print(f"[ltr] Bit {bit_index + 1}/{key_bits} -> chosen={best_guess} (peak={best_peak:.6f})",
              f"  peaks: guess0={guess_peaks[0]:.6f}, guess1={guess_peaks[1]:.6f}")

        # plotting both guesses and marking their peaks
        if verbose_plot:
            plt.figure(figsize=(9, 2.8))
            colors = {0: "blue", 1: "red"}
            labels = {0: "guess=0", 1: "guess=1"}
            for guess in (0, 1):
                corr = guess_corrs[guess]
                peak_idx = int(np.argmax(np.abs(corr)))
                peak_val = corr[peak_idx]
                plt.plot(corr, color=colors[guess], alpha=0.6, label=f"{labels[guess]}")
                plt.plot(peak_idx, peak_val, "o", color=colors[guess], markersize=6,
                         label=f"{labels[guess]} peak={peak_val:.3f}")

            plt.title(f"ltr - Bit {bit_index + 1} chosen={best_guess}")
            plt.axvline(trace_window.shape[1] // 2, linestyle=":", color="gray", label="window center")
            plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
            plt.xlabel("Window sample index")
            plt.ylabel("Pearson corr")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

    key_bin = "".join(recovered_bits)
    key_dec = int(key_bin, 2) if key_bin else None
    return {
        'convention': 'ltr',
        'key_bin': key_bin,
        'key_dec': key_dec,
        'cumulative_peak': cumulative_peak,
        'per_bit_peaks': per_bit_peaks
    }

# --- 4. MAIN PROCESS ---
TRACE_FILE = os.path.join(TRACE_FOLDER, TRACE_FILENAME)
if not os.path.isfile(TRACE_FILE):
    raise FileNotFoundError(f"Trace file not found: {TRACE_FILE}")

print(f"\n--- Processing {TRACE_FILE} ---")
ciphertexts, traces = load_traces(TRACE_FILE)
num_traces, num_samples = traces.shape
print(f"Loaded {num_traces} traces, each with {num_samples} samples.")

if num_traces < 2:
    print("Too few traces for reliable CPA. Exiting.")
    exit(0)

# preprocessing
traces_br = baseline_remove(traces)
traces_f = bandpass_filter(traces_br, BP_LOW, BP_HIGH, BP_ORDER) if BANDPASS_ON else traces_br
start, end, spike_index = detect_spike(traces_f, SPIKE_WIDTH)
traces_aligned = align_traces(traces_f, ref_index=0, search_radius=200, spike_index=spike_index)
print('Preprocessed traces shape (after baseline & alignment):', traces_aligned.shape)

# normalize per-sample
sample_mean = np.mean(traces_aligned, axis=0)
sample_std = np.std(traces_aligned, axis=0) + 1e-12
traces_norm = (traces_aligned - sample_mean) / sample_std

# window selection
trace_window = traces_norm[:, start:end]
if trace_window.shape[1] <= 2:
    trace_window = traces_norm
print(f"Spike window used for CPA: samples {start}..{end} (width {trace_window.shape[1]})")

if verbose_plot:
    plt.figure(figsize=(8, 3))
    plt.plot(np.mean(trace_window, axis=0))
    plt.title(f"Detected Multiply Spike Window for file {TRACE_FILENAME}")
    plt.xlabel("Sample Index (window)")
    plt.ylabel("Amplitude (normalized)")
    plt.show()

# Run CPA (LTR only)
result_ltr = run_cpa_ltr(ciphertexts, trace_window, RSA_N, KEY_BITS, verbose_plot=verbose_plot)

print("\n--- Summary ---")
print(f"LTR cumulative peak sum: {result_ltr['cumulative_peak']:.6f}, recovered key: {result_ltr['key_bin']} -> {result_ltr['key_dec']}")
print(f"[+] Recovered key (binary): {result_ltr['key_bin']}")
print(f"[+] Recovered key (decimal): {result_ltr['key_dec']}")
