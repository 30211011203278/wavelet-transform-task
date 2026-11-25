
import numpy as np
import matplotlib.pyplot as plt
import pywt

fs = 1000                          
t = np.arange(0, 1, 1 / fs)        
rng = np.random.default_rng(0)     
sig_a = np.where(t < 0.5, np.sin(2 * np.pi * 10 * t),
                  np.sin(2 * np.pi * 50 * t))
sig_b = np.where(t < 0.5, np.sin(2 * np.pi * 50 * t),
                  np.sin(2 * np.pi * 10 * t))
def fft_magnitude(x):
    "Return frequencies ."
    spectrum = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, spectrum
def cwt_scalogram(x):
    "Continuous Wavelet."
    freqs = np.arange(1, 101)                       
    scales = pywt.central_frequency("morl") * fs / freqs
    coefs, _ = pywt.cwt(x, scales, "morl", sampling_period=1 / fs)
    return freqs, np.abs(coefs)
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for col, (sig, name) in enumerate([(sig_a, "Signal A (10Hz -> 50Hz)"),
                                   (sig_b, "Signal B (50Hz -> 10Hz)")]):
    axes[0, col].plot(t, sig)
    axes[0, col].set_title(name)
    axes[0, col].set_xlabel("Time (s)")

    f, mag = fft_magnitude(sig)
    axes[1, col].plot(f, mag)
    axes[1, col].set_xlim(0, 100)
    axes[1, col].set_title("Fourier: frequencies only, no time info")
    axes[1, col].set_xlabel("Frequency (Hz)")

    cf, power = cwt_scalogram(sig)
    axes[2, col].pcolormesh(t, cf, power, shading="auto", cmap="jet")
    axes[2, col].set_title("Wavelet (CWT): time + frequency")
    axes[2, col].set_xlabel("Time (s)")
    axes[2, col].set_ylabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig("part1_fourier_vs_cwt.png", dpi=150)
wavelet = "db4"                    
level = 4
sig = (np.sin(2 * np.pi * 5 * t)
       + 0.5 * np.sin(2 * np.pi * 60 * t) * (t > 0.5)
       + 0.2 * rng.standard_normal(len(t)))
coeffs = pywt.wavedec(sig, wavelet, level=level)
fig, axes = plt.subplots(level + 2, 1, figsize=(10, 10))
axes[0].plot(sig)
axes[0].set_title("Original signal")
axes[1].plot(coeffs[0])
axes[1].set_title(f"Approximation cA{level} (slow / general shape)")
for i, d in enumerate(coeffs[1:]):
    lvl = level - i
    axes[i + 2].plot(d)
    axes[i + 2].set_title(f"Detail cD{lvl} (fast changes / noise)")
plt.tight_layout()
plt.savefig("part2_dwt_decomposition.png", dpi=150)
clean = (np.sin(2 * np.pi * 5 * t)
         + np.where((t > 0.6) & (t < 0.65), 1.5, 0.0))
noisy = clean + 0.4 * rng.standard_normal(len(t))
coeffs = pywt.wavedec(noisy, wavelet, level=level)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745        
threshold = sigma * np.sqrt(2 * np.log(len(noisy)))    
coeffs_thr = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft")
                            for c in coeffs[1:]]
wavelet_denoised = pywt.waverec(coeffs_thr, wavelet)[:len(noisy)]
cutoff = 20                                           
spec = np.fft.rfft(noisy)
freqs = np.fft.rfftfreq(len(noisy), 1 / fs)
spec[freqs > cutoff] = 0
fourier_denoised = np.fft.irfft(spec, n=len(noisy))
def mse(a, b):
    return np.mean((a - b) ** 2)
print(f"MSE noisy            : {mse(clean, noisy):.4f}")
print(f"MSE Fourier low-pass : {mse(clean, fourier_denoised):.4f}")
print(f"MSE Wavelet denoise  : {mse(clean, wavelet_denoised):.4f}")
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, clean)
axes[0].set_title("Clean signal")
axes[1].plot(t, noisy)
axes[1].set_title("Noisy signal")
axes[2].plot(t, fourier_denoised)
axes[2].set_title(f"Fourier low-pass ({cutoff} Hz) - MSE={mse(clean, fourier_denoised):.4f}")
axes[3].plot(t, wavelet_denoised)
axes[3].set_title(f"Wavelet denoising ({wavelet}) - MSE={mse(clean, wavelet_denoised):.4f}")
axes[3].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("part3_denoising.png", dpi=150)
plt.show()