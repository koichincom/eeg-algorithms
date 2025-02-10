# Non-Machine Learning Algorithms for EEG Processing

This document provides an in-depth overview of traditional signal processing methods used in EEG analysis. These methods are essential for spectral analysis, time-frequency representation, artifact removal, source localization, and the study of nonlinear dynamics in EEG signals. Mastering these techniques will allow you to understand the underlying principles and choose the appropriate method for your application.

---

## 1. Fourier Transform (FT)

### Objective
Decompose a time-domain EEG signal into its constituent frequencies to perform spectral analysis.

### Explanation
The Fourier Transform (FT) converts a time-domain signal into its frequency-domain representation. This transformation helps identify oscillatory components (such as alpha, beta, gamma rhythms) in EEG data.

### Mathematical Formulation

- **Continuous Fourier Transform:**

  $$
  X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j 2\pi f t} \, dt
  $$
  
- **Inverse Fourier Transform:**

  $$
  x(t) = \int_{-\infty}^{\infty} X(f) \, e^{j 2\pi f t} \, df
  $$
  
- **Discrete Fourier Transform (DFT):**
  $$
  X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1
  $$
  
- **Inverse DFT:**
  $$
  x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{j \frac{2\pi}{N} k n}
  $$

---

## 2. Short-Time Fourier Transform (STFT)

### Objective
Analyze non-stationary EEG signals by providing a time-frequency representation.

### Explanation
The STFT applies the Fourier Transform over short, sliding windows of the signal. This reveals how the frequency content of EEG signals changes over time.

### Mathematical Formulation
$$
X(t, f) = \int_{-\infty}^{\infty} x(\tau) \, w(\tau - t) \, e^{-j 2\pi f \tau} \, d\tau
$$
where:
- \( w(\tau - t) \) is a window function centered at time \( t \).

---

## 3. Wavelet Transform

### Objective
Provide multi-resolution time-frequency analysis, capturing both transient and steady-state features of EEG signals.

### Explanation
Wavelet Transforms use localized waveforms (wavelets) rather than infinite-duration sinusoids. This is especially useful for analyzing transient events or changes in frequency content over time.

### Mathematical Formulation

- **Continuous Wavelet Transform (CWT):**
  $$
  W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\!\left(\frac{t - b}{a}\right) dt
  $$
  where:
  - \( a \) is the scale parameter (inversely related to frequency),
  - \( b \) is the translation parameter (time shift),
  - \( \psi(t) \) is the mother wavelet,
  - \( \psi^* \) denotes the complex conjugate.

- **Discrete Wavelet Transform (DWT):**
  In practice, DWT is implemented using filter banks and downsampling.

---

## 4. Digital Filtering Techniques

### Objective
Remove noise and isolate frequency bands of interest from EEG signals.

### Explanation
Digital filters (e.g., low-pass, high-pass, band-pass, band-stop) are designed to selectively attenuate unwanted frequency components while preserving those that carry useful information.

### Mathematical Formulation

- **General IIR Filter Difference Equation:**
  $$
  y[n] = \sum_{k=0}^{M} b_k \, x[n-k] - \sum_{k=1}^{N} a_k \, y[n-k]
  $$
  where:
  - \( x[n] \) is the input signal,
  - \( y[n] \) is the filtered output,
  - \( \{b_k\} \) and \( \{a_k\} \) are the filter coefficients.

- **Example: Butterworth Filter**
  
  A Butterworth filter has a maximally flat frequency response in the passband. Its transfer function (in the Laplace domain) is given by:
  $$
  H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}
  $$
  where:
  - \( \omega_c \) is the cutoff frequency,
  - \( n \) is the filter order.

---

## 5. Autoregressive (AR) Modeling

### Objective
Model EEG signals as a function of past values to perform spectral estimation and signal prediction.

### Explanation
An AR model assumes that each EEG sample is a linear combination of previous samples plus white noise. This is useful for high-resolution spectral estimation.

### Mathematical Formulation
$$
x(t) = \sum_{k=1}^{p} a_k \, x(t-k) + e(t)
$$
where:
- \( p \) is the model order,
- \( \{a_k\} \) are the AR coefficients,
- \( e(t) \) is the prediction error (assumed to be white noise).

The coefficients can be estimated using the Yule-Walker equations:
$$
r(m) = \sum_{k=1}^{p} a_k \, r(m-k) + \sigma_e^2 \, \delta(m)
$$
with \( r(m) \) being the autocorrelation function and \( \delta(m) \) the Kronecker delta.

---

## 6. Independent Component Analysis (ICA)

### Objective
Separate mixed EEG signals into statistically independent components, often to remove artifacts.

### Explanation
ICA assumes that the observed EEG signals are linear mixtures of independent source signals (e.g., neural activity, eye blinks). It computes an unmixing matrix that maximizes the statistical independence of the recovered components.

### Mathematical Formulation
Given:
\[
\mathbf{x} = \mathbf{A} \, \mathbf{s}
\]
where:
- \( \mathbf{x} \) is the observed signal vector,
- \( \mathbf{s} \) is the vector of independent source signals,
- \( \mathbf{A} \) is the mixing matrix.

The goal is to find the unmixing matrix \( \mathbf{W} \) such that:
\[
\mathbf{s} = \mathbf{W} \, \mathbf{x}
\]
Algorithms like **FastICA** maximize the non-Gaussianity (e.g., kurtosis) of \( \mathbf{s} \) to achieve separation.

---

## 7. Principal Component Analysis (PCA)

### Objective
Reduce the dimensionality of EEG data and identify major patterns of variance.

### Explanation
PCA transforms EEG data into a new coordinate system where the axes (principal components) capture the directions of maximum variance. This simplifies further analysis and visualization.

### Mathematical Formulation

1. **Compute the Covariance Matrix:**
   \[
   \mathbf{C} = \frac{1}{N-1} \mathbf{X}^\top \mathbf{X}
   \]
   where \( \mathbf{X} \) is the zero-mean data matrix.

2. **Eigenvalue Decomposition:**
   \[
   \mathbf{C} \, \mathbf{p} = \lambda \, \mathbf{p}
   \]
   where:
   - \( \lambda \) are the eigenvalues,
   - \( \mathbf{p} \) are the corresponding eigenvectors (principal components).

3. **Projection:**
   \[
   \mathbf{T} = \mathbf{X} \, \mathbf{P}
   \]
   where \( \mathbf{T} \) are the transformed (projected) data.

---

## 8. Empirical Mode Decomposition (EMD)

### Objective
Adaptively decompose a non-linear, non-stationary EEG signal into a set of intrinsic mode functions (IMFs).

### Explanation
EMD iteratively extracts oscillatory modes (IMFs) from a signal by identifying local extrema, computing envelopes, and subtracting the local mean.

### Algorithm Outline

1. **Identify Local Extrema:** Find all local maxima and minima in \( x(t) \).
2. **Envelope Construction:** Interpolate the maxima to form the upper envelope \( e_{\text{max}}(t) \) and the minima for the lower envelope \( e_{\text{min}}(t) \).
3. **Compute the Mean:**
   \[
   m(t) = \frac{e_{\text{max}}(t) + e_{\text{min}}(t)}{2}
   \]
4. **Extract the Detail (Candidate IMF):**
   \[
   h(t) = x(t) - m(t)
   \]
5. **Check IMF Criteria:** If \( h(t) \) satisfies the criteria, designate \( h(t) \) as an IMF. Otherwise, repeat the sifting process.
6. **Subtract and Iterate:** Subtract the IMF from the original signal and repeat.

---

## 9. Hilbert-Huang Transform (HHT)

### Objective
Perform time-frequency analysis on non-linear and non-stationary EEG signals.

### Explanation
HHT combines EMD with the Hilbert Transform to obtain instantaneous frequency and amplitude information.

### Mathematical Formulation

- **Hilbert Transform:**
  \[
  \hat{x}(t) = \frac{1}{\pi} \, \text{p.v.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} \, d\tau
  \]

- **Analytic Signal:**
  \[
  z(t) = x(t) + j \, \hat{x}(t) = A(t) \, e^{j\phi(t)}
  \]
  where:
  - \( A(t) \) is the instantaneous amplitude,
  - \( \phi(t) \) is the instantaneous phase.

---

## 10. Cross-Correlation Analysis

### Objective
Assess the similarity and time-lag relationships between two EEG signals.

### Mathematical Formulation

- **Continuous Form:**
  \[
  R_{xy}(\tau) = \int_{-\infty}^{\infty} x(t) \, y(t+\tau) \, dt
  \]

- **Discrete Form:**
  \[
  R_{xy}[m] = \sum_{n} x[n] \, y[n+m]
  \]

---

## 11. Coherence Analysis

### Objective
Measure the frequency-domain correlation (functional connectivity) between two EEG signals.

### Mathematical Formulation
\[
C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f) \, P_{yy}(f)}
\]
where:
- \( P_{xy}(f) \) is the cross-spectral density,
- \( P_{xx}(f) \) and \( P_{yy}(f) \) are the power spectral densities.

---

## 12. Source Localization Methods

### 12.1. Dipole Fitting

#### Objective
Estimate the location, orientation, and strength of equivalent current dipoles generating EEG signals.

### Mathematical Formulation
\[
\mathbf{V} = \mathbf{L} \, \mathbf{J} + \text{noise}
\]
where:
- \( \mathbf{V} \) is the vector of measured potentials,
- \( \mathbf{L} \) is the lead field matrix,
- \( \mathbf{J} \) is the dipole moment vector.

---

### 12.2. Beamforming

#### Objective
Estimate activity from a specific brain location by applying spatial filters that suppress interference from other sources.

### Mathematical Formulation

A common beamformer is the Linearly Constrained Minimum Variance (LCMV) beamformer:
\[
\mathbf{w} = \frac{\mathbf{C}^{-1}\mathbf{L}}{\mathbf{L}^\top \mathbf{C}^{-1} \mathbf{L}}
\]
where:
- \( \mathbf{w} \) is the beamforming weight vector,
- \( \mathbf{C} \) is the covariance matrix of EEG data,
- \( \mathbf{L} \) is the lead field vector.

---

## 13. Phase Synchronization Analysis

### Objective
Quantify the degree of phase coupling between EEG signals.

### Mathematical Formulation

A common measure is the **Phase Locking Value (PLV):**
\[
\text{PLV} = \left|\frac{1}{N} \sum_{n=1}^{N} e^{j \Delta \phi(n)}\right|
\]
where:
- \( \Delta \phi(n) = \phi_x(n) - \phi_y(n) \) is the phase difference,
- \( N \) is the total number of time points.

---

## 14. Nonlinear Dynamics and Chaos Analysis

### Objective
Characterize the complex, nonlinear behavior of EEG signals.

### Examples and Mathematical Formulations

#### 14.1. Lyapunov Exponents
Measure the rate at which nearby trajectories in a reconstructed phase space diverge. A positive largest Lyapunov exponent indicates chaos.

#### 14.2. Fractal Dimension (e.g., Correlation Dimension)
Estimates the dimensionality of the signal’s attractor. The Grassberger–Procaccia algorithm relates the correlation sum \( C(r) \) to the radius \( r \) by:
\[
C(r) \sim r^{D_2}
\]
where \( D_2 \) is the correlation dimension.

#### 14.3. Entropy Measures
- **Approximate Entropy (ApEn)**
- **Sample Entropy (SampEn)**

These measures assess the regularity and unpredictability of a time series.

---

## Summary: Non-Machine Learning Algorithms for EEG Processing

### 1. Spectral and Time-Frequency Analysis
- **Fourier Transform (FT):** Decomposes EEG into frequency components.
- **Short-Time Fourier Transform (STFT):** Provides time-frequency representation for non-stationary signals.
- **Wavelet Transform (WT):** Multi-resolution analysis capturing transient EEG features.

### 2. Filtering and Modeling
- **Digital Filtering:** Isolates EEG frequency bands and removes noise.
- **Autoregressive (AR) Modeling:** Predicts EEG signals and improves spectral resolution.

### 3. Blind Source Separation and Dimensionality Reduction
- **Independent Component Analysis (ICA):** Separates EEG into independent sources, useful for artifact removal.
- **Principal Component Analysis (PCA):** Reduces dimensionality and extracts major EEG variance patterns.

### 4. Adaptive Decomposition Methods
- **Empirical Mode Decomposition (EMD):** Decomposes EEG into intrinsic mode functions (IMFs).
- **Hilbert-Huang Transform (HHT):** Extracts instantaneous frequency and amplitude features.

### 5. Functional Connectivity and Synchronization Analysis
- **Cross-Correlation:** Measures similarity and time-lag relationships between EEG signals.
- **Coherence Analysis:** Evaluates frequency-domain connectivity between brain regions.
- **Phase Synchronization (PLV):** Quantifies phase coupling in EEG signals.

### 6. Source Localization
- **Dipole Fitting:** Estimates EEG source location and orientation.
- **Beamforming (LCMV):** Uses spatial filtering to enhance EEG source estimation.

### 7. Nonlinear Dynamics and Chaos Analysis
- **Lyapunov Exponents:** Measures signal predictability and chaotic behavior.
- **Fractal Dimension (Correlation Dimension):** Assesses EEG complexity.
- **Entropy Measures (ApEn, SampEn):** Quantifies EEG irregularity and unpredictability.
