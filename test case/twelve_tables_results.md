# Empirical Benchmark Report: 12 Base Paper Tables Reproduction
**Reference**: Ding et al., *"Deepface-Based Chaotic Image Encryption Using Key Optimization and Semi-Tensor Product Theory,"* IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 35, No. 7, pp. 6421–6434, July 2025.
**Project**: RGMCET ECE Batch 14 Capstone Project
**Output Folder**: `test case/`

---

## 📁 1. Directory of Used Face & Test Images
All input face images tested are stored in [`test case/used_faces/`](./used_faces/):
1. **Face 1**: [`used_faces/Face1_Aaron_Eckhart.jpg`](./used_faces/Face1_Aaron_Eckhart.jpg) (Identity: Aaron Eckhart)
2. **Face 2**: [`used_faces/Face2_Abdullah_Gul.jpg`](./used_faces/Face2_Abdullah_Gul.jpg) (Identity: Abdullah Gul)
3. **Face 3**: [`used_faces/Face3_Al_Pacino.jpg`](./used_faces/Face3_Al_Pacino.jpg) (Identity: Al Pacino)
4. **Face 4**: [`used_faces/Face4_Alan_Greenspan.jpg`](./used_faces/Face4_Alan_Greenspan.jpg) (Identity: Alan Greenspan)
5. **Face 5 (Unenrolled Probe)**: [`used_faces/Face5_Arnold_Schwarzenegger.jpg`](./used_faces/Face5_Arnold_Schwarzenegger.jpg) (Identity: Arnold Schwarzenegger)
6. **Standard Baboon**: [`used_faces/std_baboon_512x512.png`](./used_faces/std_baboon_512x512.png) (512x512 RGB)
7. **Standard Peppers**: [`used_faces/std_peppers_512x512.png`](./used_faces/std_peppers_512x512.png) (512x512 RGB)
8. **Standard 4.1.01**: [`used_faces/std_4_1_01_256x256.png`](./used_faces/std_4_1_01_256x256.png) (256x256 RGB)
9. **Standard 4.1.04**: [`used_faces/std_4_1_04_256x256.png`](./used_faces/std_4_1_04_256x256.png) (256x256 RGB)

All corresponding encrypted cipher images and decrypted images are stored in [`test case/encrypted_faces/`](./encrypted_faces/).

---

## 📊 2. The 12 Base Paper Tables Results

### TABLE I: DIFFERENCE BETWEEN DIFFERENT ALGORITHM
*Comparison of architectural components: ROI extraction, Semi-Tensor Product (STP) diffusion, chaotic system dimensionality, and key optimization.*

| Algorithm | ROI encryption | STP diffusion | Chaotic system | Key optimization |
| --- | --- | --- | --- | --- |
| Ref. [20] | ✕ | ✕ | 2D | ✕ |
| Ref. [21] | ✕ | ✕ | 1D | ✓ |
| Ref. [29] | ✓ | ✕ | 2D | ✕ |
| Ref. [27] | ✓ | ✓ | 2D | ✕ |
| Ours (Ding et al. / Batch 14) | ✓ | ✓ | 3D-CIMBA | ✓ (APSO/PSO) |

---

### TABLE II: FACE RECOGNITION AND MATCHING RESULTS
*DeepFace database verification of probe face images (Images 1 to 5) evaluated against gallery identities using Euclidean distance at threshold 0.50. 4 images match, 1 image is rejected as unenrolled.*

| Image | Target Identity | Matched Image | Euclidean Distance | Threshold | Result |
| --- | --- | --- | --- | --- | --- |
| image 1 | Aaron_Eckhart | Aaron_Eckhart | 0.0 | 0.5 | Match |
| image 2 | Abdullah_Gul | Abdullah_Gul | 0.0 | 0.5 | Match |
| image 3 | Al_Pacino | Al_Pacino | 0.0 | 0.5 | Match |
| image 4 | Alan_Greenspan | Alan_Greenspan | 0.0 | 0.5 | Match |
| image 5 | Not In Database | None | 1.2597 | 0.5 | None |

---

### TABLE III: ENTROPY OF TESTED IMAGES
*Shannon Information Entropy $H(x) = -\sum p(x_i) \log_2 p(x_i)$ evaluated on plain and encrypted images across Red, Green, Blue channels and mean. Theoretical ceiling is 8.0000.*

| Image | Image size | Original Red | Original Green | Original Blue | Original Mean | Encrypted Red | Encrypted Green | Encrypted Blue | Encrypted Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face1 | 105x105 | 7.6835 | 7.3735 | 7.2758 | 7.4442 | 7.9834 | 7.9811 | 7.9829 | 7.9825 |
| Face2 | 125x125 | 7.7254 | 7.6162 | 7.4596 | 7.6004 | 7.9883 | 7.9885 | 7.9884 | 7.9884 |
| Face3 | 108x108 | 7.6283 | 7.4481 | 7.1372 | 7.4045 | 7.9821 | 7.9841 | 7.9841 | 7.9834 |
| Face4 | 110x110 | 7.8043 | 7.5782 | 7.4439 | 7.6088 | 7.9833 | 7.9841 | 7.9841 | 7.9839 |
| baboon | 512x512 | 7.1406 | 6.8768 | 7.3905 | 7.136 | 7.9992 | 7.9993 | 7.9994 | 7.9993 |
| Peppers | 512x512 | 7.4243 | 7.4219 | 7.1222 | 7.3228 | 7.9993 | 7.9994 | 7.9993 | 7.9993 |
| 4.1.01 | 256x256 | 5.544 | 5.5498 | 5.5402 | 5.5447 | 7.9974 | 7.9973 | 7.997 | 7.9972 |
| 4.1.04 | 256x256 | 5.5501 | 5.5485 | 5.5454 | 5.548 | 7.9975 | 7.9972 | 7.9971 | 7.9973 |

---

### TABLE IV: CORRELATION BETWEEN ADJACENT PIXELS OF TESTED IMAGES
*Pearson correlation coefficients $\rho$ ($10^{-3}$) sampled across 3,000 adjacent pixel pairs along Horizontal, Vertical, and Diagonal orientations across R, G, B channels.*

| Image | Horizontal Red (10^-3) | Horizontal Green (10^-3) | Horizontal Blue (10^-3) | Vertical Red (10^-3) | Vertical Green (10^-3) | Vertical Blue (10^-3) | Diagonal Red (10^-3) | Diagonal Green (10^-3) | Diagonal Blue (10^-3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face1 | 19.6438 | -0.1071 | -11.1454 | 29.1521 | 7.9827 | -7.4385 | -29.516 | -0.5283 | -27.0058 |
| Face2 | -15.9527 | 2.3112 | -7.8017 | -12.0858 | 27.9248 | -18.0172 | -7.1667 | -26.7012 | 5.1007 |
| Face3 | -32.0341 | -1.8557 | 26.4363 | -14.5527 | 6.3129 | -14.0991 | -19.9236 | -11.9749 | -1.2947 |
| Face4 | -27.7101 | -13.4253 | 0.5091 | 0.7596 | -10.8038 | 4.2448 | -9.4049 | 37.5492 | -12.5627 |
| baboon | -20.1011 | -20.7591 | -14.56 | 29.5739 | 3.5163 | 14.4113 | -10.1706 | -39.3654 | 25.0281 |
| Peppers | 15.2409 | -5.5164 | 11.677 | -21.8199 | -0.0727 | -20.142 | -6.8283 | -5.0967 | 17.2536 |
| 4.1.01 | -2.6874 | -12.8149 | -14.5235 | -18.7149 | 0.9434 | -31.1451 | -8.3774 | -1.8576 | 3.5988 |
| 4.1.04 | 37.2594 | -36.4919 | -9.8453 | 12.4535 | -19.1687 | -21.6782 | 21.2348 | -24.3267 | 29.0962 |

---

### TABLE V: COMPARISON OF MEAN CORRELATION BETWEEN ADJACENT PIXELS
*Empirical mean adjacent pixel correlation comparison ($10^{-3}$) with published chaotic encryption literature.*

| Algorithm | Horizontal (10^-3) | Vertical (10^-3) | Diagonal (10^-3) |
| --- | --- | --- | --- |
| Ref. [20] | 0.849 | 0.68 | 0.276 |
| Ref. [27] | -0.1022 | 0.3399 | 0.2489 |
| Ref. [46] | 1.253 | 0.0896 | 0.0074 |
| Ref. [47] | -4.9 | 6.7 | 0.6 |
| Ref. [48] | -1.5 | 2.3 | 2.1 |
| Ref. [49] | 3.65 | 0.8233 | 3.36 |
| Ours (Ding et al. 2025) | -0.3514 | -0.5548 | 0.9452 |
| Ours (Empirical Test) | -5.5939 | -3.0193 | -4.3017 |

---

### TABLE VI: COMPARISON OF KEY SPACE AND ENTROPY
*Comparison of brute-force key space resistance (NIST standard $\ge 2^{256}$) and cipher entropy against literature.*

| Algorithm | Key space | Entropy Red | Entropy Green | Entropy Blue | Entropy Average |
| --- | --- | --- | --- | --- | --- |
| Ref. [20] | 2^256 | 7.9993 | 7.9994 | 7.9991 | 7.9992 |
| Ref. [27] | 2^256 | / | / | / | 7.9898 |
| Ref. [46] | 2^425 | 7.9912 | 7.9913 | 7.9914 | 7.9913 |
| Ref. [47] | 2^512 | 7.9939 | 7.9939 | 7.9939 | 7.9939 |
| Ref. [48] | 2^512 | 7.9023 | 7.9029 | 7.9023 | 7.9025 |
| Ref. [49] | 2^399 | 7.9987 | 7.9985 | 7.9984 | 7.9985 |
| Ours (Base Paper) | 10^128 ~= 2^425.2 | 7.9968 | 7.997 | 7.9968 | 7.9969 |
| Ours (Empirical Test) | 10^128 ~= 2^425.2 | 7.9913 | 7.9914 | 7.9915 | 7.9914 |

---

### TABLE VII: CRITICAL VALUES OF THE NPCR AND UACI
*Rigorous statistical critical values for NPCR and UACI at significance level $\alpha = 0.05$ across image and facial ROI dimensions, calculated via Equations (22)–(25).*

| Size | NPCR- (%) | UACI- (%) | UACI+ (%) | Ideal NPCR (%) | Ideal UACI (%) |
| --- | --- | --- | --- | --- | --- |
| 120x120 | 99.56 | 33.2399 | 33.6862 | 99.6094 | 33.463 |
| 125x125 | 99.562 | 33.2488 | 33.6772 | 99.6094 | 33.463 |
| 133x133 | 99.5648 | 33.2617 | 33.6644 | 99.6094 | 33.463 |
| 140x140 | 99.5671 | 33.2718 | 33.6543 | 99.6094 | 33.463 |
| 256x256 | 99.5862 | 33.3584 | 33.5676 | 99.6094 | 33.463 |
| 512x512 | 99.5978 | 33.4107 | 33.5153 | 99.6094 | 33.463 |

---

### TABLE VIII: NPCR AND UACI OF TESTED IMAGES
*Measured differential cryptanalysis metrics (Number of Pixels Change Rate and Unified Average Changing Intensity) after a 1-bit plaintext flip with plaintext-associated key derivation.*

| Image | NPCR Red (%) | NPCR Green (%) | NPCR Blue (%) | UACI Red (%) | UACI Green (%) | UACI Blue (%) | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Face1 | 99.5646 | 99.6644 | 99.6644 | 33.2672 | 33.5719 | 33.5965 | Pass |
| Face2 | 99.6544 | 99.584 | 99.648 | 33.5847 | 33.3666 | 33.4561 | Pass |
| Face3 | 99.5628 | 99.5971 | 99.6742 | 33.2383 | 33.2852 | 33.2557 | Pass |
| Face4 | 99.5537 | 99.6612 | 99.6116 | 33.4787 | 33.3283 | 33.5739 | Pass |
| baboon | 99.6124 | 99.6029 | 99.5956 | 33.5067 | 33.4522 | 33.4631 | Pass |
| Peppers | 99.6296 | 99.6262 | 99.6017 | 33.5461 | 33.4526 | 33.4071 | Pass |
| 4.1.01 | 99.6094 | 99.614 | 99.6277 | 33.5481 | 33.459 | 33.5141 | Pass |
| 4.1.04 | 99.6231 | 99.6017 | 99.585 | 33.6983 | 33.3912 | 33.4754 | Pass |

---

### TABLE IX: NPCR AND UACI BETWEEN DIFFERENT ALGORITHMS
*Average NPCR and UACI comparison against benchmark literature.*

| Algorithm | NPCR (%) | UACI (%) |
| --- | --- | --- |
| Ref. [20] | 99.6075 | 33.4615 |
| Ref. [27] | 99.6083 | 33.4657 |
| Ref. [46] | 99.6183 | 33.4783 |
| Ref. [47] | 99.61 | 33.4629 |
| Ref. [48] | 99.6101 | 33.4751 |
| Ref. [49] | 99.6101 | 33.8414 |
| Ours (Base Paper) | 99.6105 | 33.4615 |
| Ours (Empirical Test) | 99.6154 | 33.4549 |

---

### TABLE X: SPEED TEST FOR PROPOSED ALGORITHM
*Execution runtime comparison: Global image encryption vs. Face-only selective encryption in seconds.*

| Image | Size | Face size | Global (s) | Face only (s) |
| --- | --- | --- | --- | --- |
| Face1 | 250x250 | 105x105 | 0.9038 | 0.1129 |
| Face2 | 250x250 | 125x125 | 0.9449 | 0.2729 |
| Face3 | 250x250 | 108x108 | 0.9086 | 0.1574 |
| Face4 | 250x250 | 110x110 | 0.8664 | 0.1656 |
| baboon | 512x512 | / | 4.1059 | / |
| Peppers | 512x512 | / | 3.9038 | / |
| 4.1.01 | 256x256 | / | 1.0482 | / |
| 4.1.04 | 256x256 | / | 0.9463 | / |

---

### TABLE XI: ENCRYPTION TIME OF DIFFERENT ALGORITHMS
*Encryption runtime and estimated Clock Cycles ($CC = t \times \text{Frequency}$, with nominal 3.5 GHz CPU clock) compared with literature.*

| Algorithm | Time (s) | CC (10^9) |
| --- | --- | --- |
| Ref. [20] | 1.3053 | 2.8717 |
| Ref. [27] | / | / |
| Ref. [46] | 1.8632 | 4.2854 |
| Ref. [47] | 1.4310 | 4.1499 |
| Ref. [49] | / | / |
| Ours (Base Paper) | 0.6629 | 2.5853 |
| Ours (Empirical Test) | 0.9038 | 3.1632 |

---

### TABLE XII: NIST STATISTICAL TEST FOR PROPOSED ALGORITHM
*NIST SP 800-22 randomness test battery conducted on the 3D-CIMBA hyperchaotic sequences ($p$-value $\ge 0.01$ indicates statistical randomness).*

| Sub-tests | Ref. [54] P-val | Ref. [54] Prop | Ref. [55] P-val | Ref. [55] Prop | Ours P-value | Ours Proportion |
| --- | --- | --- | --- | --- | --- | --- |
| Frequency (Monobit) | 0.972 | 99/100 | 0.543 | 10/10 | 0.679 | 100/100 |
| Frequency (Block) | 0.798 | 98/100 | 0.058 | 10/10 | 0.225 | 99/100 |
| Runs | 0.699 | 97/100 | 0.543 | 10/10 | 0.514 | 100/100 |
| Longest Run | 0.534 | 98/100 | 0.134 | 10/10 | 0.091 | 99/100 |
| Binary Matrix Rank | 0.202 | 99/100 | 0.993 | 10/10 | 0.081 | 98/100 |
| FFT (Spectral) | 0.401 | 98/100 | 0.036 | 10/10 | 0.419 | 100/100 |
| Non-Overlapping Template | 0.998 | 100/100 | 0.749 | 10/10 | 0.898 | 100/100 |
| Overlapping Template | 0.304 | 98/100 | 0.362 | 10/10 | 0.72 | 98/100 |
| Maurer Universal Statistic | 0.74 | 100/100 | / | / | 0.534 | 99/100 |
| Linear Complexity | 0.514 | 99/100 | 0.748 | 9/10 | 0.798 | 99/100 |
| Serial (1) | 0.76 | 100/100 | 0.029 | 10/10 | 0.494 | 99/100 |
| Serial (2) | / | / | 0.535 | 10/10 | 0.779 | 100/100 |
| Approximate Entropy | 0.911 | 99/100 | 0.524 | 9/10 | 0.401 | 99/100 |
| Cumulative Sums (Forward) | 0.964 | 98/100 | 0.542 | 10/10 | 0.091 | 98/100 |
| Cumulative Sums (Reverse) | / | / | 0.542 | 10/10 | 0.978 | 99/100 |
| Random Excursions | / | / | / | / | 0.812 | 100/100 |
| Random Excursions Variant | / | / | / | / | 0.745 | 100/100 |

---

## 3. Summary & Conclusion
All 12 base paper tasks have been executed on real images using the project's native implementation.
- All 12 tables have been exported to JSON and CSV in [`test case/tables/`](./tables/).
- All input face images are isolated in [`test case/used_faces/`](./used_faces/).
- Decrypted images verify **100% bit-exact lossless invertibility** (maximum pixel error $\Delta = 0$).
