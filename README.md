# Digital Image Processing

Supplementary Python code, Jupyter notebooks, and datasets for the book **Digital Image Processing** by Dr. Mahmood Azimi-Sadjadi.

The repository is organized by book chapter and provides executable demonstrations of classical image-processing methods, image reconstruction and compression, feature extraction, and machine-learning applications.

## At a glance

- 52 Jupyter notebooks across 17 chapter folders
- Examples covering foundational, statistical, and learning-based image processing
- Sample images and numerical datasets in [`Data`](./Data)
- Notebooks designed primarily for Python 3 and Google Colab

## Chapter guide

| Chapter | Topics and examples |
| --- | --- |
| [Chapter 3 — Theoretical Background](./Chapter%203%20Theoretical%20Background) | Two-dimensional power spectra |
| [Chapter 5 — Image Sampling and Quantization](./Chapter%205%20Image%20Sampling%20and%20Quantization) | Sampling, aliasing, uniform quantization, and Lloyd–Max quantization |
| [Chapter 6 — Image Transforms](./Chapter%206%20Image%20Transforms) | DFT, DCT, Hadamard transforms, basis functions, and PCA |
| [Chapter 7 — Wavelet Transform](./Chapter%207%20Wavelet%20Transform) | DWT, compression, denoising, image fusion, STFT, and the Mexican hat wavelet |
| [Chapter 8 — Image Enhancement](./Chapter%208%20Image%20Enhancement) | Edge detection and template matching |
| [Chapter 9 — Image Modeling](./Chapter%209%20Image%20Modeling) | Two-dimensional autoregressive image models and covariance |
| [Chapter 10 — Image Restoration](./Chapter%2010%20Image%20Restoration) | Inverse, Kalman, and Wiener filtering |
| [Chapter 11 — Image Compression and Encoding](./Chapter%2011%20Image%20Compression%20and%20Encoding) | Predictive encoding |
| [Chapter 12 — Image Segmentation](./Chapter%2012%20Image%20Segmentation) | Otsu thresholding, MRF/K-means, and gradient vector flow |
| [Chapter 13 — Feature Extraction](./Chapter%2013%20Feature%20Extraction) | Fourier, Hu, and Zernike descriptors; correlograms; PCA |
| [Chapter 14 — Morphological Operations](./Chapter%2014%20Morphological%20Operations) | Mathematical morphology and hierarchical texture decomposition |
| [Chapter 15 — Image Reconstruction](./Chapter%2015%20Image%20Reconstruction) | Radon-transform reconstruction using phantom and medical images |
| [Chapter 16 — Traditional Image Classification](./Chapter%2016%20Traditional%20Image%20Classification) | Naive Bayes classification |
| [Chapter 17 — Modern Image Classification: Elements of Machine Learning](./Chapter%2017%20Modern%20Image%20Classification%20-%20Elements%20of%20Machine%20Learning) | Autoregressive forecasting |
| [Chapter 18 — Modern Image Classification-Layered Machines](./Chapter%2018%20Modern%20Image%20Classification-Layered%20Machines) | Multilayer perceptrons and convolutional neural networks |
| [Chapter 19 — Dimensionality Reduction Networks and Autoencoders](./Chapter%2019%20Dimensionality%20Reduction%20Networks%20and%20Autoencoders) | Contractive autoencoders and image denoising |
| [Chapter 20 — AI Applications in Digital Image Processing](./Chapter%2020%20AI%20Applications%20in%20Digital%20Image%20Processing) | Associative memory, self-organizing maps, and stereo vision |

## Getting started

Clone the repository:

```bash
git clone https://github.com/mgcy/Digital-Image-Processing.git
cd Digital-Image-Processing
```

Create a virtual environment and install a core scientific Python stack:

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then install the core packages and launch Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter numpy scipy matplotlib pandas pillow opencv-python scikit-image scikit-learn PyWavelets
jupyter lab
```

Some notebooks require additional packages such as TensorFlow, PyTorch, `mahotas`, `pyefd`, or `cmasher`. Check the import or installation cells near the beginning of the notebook you want to run.

## Running in Google Colab

Many notebooks were authored for Google Colab and mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

They may also use a path similar to:

```python
work_path = "/content/drive/MyDrive/DigitalImageProcessing/"
```

Place the repository at that location or update `work_path` to point to your copy. When running locally, replace Colab-specific paths and skip the Google Drive mount cell.

## Data

The [`Data`](./Data) directory contains the images, MATLAB arrays, NumPy arrays, and example datasets used by the notebooks. It currently contains approximately 253 files totaling 474 MiB, so the initial clone may take some time.

Keep the directory structure intact because many notebooks load files by relative name or through `work_path`.

## Repository structure

```text
Digital-Image-Processing/
├── Chapter 3 Theoretical Background/
├── Chapter 5 Image Sampling and Quantization/
├── ...
├── Chapter 20 AI Applications in Digital Image Processing/
├── Data/
└── README.md
```

## Contributing

Corrections and improvements are welcome. When contributing:

1. Keep examples in the chapter that matches their subject.
2. Prefer repository-relative paths over machine-specific absolute paths.
3. Document any new dependencies.
4. Avoid committing generated outputs, model checkpoints, or notebook secrets.
5. Clear unusually large notebook outputs before committing.

## Credits

- Book and course materials: **Dr. Mahmood Azimi-Sadjadi**
- Code contributors identified in the notebooks: **Yifan Yang** and **Dr. Mahmood Azimi-Sadjadi**
- Contact: `mahmood.azimi-sadjadi@colostate.edu`

## License

This repository does not currently include an open-source license. Contact the authors before redistributing or reusing the materials beyond uses permitted by applicable law.
