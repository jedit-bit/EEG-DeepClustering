# EEG Classification with Limited Data: A Deep Clustering Approach
## Part of the emotion recognition project-2025

**PyTorch implementation**

This repository contains a PyTorch implementation that simulates EEG-like signals using CIFAR-10 images and performs classification with:

- Feature generator (φ) using 1D convolutions + GRU + normalization  
- Classifier head (C)  
- Dictionary-based clustering head (P)  
- Domain discriminator (D) for domain/adversarial simulation  

![EEG-inspired CIFAR-10](images/ReClustering.png)

## Paper Reference

This project is based on the following paper:

Tabejamaat, M., Mohammadzade, H., Negin, F., & Bremond, F. (2025). *EEG classification with limited data: A deep clustering approach*. Pattern Recognition, 157, 110934. [Link](https://www.sciencedirect.com/science/article/pii/S003132032400685X)

BibTeX citation:

```bibtex
@article{tabejamaat2025eeg,
  title={EEG classification with limited data: A deep clustering approach},
  author={Tabejamaat, Mohsen and Mohammadzade, Hoda and Negin, Farhood and Bremond, Francois},
  journal={Pattern Recognition},
  volume={157},
  pages={110934},
  year={2025},
  publisher={Elsevier}
}
```

## Folder Structure

```
EEG-DeepClustering/
├─ eeg_clustering/
│   ├─ __init__.py
│   ├─ dataset.py
│   ├─ model.py
│   ├─ train.py
│   └─ utils.py
├─ images/            
├─ notebooks/       
├─ tests/             
├─ data/              
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/jedit-bit/EEG-DeepClustering.git
cd EEG-DeepClustering
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**

- torch  
- torchvision  

## Usage

Run the training and evaluation script:

```bash
python eeg_clustering/train.py
```

- Trains the feature generator, classifier, clustering head, and domain discriminator.  
- Uses CIFAR-10 images as EEG-like sequences (resized to 16x16).  
- Prints training loss and accuracy for each epoch.  
- Evaluates on the test set and prints final test accuracy.  

## Notes

- This is a **non-official implementation** inspired by EEG signals applied to CIFAR-10.  
- Dictionary features are updated every epoch from one representative per class.  
- Domain discriminator simulates multiple “subjects” or domains for adversarial training.  

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
