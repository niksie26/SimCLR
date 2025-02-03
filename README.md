**Explanation of Files**
  1. The data from the Roboflow link has been downloaded in PyTorch YOLOv7 format.
  2. FirefightingDeviceDetectionDataset_prep.ipynb Extracts crops from the images and organizes them by class. It also resizes the image crops to be 32x32.
  3. train.zip, test.zip, valid.zip - Output of running FireFightingDeviceDetectionDataset_prep.ipynb. Copy this to Directory space of Google colab before running the notebook.
  4. NGurudath_Bobyard_SimCLR.ipynb is same as what is hosted on Google Colab.
  5. optimizer_utils.py implements LARS optimizer
  6. Hyperparameter_Analysis.ipynb - Plots the Numpy arrays collected from running different combinations of batch size, LR and temperature coeficient.  7. 

**Colab Notebook Details**
  1. The colab notebook shared is set up to run with T4 GPU
  2. A batch size of 128 and Number of iterations = 1000 takes around 11 minutes to complete.
  3. Saves the final model file and plots the loss.

**Links**
  1. Google Colab - https://colab.research.google.com/drive/1Jz76fC2vUSqlXMe2-CZjzklrkm1gFpvL?usp=sharing
  2. Github Repo - https://github.com/niksie26/SimCLR.git
