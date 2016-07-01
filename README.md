# torch-models
Various deep learning models in torch

1. DeepLab model (DeepLab-MSc-COCO-LargeFOV) without CRF for post-processing. Consists of only the model structure for training.
Based on caffe implementation at http://ccvl.stat.ucla.edu/ccvl/DeepLab-MSc-COCO-LargeFOV/train.prototxt
Features exploited:
  (1) Multiscale features 
  (2) extra annotations from MS-COCO dataset (pretrained on MS-COCO, and then fine-tuned on PASCAL)
  (3) large Field-Of-View.
