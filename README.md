# Malaria Detection with Convolutional Neural Networks


Malaria is still diagnosed mainly by examining blood smears under a microscope. That needs trained staff and time, both scarce in many clinics across East Africa. This project trains neural networks to tell **parasitized** (infected) red blood cells from **uninfected** ones, using about 27,500 labelled cell images. It then asks two questions that matter more than accuracy: *how many infected cells does the model miss?* and *is it looking at the parasite?*

> Educational project. These models are not medical devices and must not be used for diagnosis.

## What it does

| Step | Detail |
|---|---|
| Data | NIH malaria cell images (27,558, half infected), split 70 / 15 / 15 into train, validation and test |
| Model 1 | Small CNN trained from scratch, with data augmentation (flips, rotations, zoom) |
| Model 2 | MobileNetV2 pretrained on ImageNet: train a new head, then fine-tune the top blocks |
| Evaluation | Sensitivity (infected cells caught), specificity, precision, F1, ROC AUC, confusion matrix |
| Threshold analysis | How lowering the decision cut-off trades false alarms for fewer missed infections |
| Explainability | Grad-CAM heatmaps showing which part of each cell drove the prediction |
| Demo | Streamlit app: upload a cell image, get a prediction and heatmap |


## Project structure

```
malaria-detection-cnn/
├── notebooks/malaria_cnn.ipynb   step-by-step walkthrough (run this first)
├── src/
│   ├── data.py       loading, 70/15/15 split, tf.data pipelines
│   ├── models.py     baseline CNN, MobileNetV2 transfer model, fine-tuning
│   ├── train.py      command-line training (also evaluates and saves Grad-CAM)
│   ├── evaluate.py   test-set metrics, confusion matrix, ROC curve
│   └── gradcam.py    Grad-CAM heatmaps and overlays
├── app/streamlit_app.py          upload-and-predict demo
├── outputs/                      models, metrics and figures (created by training)
├── requirements.txt
└── LICENSE
```

## Limitations

- **Same patient in train and test.** The images are split by cell, not by patient, and each patient contributed many cells. Cells from one person can land in both sets, which makes test scores look better than they would on new patients. The public release has no patient IDs, so this can't be fixed here.
- **Label noise.** Reviews of this dataset have found some mislabelled cells.
- **One source.** All images come from one hospital's Giemsa-stained thin smears. Images from other clinics, stains, microscopes or phone cameras may look different, and accuracy could drop.
- **Cells, not patients.** A real diagnosis needs the whole smear: finding cells, counting parasites and identifying the species. This project covers only classifying single, already-cropped cells.

## Dataset and citation

The images come from the U.S. National Library of Medicine's malaria dataset: thin blood smears collected at Chittagong Medical College Hospital, Bangladesh. They are downloaded at run time through [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/malaria) and are not stored in this repo.

> Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude RJ, Jaeger S, Thoma GR. *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ 6:e4568 (2018).

## Licence

The code in this repo is © 2026 Barbara Nyanchera, released under the [MIT License](LICENSE). The dataset is not part of this repo and is not covered by that licence.
