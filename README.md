# Targeted Perturbation Generator
Generate adversarial video samples for the Ultralytics YOLO26 model to modify a tracked object’s label.

# Project Details
## Project Objective
This project’s primary objective is to record the attack success rate on the benchmarked COCO testing
dataset for YOLO26 to demonstrate the vulnerability of modern, state-of-the-art object detection models.
Currently, there are no adversarial attacks published on the newest YOLO26 model released in Jan-
uary 2026. This project aims to give an initial evaluation on the stability and security of the newest
Ultralytics YOLO model. Our findings can help make developers publicly aware of the limitations and
safety of using YOLO26 to detect objects from external sources.
Hijacking the confidence and label of computer vision models is primarily in the domain of adversarial
machine learning. Using techniques such as Fast Gradient Sign Method (FGSM), we can incrementally
train our adversarial examples.
## 3.2 Datasets
Please describe your dataset in this section. We plan to use the COCO dataset, which is a very popular
benchmarking dataset and is easy to access (COCO Download Link). Additionally, Ultralytics provides
an API to the latest YOLO26 models pre-trained on this dataset, which is critical for having a known
stable model for benchmarking attack performance.
COCO 2017 is used by Ultralytics in training their models, we will use the test set for our testing
as well. COCO 2017 has over hundreds of thousands of annotated images for object detection with over
80 object categories including but not limited to vehicles and animals. The annotations will provide the
bounding boxes for our object detection task as well as the ground truth labels. We then can attack
these labels to produce incorrect output with perturbations.
The following information is sourced from Ultralytics:
- Train2017: This subset contains 118K images for training object detection, segmentation, and
captioning models.
- Val2017: This subset has 5K images used for validation purposes during model training.

- Test2017: This subset consists of 20K images used for testing and benchmarking the trained models.
Ground truth annotations for this subset are not publicly available, and the results are submitted
to the COCO evaluation server for performance evaluation.
## 3.3 Machine Learning Algorithm
We plan on using common adversarial machine learning techniques such as FGSM, Project Gradient
Descent (PGD), and other Jacobian-based methods in combination with the pre-existing, pre-trained
YOLO26 model.
Using the YOLO26 model, we can continuously test our perturbed videos and analyze the success of
our attack by using the confidence of the true label as the loss function. Using FGSM, we will optimize δ,
the filtered change on the video segment to induce an incorrect label, while keeping the attack and true
input identical to the human eye. By utilizing pretrained high-performing models and a robust dataset,
this will allow the group’s focus to be on the adversarial attack itself.
The combination of the malicious algorithms and the victim algorithm will comprehensively cover the
scope of our project for evaluating the vulnerability of state-of-the-art detection models.
## 3.4 Expected Outcomes
Our project deliverables will be a complete and comprehensive report on implementing adversarial ma-
chine learning attacks on YOLO26 and analysis on the most resilient and most vulnerable classes to
attacks.
Our expected outcome to this project is to create public awareness for commonly-used high-performing
computer vision model’s susceptibility to perturbed attacks. Our documentation, algorithm design, and
experimental results will ensure for reliable and insightful results on modern computer vision attacks.
Additionally, both group members (Mason and Ethan) have little prior experience with adversarial at-
tacks, so another outcome is to learn how to setup, train, and run adversarial attacks on the latest suite
of YOLO26 models.