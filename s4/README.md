# Efficiently Modeling Long Sequences with Structured State Spaces

This repository contains the implementation of the research paper **`Efficiently Modeling Long Sequences with Structured State Spaces`** along with my explaination about SSM and S4

## Project Structure

```plaintext
s4/
├── models/
│   ├── batch_staacked_model.py
│   └── s4_model.py
├── utils/
│   ├── cnn_mode.py
│   ├── dataset.py
│   ├── helper.py
│   ├── hippo.py
│   └── rnn_model.py
├── config.yaml
├── README.md
├── s4_implementation.ipynb
├── sample.py
└── train.py
```

## Getting Started

Follow these steps to get the models ready for training:

### 1. Configuring the YAML File

You can play around with the hyperparameters of the models and change the train config according to your need.

Currently supported:

 - Only mnist dataset
 - Only s4 layer

### 2. Creating a virtual environment for the repo

Make sure you have Python & Poetry installed. Create and activate the virtual enviroment:

```bash
poetry init
poetry shell
```

### 3. Installing Dependencies

Install the project dependencies using:

```bash
poetry install
```

### 4. Model Training

With all configs done, you can start the model training with following cmd:

```bash
python -m s4.train
```


