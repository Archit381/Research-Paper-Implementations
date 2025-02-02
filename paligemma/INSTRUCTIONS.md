# PaliGemma: A versatile 3B VLM for transfer

## Getting Started

Follow these steps to get the script ready for inference:

### 1. Initializing required model files

Head over to [Paligemma-3b model](https://huggingface.co/google/paligemma-3b-pt-224/tree/main) and download the entire repository

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

### 4. Updating the bat file

Go the .bat file and change the path of the model repo and the testing image

### 5. Running the script

Run the inference script

 ```bash
paligemma\launch_inference.bat
```
