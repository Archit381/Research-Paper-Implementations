@echo off
set MODEL_PATH=C:\Users\Archit\Desktop\VLM\paligemma-3b-pt-224
set PROMPT=describe this image
set IMAGE_FILE_PATH=assets\paligemma_architecture.png
set MAX_TOKENS_TO_GENERATE=100
set TEMPERATURE=0.8
set TOP_P=0.9
set DO_SAMPLE=False
set ONLY_CPU=False

python paligemma/inference.py ^
    --model_path "%MODEL_PATH%" ^
    --prompt "%PROMPT%" ^
    --image_file_path "%IMAGE_FILE_PATH%" ^
    --max_tokens_to_generate %MAX_TOKENS_TO_GENERATE% ^
    --temperature %TEMPERATURE% ^
    --top_p %TOP_P% ^
    --do_sample %DO_SAMPLE% ^
    --only_cpu %ONLY_CPU%
