# VCOM

## Group members:

| Name             | Number    | E-Mail             |
| ---------------- | --------- | ------------------ |
| Carolina Gonçalves | 202108781 | up202108781@up.pt |
| João Costa | 202108714 | up202108714@up.pt |
| João Sobral | 202108736 | up202108736@up.pt
| Marco Costa | 202108821 | up202108821@up.pt |

## How to run
```python
python main.py --input=<path_to_input_file> --output=<path_to_output_file> --mode=<classic|deep> --verbose --out=<path_to_verbose_output> --model=<path_to_model>
```
Default input file is `./data/input.json`
Default output file is `./output.json`
Default mode is `deep`
Default verbose is deactivated by default. Take care that verbose will create many files.
Default verbose output directory is `./output`
Default model is `./models/MobileNet_V2_regression_base_multiclass_finetune_best_model.pth`

## Task 2
File `resize.py` contains the code used to resize the images prior to training.
File `task2_baseline.ipynb` contains the code used to train the models and select the best one.
File `main.py` contains the code used to run the model and complete the task. It uses the files on `src/`.