# VCOM

## Group members:

| Name             | Number    | E-Mail             |
| ---------------- | --------- | ------------------ |
| Carolina Gonçalves | 202108781 | up202108781@up.pt |
| João Costa | 202108714 | up202108714@up.pt |
| João Sobral | 202108736 | up202108736@up.pt
| Marco Costa | 202108821 | up202108821@up.pt |

## Task 1 
‣ Input:
    ‣ Image with chess board
‣ Output:
    ‣ Total number of black/white pieces on the board
    ‣ Position of the pieces on the image (bounding boxes)
    ‣ Position of the pieces on the board (8x8 matrix with 0/1 values representing absence/presence of piece - any board orientation is acceptable)

Dataset:
    ‣ 50 images randomly chosen from a public dataset
    ‣ The results will be tested in 10 undisclosed images
Deliverables:
    ‣ Short report (2 pages max) presenting the methodology and some results
    ‣ Python script (only one file)
Deadline: April, 14 (23:59 AoE)

## How to run
```python
python main.py --input=<path_to_input_file>
```
Default input file is `./data/input.json`
Images referencered in the input file should be placed in 'images' folder.