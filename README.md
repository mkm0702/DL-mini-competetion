# PASCAL VOC Efficient Segmentation

Lightweight semantic segmentation on PASCAL VOC 2012 using MobileNetV3-Small + LRASPP.

---

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download and split data
```bash
python3 stage1_data_setup.py
```
Downloads PASCAL VOC 2012 (~2 GB), splits into 80% train / 20% val, and verifies no test images leaked into the splits.

### Step 3 — Train the model
```bash
python3 stage4_train.py
```
Trains MobileNetV3-Small + LRASPP. Best checkpoint saved to `checkpoints/best_model.pth`.

### Step 4 — Generate binary masks
```bash
python3 inference.py --in_dir=data/VOCdevkit/VOC2012/JPEGImages --out_dir=groupN_output
```
Runs inference on all test images and saves binary masks (foreground = white, background = black) to `groupN_output/`.

### Step 5 — Get leaderboard numbers
```bash
python3 stage6_evaluate.py --pred_dir=groupN_output
```
Computes binary Dice score and GFLOPs per image. Prints the values to submit to the leaderboard form.

---

## Results

| Metric       | Value  |
|--------------|--------|
| Dice Score   | 0.6555 |
| GFLOPs       | 0.1267 |
| Dice / GFLOPs | 5.1736 |
