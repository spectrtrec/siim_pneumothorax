# Pneumothorax segmentation
### Generate submissions file
```
1. Generate  first submissions file:
python train.py --fine_size=512 --batch_size=4 --epoch=70 --swa_epoch=67
2. Generate second submission file:
python train.py --network=UEfficientNetV1 --fine_size=512 --batch_size=3 --epoch=70 --swa_epoch=67
```

### Resize images and masks
```python
python create_mask.py
```

### Generate folds
```python
python generate_folds.py
```