# Feature Extraction Training

This folder contains code for training the Whale Vision feature extraction model, which learns to represent individual whales from images of flukes and flanks.

## Dataset Structure

You need two main folders for training:

- `dataset/`
- `holdout/`

Each must contain two subfolders:

- `flank/`
- `fluke/`

Within each, organise images in folders named after the whale identity. For example:
```
dataset/
	flank/
		WhaleA/
			image1.jpg
			image2.jpg
		WhaleB/
			image1.jpg
	fluke/
		WhaleA/
			image3.jpg
		WhaleB/
			image2.jpg
holdout/
	flank/
		WhaleC/
			image4.jpg
	fluke/
		WhaleC/
			image5.jpg
```

## Training

1. Prepare your datasets as described above.
2. [Insert command or script to start training, e.g., `python train.py`]
3. Training logs and checkpoints will be saved during training.

## Notes

- Make sure all images are correctly labeled and placed in the appropriate folders.
- The holdout set is used for validation/testing and should not overlap with the training set.
