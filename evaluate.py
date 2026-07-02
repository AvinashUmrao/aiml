import argparse
import os
import sys
import glob
import numpy as np
from utils import load_model, predict_image, get_transform, DEFAULT_CHECKPOINT_PATH, DEFAULT_DATASET_ROOT


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATASET_ROOT,
                        help='Path to dataset root')
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    transform = get_transform(num_crops=10)
    
    test_dir = os.path.join(args.data_root, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)
    
    real_dir = os.path.join(test_dir, 'real')
    screen_dir = os.path.join(test_dir, 'screen')
    
    real_images = glob.glob(os.path.join(real_dir, '*'))
    screen_images = glob.glob(os.path.join(screen_dir, '*'))
    len_real = len(real_images)
    
    all_images = real_images + screen_images
    all_labels = [0] * len_real + [1] * len(screen_images)
    
    if len(all_images) == 0:
        print("Error: No images found in test directory", file=sys.stderr)
        sys.exit(1)
    
    true_labels = []
    predictions = []
    
    for img_path, true_label in zip(all_images, all_labels):
        screen_prob, _ = predict_image(img_path, model, transform)
        
        if screen_prob is None:
            continue
        
        prediction = 1 if screen_prob > 0.5 else 0
        predictions.append(prediction)
        true_labels.append(true_label)
    
    if len(predictions) == 0:
        print("Error: No valid images processed", file=sys.stderr)
        sys.exit(1)
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    result = true_labels == predictions
    
    real_result = result[:len_real]
    screen_result = result[len_real:]
    
    print(f'real accuracy: {real_result.sum() * 100.0 / len(real_result):.2f}%')
    print(f'screen accuracy: {screen_result.sum() * 100.0 / len(screen_result):.2f}%')
    print(f'average accuracy: {(real_result.sum() * 100.0 / len(real_result) + screen_result.sum() * 100.0 / len(screen_result)) / 2:.2f}%')


if __name__ == '__main__':
    main()
