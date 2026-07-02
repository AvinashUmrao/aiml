import argparse
import os
import sys
import time
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_model, predict_image, get_transform, DEFAULT_CHECKPOINT_PATH, DEFAULT_DATASET_ROOT


def main():
    parser = argparse.ArgumentParser(description='Benchmark model on test set')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATASET_ROOT,
                        help='Path to dataset root')
    parser.add_argument('--num-crops', type=int, default=10, choices=[1, 5, 10],
                        help='Number of crops for inference (1=fastest, 10=most accurate)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1 for memory efficiency)')
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    transform = get_transform(num_crops=args.num_crops)
    
    test_dir = os.path.join(args.data_root, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)
    
    real_dir = os.path.join(test_dir, 'real')
    screen_dir = os.path.join(test_dir, 'screen')
    
    if not os.path.exists(real_dir) or not os.path.exists(screen_dir):
        print("Error: Test directory must contain 'real' and 'screen' subdirectories", file=sys.stderr)
        sys.exit(1)
    
    real_images = glob.glob(os.path.join(real_dir, '*'))
    screen_images = glob.glob(os.path.join(screen_dir, '*'))
    
    all_images = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    latencies = []
    
    for img_path in real_images:
        all_images.append(img_path)
        all_labels.append(0)
    
    for img_path in screen_images:
        all_images.append(img_path)
        all_labels.append(1)
    
    if len(all_images) == 0:
        print("Error: No images found in test directory", file=sys.stderr)
        sys.exit(1)
    
    print(f'Evaluating {len(all_images)} images...')
    
    for img_path, true_label in zip(all_images, all_labels):
        start_time = time.time()
        screen_prob, confidence = predict_image(img_path, model, transform)
        end_time = time.time()
        
        if screen_prob is None:
            continue
            
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
        all_confidences.append(confidence)
        
        prediction = 1 if screen_prob >= 0.5 else 0
        all_predictions.append(prediction)
    
    if len(all_predictions) == 0:
        print("Error: No valid images processed", file=sys.stderr)
        sys.exit(1)
    
    accuracy = accuracy_score(all_labels[:len(all_predictions)], all_predictions)
    precision = precision_score(all_labels[:len(all_predictions)], all_predictions, zero_division=0)
    recall = recall_score(all_labels[:len(all_predictions)], all_predictions, zero_division=0)
    f1 = f1_score(all_labels[:len(all_predictions)], all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels[:len(all_predictions)], all_predictions)
    
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    avg_confidence = np.mean(all_confidences)
    
    print('\n=== Benchmark Results ===')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'\nConfusion Matrix:')
    print(cm)
    print(f'\nLatency (ms):')
    print(f'Average: {avg_latency:.2f}')
    print(f'Minimum: {min_latency:.2f}')
    print(f'Maximum: {max_latency:.2f}')
    print(f'\nAverage Prediction Confidence: {avg_confidence:.4f}')


if __name__ == '__main__':
    main()
