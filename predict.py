import argparse
import sys
from utils import load_model, predict_image, get_transform, DEFAULT_CHECKPOINT_PATH


def main():
    parser = argparse.ArgumentParser(description='Predict screen recapture probability')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--num-crops', type=int, default=10, choices=[1, 5, 10],
                        help='Number of crops for inference (1=fastest, 10=most accurate)')
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    transform = get_transform(num_crops=args.num_crops)
    
    screen_prob, _ = predict_image(args.image_path, model, transform)
    
    if screen_prob is None:
        sys.exit(1)
    
    print(screen_prob)


if __name__ == '__main__':
    main()
