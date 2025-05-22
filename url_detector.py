#!/usr/bin/env python3
"""
Malicious URL Detector CLI Tool

A command-line interface for detecting malicious URLs using a pre-trained machine learning model.
Supports multi-class classification: Defacement, Benign, Malware, Phishing, and Spam.
Leverages the FeatureExtractor class to extract URL features and make predictions.

Usage:
    python url_detector.py --model random_forest_classifier_model.pkl --url "https://example.com"
    python url_detector.py --model random_forest_classifier_model.pkl --batch urls.txt
    python url_detector.py --model random_forest_classifier_model.pkl --interactive
"""

import argparse
import pickle
import sys
import os
from typing import List, Tuple
import pandas as pd
import numpy as np

# Import the FeatureExtractor class (assuming it's in the same directory)
from feature_extractor import FeatureExtractor


class MaliciousURLDetector:
    """
    Main class for the phishing detection CLI tool.
    """

    def __init__(self, model_path: str):
        """
        Initialize the detector with a trained model.

        Parameters:
        model_path (str): Path to the pickled model file.
        """
        self.feature_extractor = FeatureExtractor()
        self.model = self.load_model(model_path)
        self.feature_names = None

    def load_model(self, model_path: str):
        """
        Load the trained model from a pickle file.

        Parameters:
        model_path (str): Path to the model file.

        Returns:
        Trained model object.
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            print(f"✗ Error: Model file '{model_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)

    def extract_features_vector(self, url: str) -> np.ndarray:
        """
        Extract features from a URL and return as a vector for prediction.

        Parameters:
        url (str): URL to extract features from.

        Returns:
        np.ndarray: Feature vector for the model.
        """
        try:
            features_dict = self.feature_extractor.extract_features(url)

            # Store feature names for the first extraction (for consistency)
            if self.feature_names is None:
                self.feature_names = list(features_dict.keys())

            # Convert to vector maintaining feature order
            feature_vector = np.array([features_dict[name] for name in self.feature_names])
            return feature_vector.reshape(1, -1)

        except Exception as e:
            print(f"✗ Error extracting features from URL '{url}': {e}")
            return None

    def predict_single_url(self, url: str) -> Tuple[str, float]:
        """
        Make a prediction for a single URL.

        Parameters:
        url (str): URL to analyze.

        Returns:
        Tuple[str, float]: Prediction label and confidence score.
        """
        # Extract features
        features = self.extract_features_vector(url)
        if features is None:
            return "ERROR", 0.0

        # Class mapping
        class_labels = {
            0: "DEFACEMENT",
            1: "BENIGN",
            2: "MALWARE",
            3: "PHISHING",
            4: "SPAM"
        }

        try:
            # Make prediction
            prediction = self.model.predict(features)[0]
            print("Output", self.model.predict(features))

            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = max(probabilities)
            else:
                # For models without probability estimates, use 1.0 for predicted class
                confidence = 1.0

            # Convert prediction to readable label
            label = class_labels.get(prediction, f"UNKNOWN_CLASS_{prediction}")

            return label, confidence

        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return "ERROR", 0.0

    def predict_batch_urls(self, urls: List[str]) -> List[Tuple[str, str, float]]:
        """
        Make predictions for a batch of URLs.

        Parameters:
        urls (List[str]): List of URLs to analyze.

        Returns:
        List[Tuple[str, str, float]]: List of (URL, prediction, confidence) tuples.
        """
        results = []
        for url in urls:
            prediction, confidence = self.predict_single_url(url)
            results.append((url, prediction, confidence))
        return results

    def format_prediction_output(self, url: str, prediction: str, confidence: float) -> None:
        """
        Format and print prediction results.

        Parameters:
        url (str): The analyzed URL.
        prediction (str): Prediction result.
        confidence (float): Confidence score.
        """
        # Color codes for terminal output
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        RESET = '\033[0m'

        # Choose color and icon based on prediction
        if prediction == "BENIGN":
            color = GREEN
            icon = "✓"
        elif prediction == "PHISHING":
            color = RED
            icon = "🎣"
        elif prediction == "MALWARE":
            color = RED
            icon = "🦠"
        elif prediction == "SPAM":
            color = YELLOW
            icon = "📧"
        elif prediction == "DEFACEMENT":
            color = MAGENTA
            icon = "🔧"
        else:
            color = BLUE
            icon = "❓"

        print(f"\n{icon} URL: {url}")
        print(f"{color}Prediction: {prediction}{RESET}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 60)

    def interactive_mode(self):
        """
        Run the detector in interactive mode, allowing users to input URLs one at a time.
        """
        print("\n🔍 Malicious URL Detector - Interactive Mode")
        print("Enter URLs to analyze (type 'quit' or 'exit' to stop):")
        print("Classes: BENIGN, PHISHING, MALWARE, SPAM, DEFACEMENT")
        print("-" * 60)

        while True:
            try:
                url = input("\nEnter URL: ").strip()

                if url.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not url:
                    continue

                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                prediction, confidence = self.predict_single_url(url)
                self.format_prediction_output(url, prediction, confidence)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"✗ An error occurred: {e}")


def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file (one URL per line).

    Parameters:
    file_path (str): Path to the file containing URLs.

    Returns:
    List[str]: List of URLs.
    """
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return []


def main():
    """
    Main function to handle command-line arguments and run the detector.
    """
    parser = argparse.ArgumentParser(
        description="Malicious URL Detector CLI Tool - Multi-class Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Classes detected:
  - BENIGN: Safe/legitimate websites
  - PHISHING: Phishing websites
  - MALWARE: Malware distribution sites
  - SPAM: Spam-related websites
  - DEFACEMENT: Defaced websites

Examples:
  Single URL prediction:
    python malicious_url_detector.py --model model.pkl --url "https://suspicious-site.com"

  Batch prediction from file:
    python malicious_url_detector.py --model model.pkl --batch urls.txt

  Interactive mode:
    python malicious_url_detector.py --model model.pkl --interactive
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to the trained model pickle file'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--url', '-u',
        help='Single URL to analyze'
    )
    group.add_argument(
        '--batch', '-b',
        help='Path to text file containing URLs (one per line)'
    )
    group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--output', '-o',
        help='Save results to CSV file (only for batch mode)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Initialize the detector
    detector = MaliciousURLDetector(args.model)

    if args.interactive:
        # Interactive mode
        detector.interactive_mode()

    elif args.url:
        # Single URL prediction
        url = args.url
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        prediction, confidence = detector.predict_single_url(url)
        detector.format_prediction_output(url, prediction, confidence)

    elif args.batch:
        # Batch prediction
        urls = load_urls_from_file(args.batch)
        if not urls:
            sys.exit(1)

        print(f"🔍 Analyzing {len(urls)} URLs...")
        results = detector.predict_batch_urls(urls)

        # Display results
        for url, prediction, confidence in results:
            if args.verbose:
                detector.format_prediction_output(url, prediction, confidence)
            else:
                # Compact output for batch mode with appropriate icons
                if prediction == "BENIGN":
                    icon = "✓"
                elif prediction == "PHISHING":
                    icon = "🎣"
                elif prediction == "MALWARE":
                    icon = "🦠"
                elif prediction == "SPAM":
                    icon = "📧"
                elif prediction == "DEFACEMENT":
                    icon = "🔧"
                else:
                    icon = "❓"
                print(f"{icon} {url} -> {prediction} ({confidence:.2%})")

        # Save to CSV if requested
        if args.output:
            df = pd.DataFrame(results, columns=['URL', 'Prediction', 'Confidence'])
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to {args.output}")

        # Summary with all classes
        total = len(results)
        class_counts = {}
        for _, prediction, _ in results:
            class_counts[prediction] = class_counts.get(prediction, 0) + 1

        print(f"\n📊 Summary:")
        print(f"Total URLs: {total}")
        for class_name, count in sorted(class_counts.items()):
            percentage = count / total * 100
            print(f"{class_name}: {count} ({percentage:.1%})")


if __name__ == "__main__":
    main()
