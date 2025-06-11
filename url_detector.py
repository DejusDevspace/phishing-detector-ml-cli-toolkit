#!/usr/bin/env python3
"""
Phishing URL Detector CLI Tool

A command-line interface for detecting phishing URLs using a pre-trained machine learning model.
Supports binary classification: Phishing and Benign.
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


class PhishingURLDetector:
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
        self.feature_names = self._get_expected_feature_names()

    def _get_expected_feature_names(self) -> List[str]:
        """
        Get the expected feature names in the correct order for the model.

        Returns:
        List[str]: List of feature names matching the dataset.
        """
        return [
            'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url',
            'number_of_digits_in_url', 'number_of_special_char_in_url',
            'number_of_hyphens_in_url', 'number_of_underline_in_url',
            'number_of_slash_in_url', 'number_of_questionmark_in_url',
            'number_of_equal_in_url', 'number_of_percent_in_url', 'domain_length',
            'number_of_dots_in_domain', 'number_of_hyphens_in_domain',
            'having_special_characters_in_domain', 'number_of_special_characters_in_domain',
            'having_digits_in_domain', 'number_of_digits_in_domain',
            'having_repeated_digits_in_domain', 'number_of_subdomains',
            'average_subdomain_length', 'number_of_special_characters_in_subdomain',
            'having_digits_in_subdomain', 'number_of_digits_in_subdomain',
            'path_length', 'having_query', 'entropy_of_url', 'entropy_of_domain'
        ]

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

            # Convert to vector maintaining the expected feature order
            feature_vector = np.array([features_dict[name] for name in self.feature_names])
            return feature_vector.reshape(1, -1)

        except Exception as e:
            print(f"✗ Error extracting features from URL '{url}': {e}")
            return None

    def predict_single_url(self, url: str, show_features: bool = False) -> Tuple[str, float]:
        """
        Make a prediction for a single URL.

        Parameters:
        url (str): URL to analyze.
        show_features (bool): Whether to display extracted features.

        Returns:
        Tuple[str, float]: Prediction label and confidence score.
        """
        # Extract features
        features = self.extract_features_vector(url)
        if features is None:
            return "ERROR", 0.0

        # Show features if requested (useful for debugging)
        if show_features:
            features_dict = self.feature_extractor.extract_features(url)
            print(f"\n📊 Extracted features for: {url}")
            for feature, value in features_dict.items():
                print(f"  {feature}: {value}")
            print("-" * 60)

        # Class mapping (0 = Benign, 1 = Phishing)
        class_labels = {
            0: "BENIGN",
            1: "PHISHING"
        }

        try:
            # Make prediction
            prediction = self.model.predict(features)[0]

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
        BLUE = '\033[94m'
        RESET = '\033[0m'

        # Choose color and icon based on prediction
        if prediction == "BENIGN":
            color = GREEN
            icon = "✅"
            status = "SAFE"
        elif prediction == "PHISHING":
            color = RED
            icon = "🎣"
            status = "SUSPICIOUS"
        else:
            color = BLUE
            icon = "❓"
            status = "UNKNOWN"

        print(f"\n{icon} URL: {url}")
        print(f"{color}Prediction: {prediction} ({status}){RESET}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 80)

    def compare_urls(self, url1: str, url2: str) -> None:
        """
        Compare predictions for two URLs and show feature differences.
        Useful for debugging issues like the trailing slash problem.

        Parameters:
        url1 (str): First URL to compare.
        url2 (str): Second URL to compare.
        """
        print(f"\n🔍 Comparing URLs:")
        print(f"URL 1: {url1}")
        print(f"URL 2: {url2}")
        print("-" * 80)

        # Get predictions
        pred1, conf1 = self.predict_single_url(url1)
        pred2, conf2 = self.predict_single_url(url2)

        # Get features
        features1 = self.feature_extractor.extract_features(url1)
        features2 = self.feature_extractor.extract_features(url2)

        print(f"\nPredictions:")
        print(f"URL 1: {pred1} ({conf1:.2%})")
        print(f"URL 2: {pred2} ({conf2:.2%})")

        # Show feature differences
        print(f"\nFeature Differences:")
        differences = []
        for feature in self.feature_names:
            val1 = features1[feature]
            val2 = features2[feature]
            if val1 != val2:
                differences.append((feature, val1, val2))
                print(f"  {feature}: {val1} vs {val2}")

        if not differences:
            print("  No feature differences found!")
        else:
            print(f"\nFound {len(differences)} different features.")

    def interactive_mode(self):
        """
        Run the detector in interactive mode, allowing users to input URLs one at a time.
        """
        print("\n🎣 Phishing URL Detector - Interactive Mode")
        print("Enter URLs to analyze (type 'quit', 'exit', or 'q' to stop):")
        print("Special commands:")
        print("  'compare <url1> <url2>' - Compare two URLs")
        print("  'features <url>' - Show extracted features")
        print("Classes: BENIGN (safe), PHISHING (suspicious)")
        print("-" * 80)

        while True:
            try:
                user_input = input("\nEnter command or URL: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith('compare '):
                    parts = user_input[8:].split()
                    if len(parts) >= 2:
                        url1, url2 = parts[0], parts[1]
                        # Add protocol if missing
                        if not url1.startswith(('http://', 'https://')):
                            url1 = 'https://' + url1
                        if not url2.startswith(('http://', 'https://')):
                            url2 = 'https://' + url2
                        self.compare_urls(url1, url2)
                    else:
                        print("Usage: compare <url1> <url2>")
                    continue

                if user_input.startswith('features '):
                    url = user_input[9:].strip()
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    prediction, confidence = self.predict_single_url(url, show_features=True)
                    self.format_prediction_output(url, prediction, confidence)
                    continue

                # Regular URL prediction
                url = user_input
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
        description="Phishing URL Detector CLI Tool - Binary Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Classes detected:
            - BENIGN: Safe/legitimate websites
            - PHISHING: Suspicious/phishing websites

        Examples:
            Single URL prediction:
                python url_detector.py --model model.pkl --url "https://suspicious-site.com"

            Compare two URLs (useful for debugging):
                python url_detector.py --model model.pkl --compare "https://google.com" "https://google.com/"

            Batch prediction from file:
                python url_detector.py --model model.pkl --batch urls.txt

            Interactive mode:
                python url_detector.py --model model.pkl --interactive
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
    group.add_argument(
        '--compare', '-c',
        nargs=2,
        metavar=('URL1', 'URL2'),
        help='Compare two URLs and show feature differences'
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

    parser.add_argument(
        '--features', '-f',
        action='store_true',
        help='Show extracted features (for debugging)'
    )

    args = parser.parse_args()

    # Initialize the detector
    detector = PhishingURLDetector(args.model)

    if args.interactive:
        # Interactive mode
        detector.interactive_mode()

    elif args.compare:
        # Compare two URLs
        url1, url2 = args.compare
        if not url1.startswith(('http://', 'https://')):
            url1 = 'https://' + url1
        if not url2.startswith(('http://', 'https://')):
            url2 = 'https://' + url2
        detector.compare_urls(url1, url2)

    elif args.url:
        # Single URL prediction
        url = args.url
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        prediction, confidence = detector.predict_single_url(url, show_features=args.features)
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
                # Compact output for batch mode
                icon = "✅" if prediction == "BENIGN" else "🎣"
                print(f"{icon} {url} -> {prediction} ({confidence:.2%})")

        # Save to CSV if requested
        if args.output:
            df = pd.DataFrame(results, columns=['URL', 'Prediction', 'Confidence'])
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to {args.output}")

        # Summary
        total = len(results)
        benign_count = sum(1 for _, pred, _ in results if pred == "BENIGN")
        phishing_count = sum(1 for _, pred, _ in results if pred == "PHISHING")

        print(f"\n📊 Summary:")
        print(f"Total URLs: {total}")
        print(f"BENIGN: {benign_count} ({benign_count/total:.1%})")
        print(f"PHISHING: {phishing_count} ({phishing_count/total:.1%})")


if __name__ == "__main__":
    main()
