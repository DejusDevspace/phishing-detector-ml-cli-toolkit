import re
import math
import tldextract
from urllib.parse import urlparse
from collections import Counter
import numpy as np

class FeatureExtractor:
    """
    A class for extracting URL-based features for phishing detection.
    Extracts features matching the new dataset requirements.
    """
    def __init__(self):
        self.url_features = None

    def calculate_entropy(self, string: str) -> float:
        """
        Calculate the Shannon entropy of a string.

        Parameters:
        string (str): Input string.

        Returns:
        float: Entropy value.
        """
        if not string:
            return 0.0

        # Count frequency of each character
        char_counts = Counter(string)
        string_length = len(string)

        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / string_length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def has_repeated_digits(self, string: str) -> int:
        """
        Check if string has repeated consecutive digits.

        Parameters:
        string (str): Input string.

        Returns:
        int: 1 if has repeated digits, 0 otherwise.
        """
        if not string:
            return 0

        # Look for consecutive repeated digits
        for i in range(len(string) - 1):
            if string[i].isdigit() and string[i] == string[i + 1]:
                return 1
        return 0

    def count_digits(self, string: str) -> int:
        """
        Count the number of digits in the string.

        Parameters:
        string (str): Input string.

        Returns:
        int: Number of digits.
        """
        if not string:
            return 0
        return sum(1 for c in string if c.isdigit())

    def count_special_characters(self, string: str) -> int:
        """
        Count the number of special characters in the string.
        Special characters are non-alphanumeric characters.

        Parameters:
        string (str): Input string.

        Returns:
        int: Number of special characters.
        """
        if not string:
            return 0
        return sum(1 for c in string if not c.isalnum())

    def has_special_characters(self, string: str) -> int:
        """
        Check if string has any special characters.

        Parameters:
        string (str): Input string.

        Returns:
        int: 1 if has special characters, 0 otherwise.
        """
        return 1 if self.count_special_characters(string) > 0 else 0

    def has_digits(self, string: str) -> int:
        """
        Check if string has any digits.

        Parameters:
        string (str): Input string.

        Returns:
        int: 1 if has digits, 0 otherwise.
        """
        return 1 if self.count_digits(string) > 0 else 0

    def extract_features(self, url: str) -> dict:
        """
        Extract features from a URL matching the new dataset requirements.

        Parameters:
        url (str): Input URL string.

        Returns:
        dict: Dictionary of extracted feature names and their values.
        """
        if not url:
            return self._get_default_features()

        try:
            # Parse URL components
            parsed = urlparse(url)
            ext = tldextract.extract(url)

            # Extract components
            domain = ext.domain or ''
            subdomain = ext.subdomain or ''
            suffix = ext.suffix or ''
            path = parsed.path or ''
            query = parsed.query or ''

            # Construct full domain
            full_domain = '.'.join(filter(None, [subdomain, domain, suffix]))

            # Handle subdomains
            subdomains = [part for part in [subdomain] if part]
            if subdomain:
                # Split subdomain by dots to get individual subdomains
                subdomain_parts = subdomain.split('.')
                subdomains = [part for part in subdomain_parts if part]

            # Calculate subdomain metrics
            num_subdomains = len(subdomains)
            avg_subdomain_length = np.mean([len(sub) for sub in subdomains]) if subdomains else 0

            # Count special characters in subdomains
            subdomain_special_chars = sum(self.count_special_characters(sub) for sub in subdomains)
            subdomain_digits = sum(self.count_digits(sub) for sub in subdomains)
            has_digits_in_subdomain = 1 if subdomain_digits > 0 else 0

            # Extract features
            features = {
                # URL-level features
                'url_length': len(url),
                'number_of_dots_in_url': url.count('.'),
                'having_repeated_digits_in_url': self.has_repeated_digits(url),
                'number_of_digits_in_url': self.count_digits(url),
                'number_of_special_char_in_url': self.count_special_characters(url),
                'number_of_hyphens_in_url': url.count('-'),
                'number_of_underline_in_url': url.count('_'),
                'number_of_slash_in_url': url.count('/'),
                'number_of_questionmark_in_url': url.count('?'),
                'number_of_equal_in_url': url.count('='),
                'number_of_percent_in_url': url.count('%'),

                # Domain-level features
                'domain_length': len(full_domain),
                'number_of_dots_in_domain': full_domain.count('.'),
                'number_of_hyphens_in_domain': full_domain.count('-'),
                'having_special_characters_in_domain': self.has_special_characters(full_domain),
                'number_of_special_characters_in_domain': self.count_special_characters(full_domain),
                'having_digits_in_domain': self.has_digits(full_domain),
                'number_of_digits_in_domain': self.count_digits(full_domain),
                'having_repeated_digits_in_domain': self.has_repeated_digits(full_domain),

                # Subdomain features
                'number_of_subdomains': num_subdomains,
                'average_subdomain_length': avg_subdomain_length,
                'number_of_special_characters_in_subdomain': subdomain_special_chars,
                'having_digits_in_subdomain': has_digits_in_subdomain,
                'number_of_digits_in_subdomain': subdomain_digits,

                # Path and query features
                'path_length': len(path),
                'having_query': 1 if query else 0,

                # Entropy features
                'entropy_of_url': self.calculate_entropy(url),
                'entropy_of_domain': self.calculate_entropy(full_domain)
            }

            self.url_features = features
            return features

        except Exception as e:
            print(f"Error extracting features from URL '{url}': {e}")
            return self._get_default_features()

    def _get_default_features(self) -> dict:
        """
        Get default feature values for error cases or empty URLs.

        Returns:
        dict: Dictionary with default feature values.
        """
        return {
            'url_length': 0,
            'number_of_dots_in_url': 0,
            'having_repeated_digits_in_url': 0,
            'number_of_digits_in_url': 0,
            'number_of_special_char_in_url': 0,
            'number_of_hyphens_in_url': 0,
            'number_of_underline_in_url': 0,
            'number_of_slash_in_url': 0,
            'number_of_questionmark_in_url': 0,
            'number_of_equal_in_url': 0,
            'number_of_percent_in_url': 0,
            'domain_length': 0,
            'number_of_dots_in_domain': 0,
            'number_of_hyphens_in_domain': 0,
            'having_special_characters_in_domain': 0,
            'number_of_special_characters_in_domain': 0,
            'having_digits_in_domain': 0,
            'number_of_digits_in_domain': 0,
            'having_repeated_digits_in_domain': 0,
            'number_of_subdomains': 0,
            'average_subdomain_length': 0,
            'number_of_special_characters_in_subdomain': 0,
            'having_digits_in_subdomain': 0,
            'number_of_digits_in_subdomain': 0,
            'path_length': 0,
            'having_query': 0,
            'entropy_of_url': 0,
            'entropy_of_domain': 0
        }

# Example usage and testing
if __name__ == "__main__":
    extractor = FeatureExtractor()

    # Test with example URLs
    test_urls = [
        "https://www.google.com",
        "http://phishing-site123.fake-bank.com/login.php?user=123&pass=456",
        "https://sub1.sub2.example.com/path/to/resource?param=value",
        "http://192.168.1.1:8080/admin/login.html/",
        ""
    ]

    for url in test_urls:
        print(f"\nURL: {url}")
        features = extractor.extract_features(url)
        for feature, value in features.items():
            print(f"  {feature}: {value}")
