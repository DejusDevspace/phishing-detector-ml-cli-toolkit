import re
import math
import tldextract
from urllib.parse import urlparse, parse_qs
from collections import Counter
from itertools import groupby
import numpy as np

class FeatureExtractor:
    """
    A class for extracting URL-based features for machine learning models.
    Extracts lexical, structural, and statistical properties from a given URL.
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
        prob = [float(string.count(c)) / len(string) for c in set(string)]
        return -sum(p * math.log(p) / math.log(2.0) for p in prob)

    def count_special_chars(self, s: str) -> int:
        """
        Count number of special characters in the string.

        Parameters:
        s (str): Input string.

        Returns:
        int: Count of special characters.
        """
        return len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', s))

    def tokenize(self, s: str, delimiters: str = '/.?\-_&=:@') -> list:
        """
        Tokenize a string based on specified delimiters.

        Parameters:
        s (str): String to tokenize.
        delimiters (str): Delimiters to split the string.

        Returns:
        list: List of tokens.
        """
        if not s:
            return []
        regex_pattern = '|'.join(map(re.escape, delimiters))
        return [token for token in re.split(regex_pattern, s) if token]

    def count_digits(self, s: str) -> int:
        """
        Count the number of digits in the string.

        Parameters:
        s (str): Input string.

        Returns:
        int: Number of digits.
        """
        if not s:
            return 0
        return sum(c.isdigit() for c in s)

    def count_letters(self, s: str) -> int:
        """
        Count the number of alphabetic letters in the string.

        Parameters:
        s (str): Input string.

        Returns:
        int: Number of letters.
        """
        if not s:
            return 0
        return sum(c.isalpha() for c in s)

    def count_symbols(self, s: str) -> int:
        """
        Count the number of non-alphanumeric symbols in the string.

        Parameters:
        s (str): Input string.

        Returns:
        int: Number of symbols.
        """
        if not s:
            return 0
        return len(re.findall(r'[^\w]', s))

    def count_delimiters(self, s: str) -> int:
        """
        Count the number of delimiters in the string.

        Parameters:
        s (str): Input string.

        Returns:
        int: Number of delimiters.
        """
        if not s:
            return 0
        delimiters = '/.?\-_&=:@'
        return sum(s.count(d) for d in delimiters)

    def safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide two numbers, returning 0 if denominator is 0.

        Parameters:
        numerator (float): Numerator value.
        denominator (float): Denominator value.

        Returns:
        float: Division result or 0 if denominator is 0.
        """
        return numerator / denominator if denominator != 0 else 0.0

    def safe_max(self, items: list, default: int = 0) -> int:
        """
        Safely get the maximum value from a list, returning default if list is empty.

        Parameters:
        items (list): List of items.
        default (int): Default value if list is empty.

        Returns:
        int: Maximum value or default.
        """
        return max(items) if items else default

    def safe_mean(self, items: list, default: float = 0.0) -> float:
        """
        Safely calculate the mean of a list, returning default if list is empty.

        Parameters:
        items (list): List of items.
        default (float): Default value if list is empty.

        Returns:
        float: Mean value or default.
        """
        return float(np.mean(items)) if items else default

    def extract_features(self, url: str) -> dict:
        """
        Extract a variety of handcrafted features from a URL.

        Parameters:
        url (str): Input URL string.

        Returns:
        dict: Dictionary of extracted feature names and their values.
        """
        if not url:
            # Return a dictionary with all features set to appropriate defaults for empty URLs
            return self._get_default_features()

        try:
            parsed = urlparse(url)
            ext = tldextract.extract(url)
            domain = ext.domain or ''
            subdomain = ext.subdomain or ''
            suffix = ext.suffix or ''
            path = parsed.path or ''
            query = parsed.query or ''

            full_domain = '.'.join(filter(None, [subdomain, domain, suffix]))
            domain_tokens = self.tokenize(domain)
            path_tokens = self.tokenize(path)
            query_vars = parse_qs(query) if query else {}

            # Handle path splitting safely
            path_parts = path.split('/') if path else ['']
            directory_parts = path_parts[:-1] if len(path_parts) > 1 else []
            filename = path_parts[-1] if path_parts else ''

            # Handle file extension safely
            if filename and '.' in filename:
                extension = filename.split('.')[-1]
            else:
                extension = ''

            # Calculate directory path
            directory_path = '/'.join(directory_parts) if directory_parts else ''

            # Extract the features
            # Return -1 for missing features with missing components
            self.url_features = {
                'domain_token_count': len(domain_tokens),
                'avgdomaintokenlen': self.safe_mean([len(t) for t in domain_tokens]),
                'longdomaintokenlen': self.safe_max([len(t) for t in domain_tokens]),
                'avgpathtokenlen': self.safe_mean([len(t) for t in path_tokens]),
                'tld': len(suffix),
                'ldl_url': len(url),
                'ldl_path': len(path),
                'urlLen': len(url),
                'domainlength': len(domain),
                'pathLength': len(path),
                'subDirLen': len(directory_path),
                'pathurlRatio': self.safe_divide(len(path), len(url)),
                'ArgUrlRatio': self.safe_divide(len(query), len(url)),
                'argDomanRatio': self.safe_divide(len(query), len(domain)),
                'domainUrlRatio': self.safe_divide(len(domain), len(url)),
                'pathDomainRatio': self.safe_divide(len(path), len(domain)),
                'argPathRatio': self.safe_divide(len(query), len(path)),
                'NumberofDotsinURL': url.count('.'),
                'CharacterContinuityRate': self.safe_divide(
                    max([len(list(g)) for _, g in groupby(url)]) if url else 0,
                    len(url)
                ),
                'Extension_DigitCount': self.count_digits(extension) if extension else -1,
                'Query_DigitCount': self.count_digits(query) if query else -1,
                'URL_Letter_Count': self.count_letters(url),
                'host_letter_count': self.count_letters(full_domain),
                'Directory_LetterCount': self.count_letters(directory_path) if directory_path else -1,
                'Filename_LetterCount': self.count_letters(filename) if filename else -1,
                'Extension_LetterCount': self.count_letters(extension) if extension else -1,
                'Query_LetterCount': self.count_letters(query) if query else -1,
                'LongestPathTokenLength': self.safe_max([len(t) for t in path_tokens]),
                'Domain_LongestWordLength': self.safe_max([len(t) for t in domain_tokens]),
                'Arguments_LongestWordLength': self.safe_max([len(k) for k in query_vars.keys()]) if query_vars else -1,
                'URLQueries_variable': len(query_vars),
                'spcharUrl': self.count_special_chars(url),
                'delimeter_path': sum(path.count(d) for d in '/-=_.:') if path else 0,
                'delimeter_Count': self.count_delimiters(url) if url else -1,
                'NumberRate_URL': self.safe_divide(self.count_digits(url), len(url)),
                'NumberRate_FileName': self.safe_divide(self.count_digits(filename), len(filename)) if filename else -1,
                'NumberRate_Extension': self.safe_divide(self.count_digits(extension), len(extension)) if extension else -1,
                'NumberRate_AfterPath': self.safe_divide(
                    self.count_digits(parsed.fragment or ''),
                    len(parsed.fragment or '')
                ) if parsed.fragment else -1,
                'SymbolCount_URL': self.count_symbols(url),
                'SymbolCount_Domain': self.count_symbols(full_domain),
                'SymbolCount_Directoryname': self.count_symbols(directory_path) if directory_path else -1,
                'SymbolCount_FileName': self.count_symbols(filename) if filename else -1,
                'SymbolCount_Extension': self.count_symbols(extension) if extension else -1,
                'Entropy_Domain': self.calculate_entropy(full_domain),
                'Entropy_Filename': self.calculate_entropy(filename) if filename else -1
            }

            return self.url_features

        except Exception as e:
            # In case of any error, return a dictionary with all features set to appropriate defaults
            print(f"Error extracting features from URL '{url}': {e}")
            return self._get_default_features()

    def _get_default_features(self) -> dict:
        """
        Get default feature values for error cases or empty URLs.

        Returns:
        dict: Dictionary with default feature values.
        """
        return {
            'domain_token_count': 0,
            'avgdomaintokenlen': 0,
            'longdomaintokenlen': 0,
            'avgpathtokenlen': 0,
            'tld': 0,
            'ldl_url': 0,
            'ldl_path': 0,
            'urlLen': 0,
            'domainlength': 0,
            'pathLength': 0,
            'subDirLen': 0,
            'pathurlRatio': 0,
            'ArgUrlRatio': 0,
            'argDomanRatio': 0,
            'domainUrlRatio': 0,
            'pathDomainRatio': 0,
            'argPathRatio': 0,
            'NumberofDotsinURL': 0,
            'CharacterContinuityRate': 0,
            'Extension_DigitCount': -1,
            'Query_DigitCount': -1,
            'URL_Letter_Count': 0,
            'host_letter_count': 0,
            'Directory_LetterCount': -1,
            'Filename_LetterCount': -1,
            'Extension_LetterCount': -1,
            'Query_LetterCount': -1,
            'LongestPathTokenLength': 0,
            'Domain_LongestWordLength': 0,
            'Arguments_LongestWordLength': -1,
            'URLQueries_variable': 0,
            'spcharUrl': 0,
            'delimeter_path': 0,
            'delimeter_Count': -1,
            'NumberRate_URL': 0,
            'NumberRate_FileName': -1,
            'NumberRate_Extension': -1,
            'NumberRate_AfterPath': -1,
            'SymbolCount_URL': 0,
            'SymbolCount_Domain': 0,
            'SymbolCount_Directoryname': -1,
            'SymbolCount_FileName': -1,
            'SymbolCount_Extension': -1,
            'Entropy_Domain': 0,
            'Entropy_Filename': -1
        }
