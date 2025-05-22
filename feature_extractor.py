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
        return np.mean(items) if items else default

    def extract_features(self, url: str) -> dict:
        """
        Extract a variety of handcrafted features from a URL.

        Parameters:
        url (str): Input URL string.

        Returns:
        dict: Dictionary of extracted feature names and their values.
        """
        if not url:
            # Return a dictionary with all features set to 0 for empty URLs
            return {f'feature_{i}': 0 for i in range(50)}  # Adjust based on actual feature count

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

            # Handle path splitting more safely
            path_parts = path.split('/') if path else ['']
            directory_parts = path_parts[:-1] if len(path_parts) > 1 else []
            filename = path_parts[-1] if path_parts else ''

            # Handle file extension more safely
            if filename and '.' in filename:
                extension = filename.split('.')[-1]
            else:
                extension = ''

            # Calculate directory path
            directory_path = '/'.join(directory_parts) if directory_parts else ''

            self.url_features = {
                'Querylength': len(query),
                'domain_token_count': len(domain_tokens),
                'path_token_count': len(path_tokens),
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
                'LongestVariableValue': self.safe_max([
                    len(v[0]) for v in query_vars.values() if v
                ]) if query_vars else 0,
                'URL_DigitCount': self.count_digits(url),
                'Directory_DigitCount': self.count_digits(directory_path),
                'Extension_DigitCount': self.count_digits(extension),
                'URL_Letter_Count': self.count_letters(url),
                'host_letter_count': self.count_letters(full_domain),
                'Directory_LetterCount': self.count_letters(directory_path),
                'Filename_LetterCount': self.count_letters(filename),
                'Extension_LetterCount': self.count_letters(extension),
                'Query_LetterCount': self.count_letters(query),
                'LongestPathTokenLength': self.safe_max([len(t) for t in path_tokens]),
                'Domain_LongestWordLength': self.safe_max([len(t) for t in domain_tokens]),
                'Arguments_LongestWordLength': self.safe_max([len(k) for k in query_vars.keys()]),
                'URLQueries_variable': len(query_vars),
                'spcharUrl': self.count_special_chars(url),
                'delimeter_path': sum(path.count(d) for d in '/-=_.:') if path else 0,
                'NumberRate_URL': self.safe_divide(self.count_digits(url), len(url)),
                'NumberRate_FileName': self.safe_divide(self.count_digits(filename), len(filename)),
                'NumberRate_Extension': self.safe_divide(self.count_digits(extension), len(extension)),
                'NumberRate_AfterPath': self.safe_divide(
                    self.count_digits(parsed.fragment or ''),
                    len(parsed.fragment or '')
                ),
                'SymbolCount_URL': self.count_symbols(url),
                'SymbolCount_Domain': self.count_symbols(full_domain),
                'SymbolCount_Directoryname': self.count_symbols(directory_path),
                'SymbolCount_FileName': self.count_symbols(filename),
                'SymbolCount_Extension': self.count_symbols(extension),
                'Entropy_Domain': self.calculate_entropy(full_domain),
                'Entropy_DirectoryName': self.calculate_entropy(directory_path)
            }

            return self.url_features

        except Exception as e:
            # In case of any error, return a dictionary with all features set to 0
            print(f"Error extracting features from URL '{url}': {e}")
            # You might want to define the exact number of features here
            default_features = {
                'Querylength': 0, 'domain_token_count': 0, 'path_token_count': 0,
                'avgdomaintokenlen': 0, 'longdomaintokenlen': 0, 'avgpathtokenlen': 0,
                'tld': 0, 'ldl_url': 0, 'ldl_path': 0, 'urlLen': 0, 'domainlength': 0,
                'pathLength': 0, 'subDirLen': 0, 'pathurlRatio': 0, 'ArgUrlRatio': 0,
                'argDomanRatio': 0, 'domainUrlRatio': 0, 'pathDomainRatio': 0,
                'argPathRatio': 0, 'NumberofDotsinURL': 0, 'CharacterContinuityRate': 0,
                'LongestVariableValue': 0, 'URL_DigitCount': 0, 'Directory_DigitCount': 0,
                'Extension_DigitCount': 0, 'URL_Letter_Count': 0, 'host_letter_count': 0,
                'Directory_LetterCount': 0, 'Filename_LetterCount': 0, 'Extension_LetterCount': 0,
                'Query_LetterCount': 0, 'LongestPathTokenLength': 0, 'Domain_LongestWordLength': 0,
                'Arguments_LongestWordLength': 0, 'URLQueries_variable': 0, 'spcharUrl': 0,
                'delimeter_path': 0, 'NumberRate_URL': 0, 'NumberRate_FileName': 0,
                'NumberRate_Extension': 0, 'NumberRate_AfterPath': 0, 'SymbolCount_URL': 0,
                'SymbolCount_Domain': 0, 'SymbolCount_Directoryname': 0, 'SymbolCount_FileName': 0,
                'SymbolCount_Extension': 0, 'Entropy_Domain': 0, 'Entropy_DirectoryName': 0
            }
            return default_features
