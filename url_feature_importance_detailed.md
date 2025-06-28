# Phishing URL Detection Features

| Feature Name                                | Description                                                         | Importance Score |
| ------------------------------------------- | ------------------------------------------------------------------- | ---------------- |
| `url_length`                                | Total length of the URL in characters.                              | 0.1460           |
| `average_subdomain_length`                  | Average length of subdomains in the URL.                            | 0.1205           |
| `entropy_of_domain`                         | Entropy measure of the domain name indicating randomness.           | 0.0984           |
| `entropy_of_url`                            | Entropy measure of the entire URL indicating randomness.            | 0.0966           |
| `domain_length`                             | Length of the domain name in characters.                            | 0.0910           |
| `number_of_digits_in_domain`                | Count of numeric digits present in the domain name.                 | 0.0541           |
| `number_of_subdomains`                      | Total number of subdomains in the URL.                              | 0.0529           |
| `number_of_dots_in_url`                     | Count of dot characters in the entire URL.                          | 0.0450           |
| `number_of_digits_in_url`                   | Count of numeric digits present in the entire URL.                  | 0.0434           |
| `number_of_special_char_in_url`             | Count of special characters in the entire URL.                      | 0.0380           |
| `number_of_slash_in_url`                    | Count of forward slash characters in the URL.                       | 0.0374           |
| `number_of_dots_in_domain`                  | Count of dot characters in the domain name.                         | 0.0278           |
| `path_length`                               | Length of the URL path component in characters.                     | 0.0266           |
| `number_of_hyphens_in_domain`               | Count of hyphen characters in the domain name.                      | 0.0200           |
| `number_of_hyphens_in_url`                  | Count of hyphen characters in the entire URL.                       | 0.0187           |
| `having_digits_in_domain`                   | Binary indicator of whether the domain contains any digits.         | 0.0143           |
| `having_repeated_digits_in_domain`          | Binary indicator of whether the domain contains repeated digits.    | 0.0115           |
| `number_of_digits_in_subdomain`             | Count of numeric digits present in subdomains.                      | 0.0098           |
| `number_of_questionmark_in_url`             | Count of question mark characters in the URL.                       | 0.0067           |
| `having_digits_in_subdomain`                | Binary indicator of whether subdomains contain any digits.          | 0.0067           |
| `number_of_special_characters_in_domain`    | Count of special characters in the domain name.                     | 0.0062           |
| `number_of_equal_in_url`                    | Count of equal sign characters in the URL.                          | 0.0061           |
| `having_query`                              | Binary indicator of whether the URL contains query parameters.      | 0.0061           |
| `number_of_underline_in_url`                | Count of underscore characters in the entire URL.                   | 0.0047           |
| `having_special_characters_in_domain`       | Binary indicator of whether the domain contains special characters. | 0.0040           |
| `having_repeated_digits_in_url`             | Binary indicator of whether the URL contains repeated digits.       | 0.0039           |
| `number_of_percent_in_url`                  | Count of percent sign characters in the URL.                        | 0.0013           |
| `number_of_special_characters_in_subdomain` | Count of special characters in subdomains.                          | 0.0010           |
