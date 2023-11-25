from urllib.parse import urlparse

def count_dots(url):
   """
   Counts the number of dots in the given URL.

   Returns:
   - int: The number of dots in the URL.
   """
   return url.count('.')
 
def get_subdomain_level(url):
   """
   Extracts the subdomain level from the given URL, excluding 'www'.

   Returns:
   - int: The subdomain level.
   """
   parsed_url = urlparse(url)
   subdomains = parsed_url.netloc.split('.')
   
   # Remove 'www' from the list of subdomains
   if subdomains[0].lower() == 'www':
      subdomains = subdomains[1:]

   # Remove domain suffix from the list of subdomains
   subdomains = subdomains[:-1]

   return len(subdomains)

def get_path_level(url):
   """
   Extracts the path level from the given URL.

   Returns:
   - int: The path level.
   """
   parsed_url = urlparse(url)
   path = parsed_url.path
   
   # Remove trailing slash if it exists
   path_components = path.split('/')
   # Remove empty components
   path_components = [component for component in path_components if component]

   return len(path_components)

def get_url_length(url):
   """
   Calculates the length of the given URL.

   Returns:
   - int: The length of the URL.
   """
   return len(url)

def count_dashes(url):
   """
   Counts the number of dashes in the given URL.

   Returns:
   - int: The number of dashes in the URL.
   """
   return url.count('-')

def count_dashes_in_hostname(url):
   """
   Counts the number of dashes in the hostname of the given URL.

   Returns:
   - int: The number of dashes in the hostname.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   return hostname.count('-')

def count_at_symbol(url):
   """
   Counts the number of at symbols in the given URL.

   Returns:
   - int: The number of at symbols in the URL.
   """
   return url.count('@')

def count_tilde_symbol(url):
   """
   Counts the number of tilde symbols in the given URL.

   Returns:
   - int: The number of tilde symbols in the URL.
   """
   return url.count('~')

def count_underscore_symbol(url):
   """
   Counts the number of underscore symbols in the given URL.

   Returns:
   - int: The number of underscore symbols in the URL.
   """
   return url.count('_')

def count_percent_symbol(url):
   """
   Counts the number of percent symbols in the given URL.

   Returns:
   - int: The number of percent symbols in the URL.
   """
   return url.count('%')

def count_query_components(url):
   """
   Counts the number of query components in the given URL.

   Returns:
   - int: The number of query components in the URL.
   """
   parsed_url = urlparse(url)
   query = parsed_url.query
   print(query)
   query_components = query.split('&')
   return len(query_components)

def count_amperstand_symbol(url):
   """
   Counts the number of ampersand symbols in the given URL.

   Returns:
   - int: The number of ampersand symbols in the URL.
   """
   return url.count('&')

def count_hash_symbol(url):
   """
   Counts the number of hash symbols in the given URL.

   Returns:
   - int: The number of hash symbols in the URL.
   """
   return url.count('#')

def count_numeric_characters(url):
   """
   Counts the number of numeric characters in the given URL.

   Returns:
   - int: The number of numeric characters in the URL.
   """
   return sum(c.isdigit() for c in url)

def check_https(url):
   """
   Checks if the given URL uses HTTPS.

   Returns:
   - bool: True if the URL uses HTTPS, False otherwise.
   """
   parsed_url = urlparse(url)    
   return int(parsed_url.scheme == 'https')

def check_ip_address(url):
   """
   Checks if the given URL is an IP address.

   Returns:
   - bool: True if the URL is an IP address, False otherwise.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   hostname_components = hostname.split('.')
   return int(all(component.isdigit() for component in hostname_components))

def check_domain_in_subdomains(url):
   """
   Checks if the domain is in the subdomains of the given URL.

   Returns:
   - bool: True if the domain is in the subdomains, False otherwise.
   """
   parsed_url = urlparse(url)
   domain = parsed_url.netloc.split('.')[-1]
   subdomains = parsed_url.netloc.split('.')[:-1]
   return int(domain in subdomains)

def check_domain_in_path(url):
   """
   Checks if the domain is in the path of the given URL.

   Returns:
   - bool: True if the domain is in the path, False otherwise.
   """
   parsed_url = urlparse(url)
   domain = parsed_url.netloc.split('.')[-1]
   path = parsed_url.path
   return int(domain in path)

def check_https_in_hostname(url):
   """
   Checks if HTTPS is in the hostname of the given URL.

   Returns:
   - bool: True if HTTPS is in the hostname, False otherwise.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   return int('https' in hostname)

def count_hostname_length(url):
   """
   Counts the length of the hostname of the given URL.

   Returns:
   - int: The length of the hostname.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   return len(hostname)

def count_path_length(url):
   """
   Counts the length of the path of the given URL.

   Returns:
   - int: The length of the path.
   """
   parsed_url = urlparse(url)
   path = parsed_url.path
   return len(path)

def count_query_length(url):
   """
   Counts the length of the query of the given URL.

   Returns:
   - int: The length of the query.
   """
   parsed_url = urlparse(url)
   query = parsed_url.query
   return len(query)

def check_double_slash_in_paths(url):
   """
   Checks if there are double slashes in the paths of the given URL.

   Returns:
   - bool: True if there are double slashes in the paths, False otherwise.
   """
   parsed_url = urlparse(url)
   path = parsed_url.path
   return int('//' in path)

def check_ext_favicon(url):
   """
   Checks if the URL has an external favicon.

   Returns:
   - bool: True if the URL has an external favicon, False otherwise.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   return int('favicon' in hostname)

def check_insecure_forms(url):
   """
   Checks if the URL has insecure forms.

   Returns:
   - bool: True if the URL has insecure forms, False otherwise.
   """
   parsed_url = urlparse(url)
   hostname = parsed_url.netloc
   return int('forms' in hostname)

def feature_extraction(url):
   dots_count = count_dots(url)
   subdomain_level = get_subdomain_level(url)
   path_level = get_path_level(url)
   url_length = get_url_length(url)
   dashes_count = count_dashes(url)
   dashes_count_hostname = count_dashes_in_hostname(url)
   at_symbol_count = count_at_symbol(url)
   tilde_count = count_tilde_symbol(url)
   underscore_count = count_underscore_symbol(url)
   percent_count = count_percent_symbol(url)
   query_count = count_query_components(url)
   ampersand_count = count_amperstand_symbol(url)
   hash_count = count_hash_symbol(url)
   numeric_count = count_numeric_characters(url)
   https = check_https(url)
   ip_address = check_ip_address(url)
   domain_in_subdomains = check_domain_in_subdomains(url)
   domain_in_path = check_domain_in_path(url)
   https_in_hostname = check_https_in_hostname(url)
   hostname_length = count_hostname_length(url)
   path_length = count_path_length(url)
   query_length = count_query_length(url)
   double_slash_in_paths = check_double_slash_in_paths(url)
   ext_favicon = check_ext_favicon(url)
   insecure_forms = check_insecure_forms(url)
   
   return [dots_count, subdomain_level, path_level, url_length, dashes_count, 
           dashes_count_hostname, at_symbol_count, tilde_count, underscore_count, 
           percent_count, query_count, ampersand_count, hash_count, numeric_count, 
           https, ip_address, domain_in_subdomains, domain_in_path, https_in_hostname, 
           hostname_length, path_length, query_length, double_slash_in_paths, ext_favicon, insecure_forms]
   

url = 'https://www.google.com/search?q=hello+world&rlz=1C1CHBF_enUS911US911&oq=hello+world&aqs=chrome..69i57j0l7.1234j0j7&sourceid=chrome&ie=UTF-8'
result = feature_extraction(url)
print(f"Feature extraction : {result}.")

