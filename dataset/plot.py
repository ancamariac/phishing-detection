import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Încarcăm setul de date din CSV
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Listăm coloanele relevante pentru analiză
col_names = ['NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
             'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
             'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
             'NumNumericChars', 'NoHttps', 'IpAddress', 'DomainInSubdomains',
             'DomainInPaths', 'HostnameLength', 'PathLength',
             'QueryLength', 'DoubleSlashInPath', 'ExtFavicon', 'InsecureForms']



# Selecționăm doar coloanele relevante
selected_data = data[col_names]

# Creăm o matrice de corelație
correlation_matrix = selected_data.corr()

# Creăm un heatmap pentru a vizualiza matricea de corelație
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matrice de Corelație între Caracteristici')
plt.show()