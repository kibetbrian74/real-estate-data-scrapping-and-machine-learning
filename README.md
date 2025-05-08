# real-estate-data-scrapping-and-machine-learning

This Python script analyzes Milan's real estate market by scraping up to 100 listings from Immobiliare.it, simulating additional data from Idealista and Subito.it, and merging all sources for analysis. It cleans the data, visualizes pricing trends, applies a linear regression model to predict property prices based on size, and uses K-Means clustering to group similar listings by price and size.

The code begins by importing necessary libraries:
**Web scraping**: `selenium`, `BeautifulSoup`
**Data handling**: `pandas`, `numpy`
**Visualization**: `matplotlib`, `seaborn`
**Machine learning**: `sklearn` (for regression and clustering)
**Utilities**: `os`, `tempfile`, `re`, `time`
