# real-estate-data-scrapping-and-machine-learning

This Python script analyzes Milan's real estate market by scraping up to 100 listings from Immobiliare.it, simulating additional data from Idealista and Subito.it, and merging all sources for analysis. It cleans the data, visualizes pricing trends, applies a linear regression model to predict property prices based on size, and uses K-Means clustering to group similar listings by price and size.

**1. Imports**

The code begins by importing necessary libraries:

**Web scraping**: `selenium`, `BeautifulSoup`

**Data handling**: `pandas`, `numpy`

**Visualization**: `matplotlib`, `seaborn`

**Machine learning**: `sklearn` (for regression and clustering)

**Utilities**: `os`, `tempfile`, `re`, `time`

**2. `scrape_immobiliare(max_ads=100)`**

This function:

- Sets up a headless Chrome browser using Selenium.

- Iteratively navigates pages of Immobiliare.it, scraping up to 100 ads.

- Extracts:

-- Title

-- Price (cleans it using regex to retain digits)

-- Location (parsed from title)

-- Size in square meters (parsed from listing features)

- Saves all listings into a DataFrame.

- Handles errors and cleans up temporary Chrome profile data.

**3. `idealista_subito_it()`**

This function:

- Simulates data for two other real estate platforms (Idealista and Subito).

- Creates fake listings with realistic random prices and sizes.

- Returns a combined DataFrame of 140 entries (70 per platform).

**4. `mergeandclean(df_scraped, df_simulated)`**

This function:

- Merges scraped and simulated data.

- Converts price and size columns to numeric.

- Removes invalid entries (e.g., very small apartments or overly expensive listings).

- Returns a cleaned dataset.

**5. `visualize_data(df)`**

This function shows:

a. A bar chart of average prices per platform.

b. A scatter plot of apartment size vs. price.

**6. `lr_run(df)`**

This function:

- Splits the data into training and test sets.

- Trains a linear regression model to predict price from apartment size.

- Prints the R² score and shows a prediction vs. actual plot.

**7. `kmeans_Clust(df)`**

This function:

- Applies K-Means clustering (5 clusters) based on price and size.

- Visualizes the clusters in a scatter plot using different colors.

**8. Main Block `(if __name__ == "__main__":)`**

Executes the workflow:

- Scrapes data

- Simulates extra listings

- Merges and cleans all data

- Visualizes it

- Runs linear regression

- Runs clustering analysis
