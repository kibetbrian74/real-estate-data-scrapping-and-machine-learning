from selenium import webdriver
#from selenium.webdriver.chrome.service import Service
#from selenium.webdriver.common.by import By
#import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import tempfile
from selenium.webdriver.chrome.options import Options
import re
from sklearn.cluster import KMeans
import seaborn as sns
import time

# 1 Scrap through Immobiliare.it (max 100 ads)
def scrape_immobiliare(max_ads=100):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")  # vital for some environments
    options.add_argument("--disable-dev-shm-usage")  # vital for docker environments
    
    #a temporary directory with proper cleanup
    temp_dir = tempfile.mkdtemp(prefix="chrome_data_")
    options.add_argument(f"--user-data-dir={temp_dir}")
    
    # these are additional options to prevent common issues
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")
    
    # Initialize WebDriver with error handling
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Failed to initialize WebDriver: {e}")
        # clean temporary directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        return pd.DataFrame()  #return an empty dataframe incase of failure
    
    ads = []
    page = 1

    try:
        while len(ads) < max_ads:
            url = f"https://www.immobiliare.it/vendita-case/milano/?pag={page}"
            try:
                driver.get(url)
                time.sleep(5)  # give Js time to load
            except Exception as e:
                print(f"Error loading page {page}: {e}")
                break

            soup = BeautifulSoup(driver.page_source, "html.parser")
            # ensure no space between selector
            listings = soup.select("li.nd-list__item.in-searchLayoutListItem")
            
            if not listings:
                print(f"No listings found on page {page}")
                break

            for ad in listings:
                try:
                    title_elem = ad.select_one("a.in-listingCardTitle.is-spaced")
                    price_elem = ad.select_one("div.in-listingCardPrice")
                    
                    if not title_elem or not price_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True)
                    
                    # Clean price - remove all non-digit characters
                    clean_price = re.sub(r'[^\d]', '', price)
                    
                    def extract_location(title):
                        separators = ["apartment", "in", "via"]
                        for separator in separators:
                            parts = title.split(separator, 1)
                            if len(parts) > 1:
                                return parts[1].strip()
                        return title  #title is returned as a fallback
                    
                    location = extract_location(title)
                    size = "0"  #default values incase no title is found
                    
                    feature_items = ad.find_all("div", class_="in-listingCardFeatureList__item")
                    for item in feature_items:
                        item_text = item.get_text(strip=True)
                        if "m²" in item_text:
                            # extract numeric value in size, no m2
                            size_match = re.search(r'(\d+)', item_text)
                            if size_match:
                                size = size_match.group(1)
                            break
                    
                    ads.append({
                        "title": title,
                        "price": clean_price,
                        "location": location,
                        "size_sqm": size
                    })
                    
                    if len(ads) >= max_ads:
                        break
                except Exception as e:
                    print(f"Error processing ad: {e}")
                    continue
                    
            page += 1
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        driver.quit()
        # Clean up the temporary directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    #convert to df and ensure numeric types
    immob_df = pd.DataFrame(ads)
    immob_df['price'] = pd.to_numeric(immob_df['price'], errors='coerce').fillna(0).astype(int)
    immob_df['size_sqm'] = pd.to_numeric(immob_df['size_sqm'], errors='coerce').fillna(0).astype(int)
    
    return immob_df #return a dataframe

# 2 simulate data from Idealista and Subito
def idealista_subito_it():
    np.random.seed(42)
    df_idealista = pd.DataFrame({
        "title": ["Apartment in Navigli area"] * 70,
        "price": np.random.randint(150000, 600000, size=70),
        "size_sqm": np.random.randint(30, 120, size=70),
        "location": ["Navigli"] *70,
        "source": ["Idealista"]* 70
    })

    df_subito = pd.DataFrame({
        "title": ["Studio in Centrale"]*70,
        "price": np.random.randint(80000, 200000, size=70),
        "size_sqm": np.random.randint(20, 60,size=70),
        "location": ["Centrale"]*70,
        "source": ["Subito"]*70
    })

    return pd.concat([df_idealista, df_subito], ignore_index=True) # combine idealista data and that from subito

# 3 combine and clean the data data
def mergeandclean(df_scraped, df_simulated):
    df_scraped["source"] = "Immobiliare"
    df_all = pd.concat([df_scraped, df_simulated], ignore_index=True) # merge

    df_all["price"] = pd.to_numeric(df_all["price"], errors="coerce") # convert prices to numerical
    df_all["size_sqm"] = pd.to_numeric(df_all["size_sqm"], errors="coerce") # convert sizes to numerical

    df_all = df_all.dropna() # remove NaNs or empty cells
    df_all = df_all[df_all["size_sqm"] > 20] # size must be greater than 20sqm
    df_all = df_all[df_all["price"] < 1000000] # prices must be less than 1 million

    return df_all

# 4 Visualization
def visualize_data(df):
    df.groupby("source")["price"].mean().plot(kind="bar", title="Average Price by Source")
    plt.ylabel("Price (EURO)")
    plt.tight_layout()
    plt.show()

    plt.scatter(df["size_sqm"], df["price"]) # a scatter plot of size vs price
    plt.xlabel("Size (sqm)")
    plt.ylabel("Price (EURO)")
    plt.title("Price vs Apartment Size")
    plt.grid(True)
    plt.tight_layout() #ensure everything fits in the plot
    plt.show()

# 5 linear regression
def lr_run(df):
    X = df[["size_sqm"]]
    y = df["price"]
    # random_ensures that every time the code is run, the split is exactly the same
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    #0.2 test size means 20% of df is used as testing data, the remainder was used as training data

    lr_model = LinearRegression() # create a LR model instance
    lr_model.fit(X_train, y_train) # train the model
    y_pred = lr_model.predict(X_test) # prediction

    print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}") # displays the r2 score value for the linear regression model

    plt.scatter(X_test, y_test, label="Actual", alpha=0.7)
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Size (sqm)")
    plt.ylabel("Price (EURO)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 6 Clustering
def kmeans_Clust(df):
    km = KMeans(n_clusters=5, random_state=42) # kmeans clustering model is created from scikit-learn
    # is set to find 5 clusters in the data
    df["Cluster"] = km.fit_predict(df[["price", "size_sqm"]]) #applies kmeans model to the "price" and "size_sqm" columns

    sns.scatterplot(data=df, x="size_sqm", y="price", hue="Cluster", palette="Set2") #uses seaborn to create a scatterplot
    plt.title("Clustering Apartments by Price and Size")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": #this checks if this script is being run directly,ie,not imported as a module
# if so, it runs the steps below
    print("Scraping Immobiliare.it...")
    df_scraped = scrape_immobiliare(max_ads=100) #scrapes real estate data from imobiliare (up to 100 ads)

    print("Simulating Idealista and Subito.it listings...")
    df_simulated = idealista_subito_it() # fetches listings from idealista and subito

    print("Combining and cleaning datasets...")
    df = mergeandclean(df_scraped, df_simulated) # does the merging & cleaning of both datasets into 1 df called df

    print("Visualizing data...")
    visualize_data(df) # this one generates visualizations from the combined dataset (df)

    print("Running regression model...")
    lr_run(df) #runs a linear regression model to predict price

    print("Running clustering analysis...")
    # finally, this line calls the earlier-defined kmeans_Clust func to perform clustering and plot the outputs
    kmeans_Clust(df)