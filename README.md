# product-name-recognition

This model was trained to extract all product names from a given list of websites. First, to build the dataset, I crawled 100 valid pages from the given 
list of URLs (a page is considered valid if it returned the status code 200 and if it contains a product), and extracted product names given
a few rules: if there is an 'Add to cart' button on the page, if there are HTML elements with the classes containing 'product' and 'title' or if there
are links to a different page, whose URL contains '/product/' or '/products/'. This whole step of automatic data acquisition and labeling is done
by running [collect_train_data.py](collect_train_data.py).
```
python collect_train_data.py [--valid_pages=100]
```
The model is trained to classify text as either a product name or not, thus, the collected data is in the form of (sentence, label) and can be found
in [train_data/data.csv](train_data/data.csv).
After collecting the data, I did some manual data cleansing to correct some of the wrong labeling. \
\
For the model itself, I used the huggingface distilbert_base_uncased pretrained model for sequence classification and trained it to label text as either PRODUCT (1) or NONE (0).
\
To run the model and extract the required data from all ~700 websites:
```
python model.py
```

## Results
Given the list of 700+ websites, the model extracted a total of 3552 products. The model could still be improved, as it includes duplicates and sometimes text
that does not represent the name of a product (e.g. 'Buy', 'View'). These errors come from mislabeled training data.\
A preview of the results:
website | product_name
--- | ---
https://www.factorybuys.com.au/products/euro-top-mattress-king | Factory Buys 32cm Euro Top Mattress - King
https://themodern.net.au/products/hamar-plant-stand-ash        | Hamar Plant Stand - Ash
https://themodern.net.au/products/hamar-plant-stand-ash        | Addison Table Lamp - Jade
https://dhfonline.com/products/gift-card                       | Gift Card
https://dhfonline.com/products/gift-card                       | TARTS DONNA'S SIGNATURE SCENT CANDLE
https://claytongrayhome.com/products/palecek-coco-ruffle-mirror|Rogan Planter Tall
https://claytongrayhome.com/products/palecek-coco-ruffle-mirror|Hallie Wall Decor

The full list can be found in [products_final.csv](products_final.csv).
