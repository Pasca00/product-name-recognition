import pandas as pd
import requests
import lxml.html
from lxml.html.clean import Cleaner
import unicodedata
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./product-seq-classification.model/')
    model = AutoModelForSequenceClassification.from_pretrained('./product-seq-classification.model/')

    cleaner = Cleaner()
    cleaner.comments = True
    cleaner.style = True
    cleaner.scripts = True

    pages = []
    products = []

    websites = pd.read_csv('./input/furniture stores pages.csv')
    print('Searching for products...')
    for i in tqdm(range(len(websites))):
        try:
            response = requests.get(websites.iloc[i]['max(page)'])
            if response.status_code == 200:
                html = response.content
                html_tree = cleaner.clean_html(lxml.html.fromstring(html)).find('body')

                if html_tree is not None:
                    for element in html_tree.iter():
                        if len(element) == 0:
                            text = unicodedata.normalize(
                                'NFKD', 
                                element.text_content()
                            ).encode('ascii', 'ignore').decode().strip()

                            tokenized_text = tokenizer(text, return_tensors='pt', truncation=True)
                            with torch.no_grad():
                                logits = model(**tokenized_text).logits
    
                            predicted_class_id = logits.argmax().item()
                            if model.config.id2label[predicted_class_id] == 'PRODUCT':
                                pages.append(websites.iloc[i]['max(page)'])
                                products.append(text)

        except requests.exceptions.RequestException as e:
            continue

    df = pd.DataFrame({'website': pages, 'product_name': products})
    df.to_csv('./products_final.csv')