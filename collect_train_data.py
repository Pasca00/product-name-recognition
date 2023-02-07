import pandas as pd
import requests
import lxml.html
from lxml.html.clean import Cleaner
import re
import argparse
import unicodedata
from lxml.etree import tostring

pattern = r"[\w']+|[.,!?;=<>_\-\"/]"

def write_element_text_to_file(file, element, is_product):
    text = unicodedata.normalize(
        'NFKD', 
        tostring(element).decode()
    ).encode('ascii', 'ignore').decode().strip()

    # text = unicodedata.normalize(
    #     'NFKD', 
    #     element.text_content()
    # ).encode('ascii', 'ignore').decode().strip()

    if text == '':
        return

    words = re.findall(pattern, text)
    text_content = False
    start_of_text = False
    for i, w in enumerate(words):
        if text_content and w == '<':
            text_content = False

        if is_product and text_content:
            label = 'B-P' if start_of_text else 'I-P'
            start_of_text = False
        else:
            label = 'O'
        
        if w == '>':
            text_content = True
            start_of_text = True
        
        file.write(w + ' ' + label + '\n')

    file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--valid_pages', default=100, help='Number of valid pages visited required', type=int)
    opt = parser.parse_args()

    websites = pd.read_csv('input/furniture stores pages.csv')

    products = set()

    i = -1
    valid_pages = 0
    sentences = []
    labels = []
    while valid_pages < opt.valid_pages:
        i += 1
        print(f'(Curr website index: { i }, Curr valid website index: { valid_pages })')

        try:
            response = requests.get(websites.iloc[i]['max(page)'])
            if response.status_code == 200:
                html = response.content

                cleaner = Cleaner()
                cleaner.comments = True
                cleaner.style = True
                cleaner.scripts = True
                html_tree = cleaner.clean_html(lxml.html.fromstring(html)).find('body')
                
                if html_tree is not None:
                    is_buy_page = True
                    for element in html_tree.iter():
                        if element.__class__.__name__ == 'HtmlElement' and 'add to cart' in element.text_content().lower():
                            is_buy_page = True
                            break
                    
                    if is_buy_page:
                        for element in html_tree.iter():
                            if len(element) == 0:
                                classes = element.attrib.get('class')
                                href = element.attrib.get('href')
                                text = unicodedata.normalize(
                                    'NFKD', 
                                    element.text_content()
                                ).encode('ascii', 'ignore').decode().strip()

                                if text == '':
                                    continue
                                
                                sentences.append(str(text))

                                if (classes and 'product' in classes.lower() and 'title' in classes.lower()) \
                                    or (href and ('/product/' in href.lower() or '/products/' in href.lower())):
                                    labels.append(1)
                                else:
                                    labels.append(0)

                    valid_pages += 1
                    
                    
            else:
                print(f'Website { response.url } has returned status code { response.status_code }')
        except requests.exceptions.RequestException as e:
            continue


    df = pd.DataFrame(data={'label': labels, 'text': sentences})
    df.to_csv('./train_data/data.csv', index=False)