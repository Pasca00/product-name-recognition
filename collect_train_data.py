import pandas as pd
import requests
import lxml.html
from lxml.html.clean import Cleaner
import re
import argparse
import unicodedata
from lxml.etree import tostring

pattern = r"[\w']+|[.,!?;]"

def write_element_text_to_file(file, element, is_product):
    # text = unicodedata.normalize(
    #     'NFKD', 
    #     tostring(element)
    # ).encode('ascii', 'ignore').decode().strip()

    text = unicodedata.normalize(
        'NFKD', 
        element.text_content()
    ).encode('ascii', 'ignore').decode().strip()

    if text == '':
        return

    words = re.findall(pattern, text)
    for i, w in enumerate(words):
        if is_product:
            label = 'B-P' if i == 0 else 'I-P'
        else:
            label = 'O'
        
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
                    
                    products_elements = set()
                    if is_buy_page:
                        for element in html_tree.iter():
                            classes = element.attrib.get('class')
                            if classes and 'product' in classes.lower() and 'title' in classes.lower():
                                products_elements.add(element)
                    
                    if len(products_elements) == 0:
                        continue
                    
                    data_file = open(f'./train_data/in{valid_pages}', 'w')

                    for element in html_tree.iter():
                        if len(element) == 0:
                            write_element_text_to_file(
                                data_file, 
                                element, 
                                True if element in products_elements else False
                            )

                    data_file.close()

                    valid_pages += 1
                    
                    
            else:
                print(f'Website { response.url } has returned status code { response.status_code }')
        except requests.exceptions.RequestException as e:
            continue


    
    # if not os.path.exists('./text/'):
    #     os.makedirs('./text/')

    # words_file = open(f'./text/in', 'w')
    # for p in products:
    #     print(p)
    #     words = p.split()
    #     for j, word in enumerate(words):
    #         c = 'B-P\n' if j == 0 else 'I-P\n'
    #         words_file.write(word + ' ' + c)

    # words_file.close()