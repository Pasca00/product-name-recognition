import pandas as pd
import requests
import lxml.html
import os
import argparse

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
                html_parser = lxml.html.HTMLParser(remove_comments=True)
                html_tree = lxml.html.fromstring(html, parser=html_parser).find('body')
                if html_tree is not None:
                    valid_pages += 1

                    is_buy_page = True
                    for element in html_tree.iter():
                        if element.__class__.__name__ == 'HtmlElement' and 'add to cart' in element.text_content().lower():
                            is_buy_page = True
                            break

                    if is_buy_page:
                        for element in html_tree.iter():
                            classes = element.attrib.get('class')
                            if classes and 'product' in classes.lower() and 'title' in classes.lower():
                                text = ''.join(ch for ch in element.text_content() if ch.isalnum() or ch == ' ' or ch == '-')
                                products.add(text.strip())
            else:
                print(f'Website { response.url } has returned status code { response.status_code }')
        except requests.exceptions.RequestException as e:
            continue


    
    if not os.path.exists('./text/'):
        os.makedirs('./text/')

    words_file = open(f'./text/in', 'w')
    for p in products:
        print(p)
        # words = p.split()
        # for j, word in enumerate(words):
        #     c = 'B-P\n' if j == 0 else 'I-P\n'
        #     words_file.write(word + ' ' + c)

    words_file.close()