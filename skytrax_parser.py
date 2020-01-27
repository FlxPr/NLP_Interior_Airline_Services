from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from math import floor
import time

link_file_name = 'skytrax_reviews_links.txt'
data_file_name = 'skytrax_reviews_data.csv'

# Makes selenium open Chrome in background
chrome_options = Options()
# UNCOMMENT TO HAVE BROWSER RUN IN BACKGROUND
# chrome_options.add_argument("--headless")

def get_links():
    # USE THE SAME LINK LIST AS I SENT AND DO NOT CALL THIS (the links will be in a different order and we might miss some pages)
    links = list()
    browser = webdriver.Chrome(chrome_options=chrome_options)
    for review_link in ['https://www.airlinequality.com/review-pages/a-z-{review_type}-reviews/'
                        .format(review_type=review_type) for review_type in ('airline', 'seat')]:
        browser.get(review_link)
        airline_lists = [airline_list.find_elements_by_tag_name('li')
                         for airline_list in browser.find_elements_by_class_name('items')]
        for airline_list in airline_lists:
            for airline in airline_list:
                link = airline.find_element_by_tag_name('a').get_attribute('href')
                links.append(link)
    browser.close()
    links = sorted(list(set(links)))
    # Save unique links in text file
    link_file = open(link_file_name, 'wt')
    link_file.write('\n'.join(links))


links = open(link_file_name).read().split('\n')
try:
    data = pd.read_csv(data_file_name, index_col=None)
except FileNotFoundError:
    data = pd.DataFrame(columns=['Aircraft',
                                 'Aircraft Type',
                                 'Aisle Space',
                                 'Cabin Staff Service',
                                 'Date Flown',
                                 'Food & Beverages',
                                 'Ground Service',
                                 'Inflight Entertainment',
                                 'Power Supply',
                                 'Recommended',
                                 'Route',
                                 'Seat Comfort',
                                 'Seat Layout',
                                 'Seat Legroom',
                                 'Seat Privacy',
                                 'Seat Recline',
                                 'Seat Storage',
                                 'Seat Type',
                                 'Seat Width',
                                 'Seat/bed Length',
                                 'Seat/bed Width',
                                 'Sitting Comfort',
                                 'Sleep Comfort',
                                 'Type Of Traveller',
                                 'Unnamed: 0',
                                 'Unnamed: 0.1',
                                 'Value For Money',
                                 'Viewing Tv Screen',
                                 'Wifi & Connectivity',
                                 'airline',
                                 'best_rating',
                                 'comment',
                                 'comment_date',
                                 'header',
                                 'rating',
                                 'review_type'])
    data.to_csv(data_file_name)


moritz_part = slice(340, 430)
noel_part = slice(430, 520)
felix_part = slice(540, 610)
shamir_part = slice(610, 700)

browser = webdriver.Chrome(chrome_options=chrome_options)
reviews_per_page = 50
for link in links[felix_part]:
    tic = time.time()
    browser.get(link + '/?sortby=post_date%3ADesc&pagesize={}'.format(reviews_per_page))  # show all reviews in one page
    airline_name = browser.find_element_by_class_name('info').find_element_by_tag_name('h1').text
    review_type = browser.find_element_by_class_name('info').find_element_by_tag_name('h2').text
    review_count = int(browser.find_element_by_class_name('review-count').text.split(' ')[-2])
    if review_type in data[data.airline == airline_name].review_type.unique():
        print('Already scraped {review_type} for {airline_name}'
              .format(review_type=review_type, airline_name=airline_name))
        continue
    print('Scraping {airline} {review}'.format(airline=airline_name, review=review_type))

    data_lines = []
    number_of_pages = review_count//reviews_per_page + 1 if review_count % reviews_per_page != 0 \
        else review_count//reviews_per_page
    for page_number in range(1, number_of_pages + 1):

        print('scraping page {} out of {}'.format(page_number, number_of_pages))
        reviews_container = browser.find_element_by_tag_name('article')
        reviews = reviews_container.find_elements_by_tag_name('article')
        for review in reviews:
            data_line = {'airline': airline_name, 'review_type': review_type}
            try:
                data_line['rating'], data_line['best_rating'] = review.find_element_by_class_name('rating-10').text.split('/')
            except ValueError:
                pass
            data_line['header'] = review.find_element_by_class_name('text_header').text
            data_line['comment_date'] = review.find_element_by_tag_name('time').get_attribute('datetime')
            data_line['comment'] = review.find_element_by_class_name('text_content').text
            ratings_table = review.find_element_by_class_name('review-ratings').find_element_by_tag_name('tbody')
            for rating in ratings_table.find_elements_by_tag_name('tr'):
                try:
                    key_value = rating.find_elements_by_tag_name('td')
                    if 'stars' in key_value[1].get_attribute('class'):
                        number_of_stars = len(key_value[1].find_elements_by_class_name('star.fill '))
                        data_line[key_value[0].text] = number_of_stars
                    else:
                        data_line[key_value[0].text] = key_value[1].text
                except Exception:
                    pass
            data_lines.append(data_line)
            time.sleep(0.5)
        browser.get(link + '/page/{}/?sortby=post_date%3ADesc&pagesize={}'.format(page_number, reviews_per_page))

    data = data.append(pd.DataFrame(data_lines))
    data.to_csv(data_file_name, index=None)
    toc = time.time()
    print('Scraped {n_records} in {seconds} seconds'.format(n_records=len(data_lines), seconds=floor(toc-tic)))
    time.sleep(1)
