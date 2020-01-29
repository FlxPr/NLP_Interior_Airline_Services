from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from math import floor
import time

link_file_name = 'skytrax_reviews_links.txt'
data_file_name = 'skytrax_reviews_datatest.csv'
reviews_per_page = 50  # Big number of reviews shown per page makes scraping inefficient

# Makes selenium open Chrome in background
chrome_options = Options()
chrome_options.add_argument("--headless")


def get_links(file_path=link_file_name):
    """
    Gets all links to reviews per airline and per type for skytrax, sorts them and saves result to text file
    :return: None
    """
    link_list = list()
    browser = webdriver.Chrome(chrome_options=chrome_options)
    for review_link in ['https://www.airlinequality.com/review-pages/a-z-{review_type}-reviews/'
                        .format(review_type=review_type) for review_type in ('airline', 'seat')]:
        browser.get(review_link)
        airline_lists = [airline_list.find_elements_by_tag_name('li')
                         for airline_list in browser.find_elements_by_class_name('items')]
        for airline_list in airline_lists:
            for airline in airline_list:
                link = airline.find_element_by_tag_name('a').get_attribute('href')
                link_list.append(link)
    browser.close()

    # Save sorted unique links in text file
    link_list = sorted(list(set(link_list)))
    link_file = open(file_path, 'wt')
    link_file.write('\n'.join(link_list))


def scrap_review(review, airline_name, review_type):
    """
    scraps a single review block
    :param review: WebDriver object containing the review
    :param airline_name: name of the airline
    :param review_type: type of review
    :return: Dictionary containing scraping result
    """
    data_line = {'airline': airline_name, 'review_type': review_type,
                 'header': review.find_element_by_class_name('text_header').text,
                 'comment_date': review.find_element_by_tag_name('time').get_attribute('datetime'),
                 'comment': review.find_element_by_class_name('text_content').text}
    try:
        data_line['rating'], data_line['best_rating'] = review.find_element_by_class_name('rating-10') \
            .text.split('/')
    except ValueError:
        pass

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
    return data_line


def scrap_page(browser, airline_name, review_type):
    """
    scraps a single page of reviews
    :param browser: Web browser controlled by Selenium
    :param airline_name: name of the airline
    :param review_type: type of review
    :param data_lines: list of dictionaries containing parsed reviews
    :return: data_lines
    """
    reviews = browser.find_element_by_tag_name('article').find_elements_by_tag_name('article')
    data_lines = []
    for review in reviews:
        data_lines.append(scrap_review(review, airline_name, review_type))
    return data_lines


def scrap_airline(link, browser, scraped_data):
    """
    scrap all the comments for a single airline and review type
    :param link: Link to first page of skytrax reviews
    :param browser: Browser controlled by Selenium
    :param scraped_data: Already scrapd data for preventing redundant scraping
    :return:
    """
    tic = time.time()
    browser.get(link + '/?sortby=post_date%3ADesc&pagesize={}'.format(reviews_per_page))

    airline_name = browser.find_element_by_class_name('info').find_element_by_tag_name('h1').text
    review_type = browser.find_element_by_class_name('info').find_element_by_tag_name('h2').text
    review_count = int(browser.find_element_by_class_name('review-count').text.split(' ')[-2])

    if review_type in scraped_data[scraped_data.airline == airline_name].review_type.unique():
        print('Already scraped {review_type} for {airline_name}'
              .format(review_type=review_type, airline_name=airline_name))
        return

    print('Scraping {airline} {review}'.format(airline=airline_name, review=review_type))

    data_lines = []
    number_of_pages = review_count // reviews_per_page + 1 if review_count % reviews_per_page != 0 \
        else review_count // reviews_per_page

    for page_number in range(1, number_of_pages + 1):
        print('scraping page {} out of {}'.format(page_number, number_of_pages))
        data_lines.extend(scrap_page(browser, airline_name, review_type))
        browser.get(link + '/page/{}/?sortby=post_date%3ADesc&pagesize={}'.format(page_number, reviews_per_page))

    toc = time.time()
    print('scraped {n_records} in {seconds} seconds'.format(n_records=len(data_lines), seconds=floor(toc - tic)))

    scraped_data = scraped_data.append(pd.DataFrame(data_lines))
    scraped_data.to_csv(data_file_name, index=None)
    time.sleep(1)


def scrap_skytrax(links_to_scrap):
    """
    scraps links provided from skytrax and saves results to a dataframe. Does not parse airline if already
    present in the data frame.
    :param links_to_scrap: list of links to pages of airline reviews
    :return: None
    """
    try:
        data = pd.read_csv(data_file_name, index_col=None)
    except FileNotFoundError:
        data = pd.DataFrame(columns=['airline', 'review_type'])
        data.to_csv(data_file_name)

    browser = webdriver.Chrome(chrome_options=chrome_options)
    for link in links_to_scrap:
        scrap_airline(link, browser, data)


if __name__ == '__main__':
    # TODO uncomment before delivering code to Eleven
    # try:
    #     links = open(link_file_name).read().split('\n')
    # except FileNotFoundError:
    #     print('Getting page links from skytrax')
    #     get_links()
    #     links = open(link_file_name).read().split('\n')

    links = open(link_file_name).read().split('\n')
    scrap_skytrax(links[3:4])
