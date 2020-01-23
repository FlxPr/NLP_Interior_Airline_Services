from selenium import webdriver

# Scrape AirlineEquality.com
browser = webdriver.Chrome()
browser.get('https://www.airlinequality.com/review-pages/a-z-airline-reviews/')
airline_lists = [airline_list.find_elements_by_tag_name('li')
                 for airline_list in browser.find_elements_by_class_name('items')]

links = list()
for airline_list in airline_lists:
    for airline in airline_list:
        link = airline.find_element_by_tag_name('a').get_attribute('href')
        links.append(link)

# Save links in text file
link_file = open('links.txt', 'wt')
link_file.write('\n'.join(links))

links = open('links.txt').read().split('\n')
link = links[0]


for link in links:
    browser = webdriver.Chrome()
    browser.get(link)

    name = browser.find_element_by_class_name('info').find_element_by_tag_name('h1').text
    reviews_container = browser.find_element_by_tag_name('article')
    reviews = reviews_container.find_elements_by_tag_name('article')
    for review in reviews:
        rating, best_rating = review.find_element_by_class_name('rating-10').text.split('/')
        header = review.find_element_by_class_name('text_header').text
        comment_date = review.find_element_by_tag_name('time').get_attribute('datetime')
        comment = review.find_element_by_class_name('text_content').text
        ratings = review.find_element_by_class_name('review-ratings')
        for rating in ratings:
            break  # Fill the ratings

    browser.close()

