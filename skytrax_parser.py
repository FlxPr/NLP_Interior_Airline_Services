from selenium import webdriver


# Scrape AirlineEquality.com
browser = webdriver.Chrome()
browser.get('https://www.airlinequality.com/review-pages/a-z-airline-reviews/')
airline_lists = [airline_list.find_elements_by_tag_name('li')
                 for airline_list in browser.find_elements_by_class_name('items')]  # A being first letter of airline

links = list()
for airline_list in airline_lists:
    for airline in airline_list:
        link = airline.find_element_by_tag_name('a').get_attribute('href')
        links.append(link)

link_file = open('links.txt', 'wt')
link_file.write('\n'.join(links))
