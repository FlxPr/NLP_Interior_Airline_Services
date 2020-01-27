import time
from selenium import webdriver
import pandas as pd
from datetime import datetime
import requests
from seleniumrequests import Chrome

browser = Chrome(
    '/Users/michaelullah/Documents/DSBA/CS/S2/SuperCase-elevenstrategy/NLP_Interior_Airline_Services/chromedriver')

# browser.get('https://www.tripadvisor.com/Airlines')
# airlines = browser.find_elements_by_class_name('airlineName')
# print([airline.text for airline in airlines])
#
# airline_links = []
# for i in range(62):
#     print(i)
#     airline_sections = browser.find_elements_by_class_name("review_button.ui_button.secondary.small")
#     airline_links = airline_links + [a.get_attribute('href') for a in airline_sections]
#     button_next = browser.find_element_by_class_name('nav.next.ui_button.primary')
#     button_next.click()
#     time.sleep(1)


# airlines_links = open('tripadvisor_airlines_links.txt', 'wt')
# airlines_links.write('\n'.join(set(airline_links)))


airlines_links = open('tripadvisor_airlines_links_shortlist.txt').read().split('\n')


for link in airlines_links:
    scraped = []
    browser.get(link)
    time.sleep(2)

    k = 0

    try:
        airline = browser.find_element_by_class_name(
            'flights-airline-review-page-airline-review-header-AirlineDetailHeader__airlineName--2JeT1').text
    except:
        airline = ''

    marker = 0
    while marker == 0 & k <= 10:
        k += 1

        reviews = browser.find_elements_by_class_name('location-review-card-Card__ui_card--2Mri0.'
                                                      'location-review-card-Card__card--o3LVm.'
                                                      'location-review-card-Card__section--NiAcw')

        for review in reviews[:5]:
            print(review)

            try:
                stars_container = review.find_element_by_class_name(
                    "location-review-review-list-parts-RatingLine__bubbles--GcJvM")
                star = stars_container.find_element_by_tag_name('span').get_attribute("class")
            except:
                star = ""

            try:
                title_container = review.find_element_by_class_name("location-review-review-list-parts-ReviewTitle"
                                                                    "__reviewTitle--2GO9Z")
                title = title_container.find_element_by_tag_name('span').text
            except:
                title = ""

            try:
                comment = review.find_element_by_class_name("location-review-review-list-parts-ExpandableReview"
                                                            "__reviewText--gOmRC").find_element_by_tag_name('span').text
            except:
                comment = ""

            try:
                date = review.find_element_by_class_name(
                    'social-member-event-MemberEventOnObjectBlock__event_type--3njyv').text
            except:
                date = ""

            try:
                contributions = review.find_element_by_class_name('social-member-MemberHeaderStats__bold--3z3qh').text
            except:
                contributions = ""

            print(comment)

            if any([star != "", title != "", comment != "", date != "", contributions != ""]):
                scraped.append(
                    {'airline': airline, 'star': star, 'title': title, 'comment': comment, 'date': date, 'contributions': contributions})

            time.sleep(1)

        try:
            button_next = browser.find_element_by_class_name('ui_button.nav.next.primary')
            button_next.click()
            time.sleep(1)
        except:
            marker = 1

    scraped = pd.DataFrame(scraped)
    scraped.to_csv('scraped' + str(airline) + '.csv')



    # browser.close()





