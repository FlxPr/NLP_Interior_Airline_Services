import time
from selenium import webdriver

browser = webdriver.Chrome(
    '/Users/michaelullah/Documents/DSBA/CS/S2/SuperCase-elevenstrategy/NLP_Interior_Airline_Services/chromedriver')

browser.get('https://www.tripadvisor.com/Airlines')
airlines = browser.find_elements_by_class_name('airlineName')
print([airline.text for airline in airlines])

airline_section = browser.find_elements_by_class_name("review_button.ui_button.secondary.small")
airline_links = [a.get_attribute('href') for a in airline_section]


scraped = []

for link in airline_links:
    browser.get(link)
    reviews = browser.find_elements_by_class_name('location-review-card-Card__ui_card--2Mri0.'
                                                  'location-review-card-Card__card--o3LVm.'
                                                  'location-review-card-Card__section--NiAcw')

    for review in reviews:
        print(review)
        stars_container = review.find_element_by_class_name(
            "location-review-review-list-parts-RatingLine__bubbles--GcJvM")
        star = stars_container.find_element_by_tag_name('span').get_attribute("class")

        title_container = review.find_element_by_class_name("location-review-review-list-parts-ReviewTitle"
                                                            "__reviewTitle--2GO9Z")
        title = title_container.find_element_by_tag_name('span').text

        comment = review.find_element_by_class_name("location-review-review-list-parts-ExpandableReview"
                                                    "__reviewText--gOmRC").find_element_by_tag_name('span').text

        scraped.append({'star': star, 'title': title, 'comment': comment})


    break

    time.sleep(3)


# location-review-card-Card__ui_card--2Mri0.location-review-card-Card__card--o3LVm.location-review-card-Card__section--NiAcw
# location-review-card-Card__ui_card--2Mri0 location-review-card-Card__card--o3LVm location-review-card-Card__section--NiAcw

# location-review-review-list-parts-RatingLine__bubbles--GcJvM