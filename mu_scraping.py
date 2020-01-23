import time
from selenium import webdriver

browser = webdriver.Chrome(
    '/Users/michaelullah/Documents/DSBA/CS/S2/SuperCase-elevenstrategy/NLP_Interior_Airline_Services/chromedriver')

browser.get('https://www.tripadvisor.com/Airlines')
airlines = browser.find_elements_by_class_name('airlineName')
print([airline.text for airline in airlines])

reviews_buttons = browser.find_elements_by_class_name('review_button ui_button secondary small').get_attribute('href')
print([a for a in reviews_buttons])


search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
driver.quit()
