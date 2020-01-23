import time
from selenium import webdriver

driver = webdriver.Chrome(
    '/Users/michaelullah/Documents/DSBA/CS/S2/SuperCase-elevenstrategy/NLP_Interior_Airline_Services/chromedriver')

driver.get('https://www.tripadvisor.com/Airlines')
time.sleep(5) # Let the user actually see something!
airlines = driver.find_elements_by_class_name('airlineName')

print([airline.text for airline in airlines])


search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
driver.quit()
