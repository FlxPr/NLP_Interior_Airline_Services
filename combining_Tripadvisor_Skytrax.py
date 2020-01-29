import pandas as pd
import numpy as np

skytrax_df = pd.read_csv('skytrax_reviews_data.csv')
tripadvisor_df = pd.read_csv('tripadvisor_all_clean.csv')

dict_columns_to_rename = {'star': 'rating',
                          'title': 'header',
                          'comment': 'comment',
                          'date': 'comment_date',
                          'route': 'Route',
                          'trip_class': 'Seat Type',
                          'Legroom': 'Seat Legroom',
                          'Seat comfort': 'Seat Comfort',
                          'In-flight Entertainment': 'Inflight Entertainment',
                          'Value for money': 'Value For Money',
                          'Food and Beverage': 'Food & Beverages'}

tripadvisor_df.rename(columns=dict_columns_to_rename, inplace=True)

tripadvisor_df['Seat Type'].replace(to_replace='Economy', value='Economy Class')
tripadvisor_df['best_rating'] = 5

tripadvisor_df['Website'] = 'Tripadvisor'
skytrax_df['Website'] = 'Skytrax'

aggregated_df = skytrax_df.append(tripadvisor_df, sort=True)

aggregated_df.to_csv('Skytrax_and_Tripadvisor.csv')
