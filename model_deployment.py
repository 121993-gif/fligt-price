import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

select_page = st.sidebar.radio('Select page', ['Intoduction','Analysis', 'Model Regression'])

if select_page == 'Intoduction':
    
    def main():
        st.title('Flight Price Prediction ')
        
        #st.image('')
        
        st.write('### introduction to my data :')
        
        st.write('''In the rapidly evolving landscape of air travel, understanding and predicting flight prices has become crucial for both consumers and industry stakeholders. This project explores the factors influencing flight prices, utilizing data-driven techniques to forecast fare trends. By analyzing various parameters such as historical price data, seasonality, and economic indicators, we aim to develop an accurate predictive model that can assist travelers in making informed decisions. Leveraging machine learning algorithms, this initiative not only seeks to provide insights into pricing dynamics but also contributes to enhancing the overall travel experience by empowering users with knowledge of potential fare fluctuations.

                    discovery, every insightful trend, lies a team of dedicated professionals armed with the tools of analysis and the power of visualization. Welcome to our exploration of the dynamic landscape of data science careers and salaries, where numbers tell stories and trends reveal themselves in vibrant hues.

                ''')
        
        st.header('Dataset Feature Overview')
        
        st.write('''
                *Airline*: The name of the airline.
        
                *Date_of_Journey*: The date of the journey.
                
                *Source: The source from which the service begins. 
                
                *Destination*: The destination where the service ends.
                
                *Route*: The route taken by the flight to reach the destination.

                *Dep_Time*: The time when the journey starts from the source.

                *Arrival_Time*: Time of arrival at the destination.

                *Duration*: Total duration of the flight.

                *Total_Stops*: Total stops between the source and destination.

                *Additional_Info*: Additional information about the flight

                *Price*: The price of the ticket

                ''')
        
    if __name__ == '__main__': 
        main() 


if select_page == 'Analysis':
    def main():
        cleaned_df = pd.read_csv('cleaned_df.csv')
        st.write('### Head of Dataframe')
        st.dataframe(cleaned_df.head(10))
        
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
        
        tab1.write('### What is the frequency distribution of flights for each airline?')
        airline_counts = cleaned_df['Airline'].value_counts().reset_index()
        airline_counts.columns = ['Airline', 'Number of Flights']
        tab1.plotly_chart(px.bar(airline_counts, 
                         x='Airline', 
                         y='Number of Flights', 
                         title='Frequency Distribution of Flights for Each Airline',
                         color='Number of Flights',
                         text='Number of Flights'))
        tab1.write('### Which airline has the highest number of flights?')
        airline_counts = cleaned_df['Airline'].value_counts().reset_index()
        airline_counts.columns = ['Airline', 'Number of Flights']
        highest_airline = airline_counts.loc[airline_counts['Number of Flights'].idxmax()]
        tab1.plotly_chart(px.bar(airline_counts,
                         x='Airline',
                         y='Number of Flights',
                         title='Frequency Distribution of Flights for Each Airline',
                         color='Number of Flights',
                         text='Number of Flights'))
        tab1.write('### How many flights are departing from each source city?')
        source_counts = cleaned_df['Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Number of Flights']
        tab1.plotly_chart( px.bar(source_counts,
                         x='Source',
                         y='Number of Flights',
                         title='Number of Flights Departing from Each Source City',
                         color='Number of Flights',
                         text='Number of Flights'))
        tab1.write('### What is the distribution of flights based on the number of stops (e.g., non-stop, 1 stop, etc.)?')
        stops_distribution = cleaned_df['Total_Stops'].value_counts().reset_index()
        stops_distribution.columns = ['Total Stops', 'Number of Flights']
        tab1.plotly_chart(px.bar(stops_distribution,
                         x='Total Stops',
                         y='Number of Flights',
                         title='Distribution of Flights Based on Number of Stops',
                         color='Number of Flights',
                         text='Number of Flights'))
        tab1.write('### How does the total number of stops affect the frequency of flights?')
        tab1.plotly_chart(px.bar(stops_distribution,
                         x='Total Stops',
                         y='Number of Flights',
                         title='Effect of Total Number of Stops on Frequency of Flights',
                         color='Number of Flights',
                         text='Number of Flights'))
        tab1.write('###  What are the most common additional info categories provided?')
        additional_info_counts = cleaned_df['Additional_Info'].value_counts().reset_index()
        additional_info_counts.columns = ['Additional Info', 'Count']
        tab1.plotly_chart(px.bar(additional_info_counts,
                         x='Additional Info',
                         y='Count',
                         title='Most Common Additional Info Categories',
                         color='Count',
                         text='Count'))
        tab1.write('###  What is the distribution of flight prices in the dataset?')
        tab1.plotly_chart(px.histogram(cleaned_df, 
                         x='Price', 
                         title='Distribution of Flight Prices',
                         nbins=30, 
                         labels={'Price': 'Flight Price'},
                         color_discrete_sequence=['blue']))
        tab1.write('###  How many flights are scheduled on each date?')
        flights_per_date = cleaned_df['Date'].value_counts().reset_index()
        flights_per_date.columns = ['Date', 'Number of Flights']
        flights_per_date = flights_per_date.sort_values(by='Date')
        tab1.plotly_chart(px.bar(flights_per_date, 
                         x='Date', 
                         y='Number of Flights', 
                         color='Number of Flights',
                         color_continuous_scale='viridis',
                         title='Number of Flights per Date'))
        

         
        tab1.write('### is there a trend in the number of flights over the observed dates?' )
        tab1.plotly_chart(px.line(flights_per_date, 
                         x='Date', 
                         y='Number of Flights', 
                         markers=True,
                         title='Number of Flights per Date'))
        tab1.write('### What is the distribution of flights across different months?' )
        monthly_flights = cleaned_df['Month'].value_counts().reset_index()
        monthly_flights.columns = ['Month', 'Number of Flights']
        tab1.plotly_chart(px.bar(monthly_flights,
                         x='Month',
                         y='Number of Flights',
                         title='Distribution of Flights Across Different Months',
                         color='Number of Flights',
                         text='Number of Flights'))
        
        tab1.write('### Are there certain hours or minutes that are more common for arrivals?' )
        arrival_hour_counts = cleaned_df['Arrival_hours'].value_counts().reset_index()
        arrival_hour_counts.columns = ['Arrival Hour', 'Number of Flights']
        arrival_minute_counts = cleaned_df['Arrival_minutes'].value_counts().reset_index()
        arrival_minute_counts.columns = ['Arrival Minute', 'Number of Flights']
        tab1.plotly_chart(px.bar(arrival_hour_counts,
                         x='Arrival Hour',
                         y='Number of Flights',
                         title='Frequency of Arrival Hours',
                         labels={'Number of Flights': 'Number of Flights'},
                         color='Number of Flights'))
        
        tab1.plotly_chart( px.bar(arrival_minute_counts,
                         x='Arrival Minute',
                         y='Number of Flights',
                         title='Frequency of Arrival Minutes',
                         labels={'Number of Flights': 'Number of Flights'},
                         color='Number of Flights'))
        tab1.write('### What is the distribution of departure times throughout the day?' )
        departure_hour_counts = cleaned_df['Dept_hour'].value_counts().reset_index()
        departure_hour_counts.columns = ['Departure Hour', 'Number of Flights']

        departure_minute_counts = cleaned_df['Dept_min'].value_counts().reset_index()
        departure_minute_counts.columns = ['Departure Minute', 'Number of Flights']

        tab1.plotly_chart(px.bar(departure_hour_counts,
                         x='Departure Hour',
                         y='Number of Flights',
                         title='Distribution of Departure Hours',
                         labels={'Number of Flights': 'Number of Flights'},
                         color='Number of Flights'))

        tab1.plotly_chart( px.bar(departure_minute_counts,
                         x='Departure Minute',
                         y='Number of Flights',
                         title='Distribution of Departure Minutes',
                         labels={'Number of Flights': 'Number of Flights'},
                         color='Number of Flights'))
        tab1.write('### What is the distribution of flight durations?' )
        tab1.plotly_chart(px.histogram(cleaned_df, 
                         x='Total_Duration_time_in_minutes', 
                         nbins=30, 
                         title='Distribution of Flight Durations',
                         labels={'Total_Duration_time_in_minutes': 'Flight Duration (minutes)'}, 
                         color_discrete_sequence=['skyblue']))
                           
                           
        
        
        
  
                
        
        
        
        

        
        

    
        tab2.write('### How does the airline affect flight prices?')
        average_price = cleaned_df.groupby('Airline')['Price'].mean().reset_index()
        average_price = average_price.sort_values(by='Price', ascending=False)

        tab2.plotly_chart(px.bar(average_price, 
                 x='Airline', 
                 y='Price', 
                 title='Average Flight Prices by Airline', 
                 labels={'Airline': 'Airline', 'Price': 'Average Price'},
                 color='Price'))


        tab2.write('### What is the relationship between the number of stops and flight prices??')
        tab2.plotly_chart(px.box(cleaned_df, x='Total_Stops', y='Price'))

        tab2.write('### Is there a correlation between departure times and flight prices?')
        tab2.plotly_chart(px.scatter(cleaned_df, x='Departure_Time', y='Price',
                         title='Flight Prices vs. Departure Time',
                         labels={'Departure_Time': 'Departure Time (minutes from midnight)', 'Price': 'Price'},
                         trendline='ols'))
        tab2.write('### Do different source and destination pairs affect flight prices?')
        avg_price = cleaned_df.groupby(['Source', 'Destination'])['Price'].mean().reset_index()
        tab2.plotly_chart(px.density_heatmap(avg_price,x='Destination', y='Source', z='Price',
                         title='Average Flight Price by Source-Destination Pair',
                         labels={'Price': 'Average Price'},
                         color_continuous_scale='YlGnBu',
                         hover_data={'Price': True}))                    
        tab2.write('### How does the total duration of flights relate to their prices?')
        tab2.plotly_chart(px.scatter(cleaned_df, x='Total_Duration_time_in_minutes', y='Price',
                         title='Flight Prices vs. Total Duration',
                         labels={'Total_Duration_time_in_minutes': 'Total Duration (minutes)', 'Price': 'Price'},
                         trendline='ols'))
        correlation = cleaned_df[['Total_Duration_time_in_minutes', 'Price']].corr().iloc[0, 1]
        tab2.write('Correlation between total duration and price: {}'.format(correlation))
                          
        
        tab2.write('Is there a seasonal effect on flight prices based on the month? ?')
        avg_price_by_month = cleaned_df.groupby('Month')['Price'].mean().reset_index()
        tab2.plotly_chart( px.line(avg_price_by_month, x='Month', y='Price',
                   title='Average Flight Price by Month',
                   labels={'Month': 'Month', 'Price': 'Average Price'},
                   markers=True))
        
        tab2.write('### How does the additional information category impact flight prices?')
        tab2.plotly_chart(px.box(cleaned_df,x='Additional_Info', y='Price',
                 title='Impact of Additional Info on Flight Prices',
                 labels={'Additional_Info': 'Additional Information', 'Price': 'Price'}))
        

        tab2.write('### What is the relationship between arrival times and flight prices?')
        tab2.plotly_chart(px.scatter(cleaned_df, 
                         x='Arrival_hours', 
                         y='Price',
                         color='Arrival_minutes',  
                         title='Flight Prices vs. Arrival Times',
                         labels={'Arrival_hours': 'Arrival Hour', 'Price': 'Price'},
                         color_continuous_scale=px.colors.sequential.Viridis)) 
        correlation_hours = cleaned_df[['Arrival_hours', 'Price']].corr().iloc[0, 1]
        tab2.write('Correlation between Arrival Hours and Price: {}'.format(correlation_hours))

        tab2.write('### Are there trends in flight prices over different dates?')
        avg_price_by_date = cleaned_df.groupby('Date')['Price'].mean().reset_index()
        tab2.plotly_chart(px.line(avg_price_by_date, 
                           x='Date', 
                           y='Price', 
                           title='Trend of Average Flight Prices Over Dates',
                           labels={'Date': 'Date', 'Price': 'Average Price'},
                           markers=True))
    

        tab2.write('##### Does the source city have a significant impact on the total flight duration?')
        tab2.plotly_chart(px.bar(cleaned_df, 
                          x='Source', 
                          y='Total_Duration_time_in_minutes',
                          title='Total Flight Duration by Source City',
                          labels={'Source': 'Source City', 
                                  'Total_Duration_time_in_minutes': 'Total Duration (minutes)'}))
        
        tab3.write('##### How do multiple factors (e.g., Airline, Source, Destination) together influence flight prices')
        tab3.plotly_chart(px.box(cleaned_df, x='Airline', y='Price', color='Source',
                          title='Flight Prices by Airline and Source',
                          labels={'Airline': 'Airline', 'Price': 'Price', 'Source': 'Source'}))

        tab3.write('##### What is the impact of Total Stops and Total Duration on flight prices when considering the Airline?')
        tab3.plotly_chart(px.scatter(cleaned_df,x='Total_Stops', y='Price', color='Airline',
                          title='Flight Prices vs. Total Stops',
                          labels={'Total_Stops': 'Total Stops', 'Price': 'Price'}))
        tab3.plotly_chart(px.scatter(cleaned_df,x='Total_Duration_time_in_minutes', y='Price', color='Airline',
                          title='Flight Prices vs. Total Duration',
                          labels={'Total_Duration_time_in_minutes': 'Total Duration (minutes)', 'Price': 'Price'}))

        tab3.write('##### Do different combinations of Source, Destination, and Total Stops affect flight prices?')
        tab3.plotly_chart(px.box(cleaned_df, x='Source', y='Price', color='Total_Stops',
                          title='Flight Prices by Source and Total Stops',
                          labels={'Source': 'Source', 'Price': 'Price', 'Total_Stops': 'Total Stops'}))       
        tab3.plotly_chart(px.box(cleaned_df, x='Destination', y='Price', color='Total_Stops',
                          title='Flight Prices by Destination and Total Stops',
                          labels={'Destination': 'Destination', 'Price': 'Price', 'Total_Stops': 'Total Stops'}))  
        tab3.write('##### Is there a seasonal effect (Month) on flight prices across different Airlines?')
        tab3.plotly_chart(px.box(cleaned_df,x='Month', y='Price', color='Airline',
                          title='Flight Prices by Month and Airline',
                          labels={'Month': 'Month', 'Price': 'Price', 'Airline': 'Airline'}))
        avg_price_by_month = cleaned_df.groupby(['Month', 'Airline'])['Price'].mean().reset_index()
        tab3.plotly_chart(px.line(avg_price_by_month, x='Month', y='Price', color='Airline',
                          title='Average Flight Prices by Month for Each Airline',
                          labels={'Month': 'Month', 'Price': 'Average Price', 'Airline': 'Airline'},
                          markers=True))
        
        tab3.write('##### How do arrival times (Arrival_hours and Arrival_minutes) correlate with flight prices while considering the Total Stops?')
        tab3.plotly_chart(px.scatter(cleaned_df, x='Arrival_hours', y='Price', color='Total_Stops',
                          size='Total_Stops', size_max=15,
                          title='Flight Prices vs. Arrival Hours (Colored by Total Stops)',
                          labels={'Arrival_hours': 'Arrival Hours', 'Price': 'Price', 'Total_Stops': 'Total Stops'}))
        tab3.plotly_chart(px.scatter(cleaned_df, x='Arrival_minutes', y='Price', color='Total_Stops',
                           size='Total_Stops', size_max=15,
                           title='Flight Prices vs. Arrival Minutes (Colored by Total Stops)',
                           labels={'Arrival_minutes': 'Arrival Minutes', 'Price': 'Price', 'Total_Stops': 'Total Stops'}))
                          
        
    if __name__=='__main__':
        main() 

elif select_page == 'Model Regression':
    
    def main(): 
        
        st.title('Model Regression')
        
        pipeline = joblib.load('ct_pipeline.pkl')

        def Prediction(Airline, Source, Destination, Total_Stops, Additional_Info, Date, Month, Arrival_hours, Arrival_minutes, Dept_hour, Dept_min, Total_Duration_time_in_minutes ):
            df = pd.DataFrame(columns=['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info', 'Date', 'Month', ' Arrival_hours', 'Arrival_minutes','Dept_hour','Dept_min','Total_Duration_time_in_minutes'])
            df.at[0, 'Airline'] = Airline
            df.at[0, 'Source'] = Source
            df.at[0, 'Destination'] = Destination
            df.at[0, 'Total_Stops'] = Total_Stops
            df.at[0, 'Additional_Info'] = Additional_Info
            df.at[0, 'Date'] = Date
            df.at[0, 'Month'] = Month
            df.at[0, 'Arrival_minutes'] =  Arrival_minutes
            df.at[0, 'Arrival_hours'] =  Arrival_hours
            df.at[0, 'Dept_hour'] = Dept_hour
            df.at[0, 'Dept_min'] = Dept_min
            df.at[0, 'Total_Duration_time_in_minutes'] = Total_Duration_time_in_minutes
            
            result = pipeline.predict(df)[0]
            return result

        # Now we will decide how the user can select each feature
        Airline = st.selectbox('Please select your Source',['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet','Vistara', 'Air Asia', 'GoAir', 'Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy', 'Trujet'])
        Source = st.selectbox('Please select your Source',['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai'])
        Destination = st.selectbox('Please select your Destination',['Delhi', 'Kolkata', 'Banglore', 'New Delhi', 'Cochin','Hyderabad'])
        Total_Stops = st.selectbox('Select Total Stops:', options=[0, 1, 2, 3])
        additional_info_options = ["No info", "In-flight meal not included", "No check-in baggage included","1 Long layover","Change airports","Business class","No Info","1 Short layover","Red-eye flight","2 Long layover"]
        Additional_Info = st.selectbox('Select Additional Info:', options=additional_info_options)
        days = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        Date = selected_day = st.selectbox('Select a Day:', options=days)
        Month = st.selectbox('Please select your work setting',[3, 4, 5, 6] )
        Arrival_hours = st.selectbox('Select Arrival Hour:', options=list(range(24)))
        arrival_minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        Arrival_minutes = st.selectbox('Select Arrival Minute:', options=arrival_minutes)
        dept_hours = list(range(24))
        Dept_hour = st.selectbox('Select Departure Hour:', options=dept_hours)
        Dept_min = st.selectbox("Select Departure Minute:", options=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        st.write(f'You selected Departure Minute: {Dept_min} minute(s)')
        Total_Duration_time_in_minutes = st.number_input("Enter Total Duration Time (in minutes):", min_value=0, max_value=3000, step=1)
        st.write(f'You selected Departure Minute: {Dept_min} minute(s)')
        if st.button('Predict'):
            result = Prediction( Airline, Source, Destination, Total_Stops, Additional_Info, Date, Month, Arrival_hours, Arrival_minutes, Dept_hour, Dept_min, Total_Duration_time_in_minutes )

            st.write('### Prediction Result:')
            st.write(f'The predicted result is: {round(np.exp(result), 2)}')

    if __name__=='__main__':
        main() 
