import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('fridgereviewssorted.xlsx')  # Replace with your Excel file path
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format

# Function to map sentiment labels to positions based on sentiment score
def label_to_position(sentiment_label, sentiment_score):
    if sentiment_label == 'Positive':
        return 1 + (sentiment_score - 0.5) * 0.5  # Slight variation above 1
    elif sentiment_label == 'Neutral':
        return 0 + (sentiment_score - 0.5) * 0.5  # Slight variation around 0
    elif sentiment_label == 'Negative':
        return -1 + (sentiment_score - 0.5) * 0.5  # Slight variation below -1

# Map the sentiment labels to their respective positions
df['Position'] = df.apply(lambda row: label_to_position(row['Sentiment'], row['Sentiment Score']), axis=1)

# Sort the DataFrame by Date to ensure the line plot is correct
df = df.sort_values(by='Date')

# Plotting
plt.figure(figsize=(14, 8))

# Create a line plot
plt.plot(df['Date'], df['Position'], marker='o', linestyle='-', color='b', alpha=0.6)

# Add horizontal lines for sentiment categories
plt.axhline(y=1, color='g', linestyle='--', label='Positive')
plt.axhline(y=0, color='b', linestyle='--', label='Neutral')
plt.axhline(y=-1, color='r', linestyle='--', label='Negative')

# Adding titles and labels
plt.title('Evolution of Product Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Position')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])  # Label y-axis ticks
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.show()


'''
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('fridgereviewssorted.xlsx')  # Replace with your Excel file path
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format

# Function to map sentiment labels to positions based on sentiment score
def label_to_position(sentiment_label, sentiment_score):
    if sentiment_label == 'Positive':
        return 1 + (sentiment_score - 0.5) * 0.5  # Slight variation above 1
    elif sentiment_label == 'Neutral':
        return 0 + (sentiment_score - 0.5) * 0.5  # Slight variation around 0
    elif sentiment_label == 'Negative':
        return -1 + (sentiment_score - 0.5) * 0.5  # Slight variation below -1

# Map the sentiment labels to their respective positions
df['Position'] = df.apply(lambda row: label_to_position(row['Sentiment'], row['Sentiment Score']), axis=1)

# Group by month and calculate the average sentiment position for each month
df['Month'] = df['Date'].dt.to_period('M')  # Extract month and year
monthly_avg = df.groupby('Month')['Position'].mean().reset_index()

# Convert 'Month' back to datetime format for plotting
monthly_avg['Month'] = monthly_avg['Month'].dt.to_timestamp()

# Plotting
plt.figure(figsize=(14, 8))

# Create a line plot of monthly average sentiment positions
plt.plot(monthly_avg['Month'], monthly_avg['Position'], marker='o', linestyle='-', color='b', alpha=0.6)

# Add horizontal lines for sentiment categories
plt.axhline(y=1, color='g', linestyle='--', label='Positive')
plt.axhline(y=0, color='b', linestyle='--', label='Neutral')
plt.axhline(y=-1, color='r', linestyle='--', label='Negative')

# Adding titles and labels
plt.title('Monthly Evolution of Product Reviews Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Position')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])  # Label y-axis ticks
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.show()
'''