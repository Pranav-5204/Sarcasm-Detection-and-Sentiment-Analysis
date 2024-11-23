'''import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('sorted_reviews.xlsx')  # Replace with your Excel file path
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format

# Group by month and sarcasm label, then count the occurrences
df['Month'] = df['Date'].dt.to_period('M')  # Extract month and year
sarcasm_count_monthly = df.groupby(['Month', 'Sarcasm']).size().unstack(fill_value=0)

# Plotting the bar chart
sarcasm_count_monthly.plot(kind='bar', stacked=True, figsize=(14, 8), color=['r', 'g'])
plt.title('Monthly Sarcasm Distribution in Product Reviews')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.legend(['No Sarcasm', 'Sarcasm'])
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()'''


'''
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('sorted_reviews.xlsx')  # Replace with your Excel file path

# Count the occurrences of sarcasm and no sarcasm labels
sarcasm_counts = df['Sarcasm'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sarcasm_counts, labels=sarcasm_counts.index, autopct='%1.1f%%', colors=['r', 'g'], startangle=90)
plt.title('Overall Sarcasm Distribution in Product Reviews')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()
'''
'''import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('sorted_reviews.xlsx')  # Replace with your Excel file path

# Check the unique values in the 'Sarcasm' column
print(df['Sarcasm'].unique())

# Calculate the average sarcasm score for Sarcasm (1) and No Sarcasm (0)
sarcasm_avg_scores = df.groupby('Sarcasm')['Sarcasm Score'].mean()

# Plotting the pie chart (Sarcasm vs No Sarcasm with Average Scores)
plt.figure(figsize=(8, 8))
plt.pie(sarcasm_avg_scores, labels=sarcasm_avg_scores.index, autopct='%1.1f%%', colors=['r', 'g'], startangle=90)
plt.title(f'Average Sarcasm Score for Sarcastic vs Non-Sarcastic Reviews\n(Sarcasm Score - Sarcasm: {sarcasm_avg_scores[1]:.2f}, No Sarcasm: {sarcasm_avg_scores[0]:.2f})')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()'''

'''
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('sorted_reviews.xlsx')  # Replace with your Excel file path
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is in datetime format

# Group by month and sarcasm label, then calculate the mean sarcasm score
df['Month'] = df['Date'].dt.to_period('M')  # Extract month and year
sarcasm_avg_score_monthly = df.groupby(['Month', 'Sarcasm'])['Sarcasm Score'].mean().unstack(fill_value=0)

# Plotting the bar chart (Average Sarcasm Score)
sarcasm_avg_score_monthly.plot(kind='bar', stacked=False, figsize=(14, 8), color=['r', 'g'])
plt.title('Monthly Average Sarcasm Score in Product Reviews')
plt.xlabel('Month')
plt.ylabel('Average Sarcasm Score')
plt.xticks(rotation=45)
plt.legend(['No Sarcasm', 'Sarcasm'])
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
'''
'''
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset from an Excel file
df = pd.read_excel('sorted_reviews.xlsx')  # Replace with your Excel file path

# Check unique values in the 'Sarcasm' column
print(df['Sarcasm'].unique())

# Calculate the average sarcasm score for Sarcasm (1) and No Sarcasm (0)
sarcasm_avg_scores = df.groupby('Sarcasm')['Sarcasm Score'].mean()

# Plotting the pie chart to emphasize sarcasm detection
plt.figure(figsize=(8, 8))
plt.pie(sarcasm_avg_scores, labels=[f'Sarcastic\n(Avg. Score: {sarcasm_avg_scores[1]:.2f})', f'Non-Sarcastic\n(Avg. Score: {sarcasm_avg_scores[0]:.2f})'],
        autopct='%1.1f%%', colors=['r', 'g'], startangle=90)
plt.title(f'Importance of Sarcasm Detection in Reviews\nSarcasm Score Distribution')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()'''

