'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)  # For reproducibility
impact_magnitude = np.random.uniform(4, 16, 50)  # Impact magnitude between 4 and 16 mg
angle_shift = np.random.uniform(70, 90, 50)  # Angle shift between 70 and 90 degrees

# Determine if an accident is detected
accident_detected = [(1 if (angle > 75 or impact > 5) else 0) for angle, impact in zip(angle_shift, impact_magnitude)]

# Calculate accuracy based on the detection
accuracy = []
for impact in impact_magnitude:
    if impact <= 5:
        accuracy.append(75)  # Minimum accuracy for small impact
    elif impact <= 6:
        accuracy.append(75)
    elif impact <= 7:
        accuracy.append(80)
    elif impact <= 8:
        accuracy.append(85)
    elif impact <= 9:
        accuracy.append(90)
    elif impact <= 10:
        accuracy.append(95)
    elif impact <= 11:
        accuracy.append(95)
    elif impact <= 12:
        accuracy.append(100)
    else:
        accuracy.append(100)  # High accuracy for larger impact

# Create a DataFrame
data = {
    "Impact Magnitude (mg)": impact_magnitude,
    "Angle Shift (degrees)": angle_shift,
    "Accident Detected": ["Yes" if detected else "No" for detected in accident_detected],
    "Accuracy (%)": accuracy
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Group data for plotting
accident_counts = df.groupby('Accident Detected').size()
accuracy_data = df.groupby('Impact Magnitude (mg)')['Accuracy (%)'].mean().reset_index()

# Plotting Accident Detection Counts
plt.figure(figsize=(12, 6))

# Line graph for Accident Detection
plt.subplot(1, 2, 1)  # Create a subplot for accident detection
plt.plot(accident_counts.index, accident_counts.values, marker='o', color='blue', label='Counts')
plt.title('Accident Detection Counts')
plt.xlabel('Accident Detected')
plt.ylabel('Counts')
plt.xticks(rotation=0)
plt.grid()
plt.legend()

# Line graph for Average Accuracy by Impact Magnitude
plt.subplot(1, 2, 2)  # Create a subplot for accuracy
plt.plot(accuracy_data['Impact Magnitude (mg)'], accuracy_data['Accuracy (%)'], marker='o', color='orange', label='Average Accuracy')
plt.title('Average Accuracy by Impact Magnitude')
plt.xlabel('Impact Magnitude (g)')
plt.ylabel('Average Accuracy (%)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

'''
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)  # For reproducibility
impact_magnitude = np.random.uniform(4, 16, 50)  # Impact magnitude between 4 and 16 mg
angle_shift = np.random.uniform(70, 90, 50)  # Angle shift between 70 and 90 degrees

# Determine if an accident is detected
accident_detected = [(1 if (angle > 75 or impact > 5) else 0) for angle, impact in zip(angle_shift, impact_magnitude)]

# Calculate accuracy based on the angle shift
accuracy = []
for angle in angle_shift:
    if angle <= 75:
        accuracy.append(75)  # Minimum accuracy for small angle shift
    elif angle <= 76:
        accuracy.append(75)
    elif angle <= 77:
        accuracy.append(80)
    elif angle <= 78:
        accuracy.append(85)
    elif angle <= 79:
        accuracy.append(90)
    elif angle <= 80:
        accuracy.append(95)
    elif angle <= 81:
        accuracy.append(95)
    else:
        accuracy.append(100)  # High accuracy for larger angle shifts

# Create a DataFrame
data = {
    "Impact Magnitude (mg)": impact_magnitude,
    "Angle Shift (degrees)": angle_shift,
    "Accident Detected": ["Yes" if detected else "No" for detected in accident_detected],
    "Accuracy (%)": accuracy
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Group data for plotting
accident_counts = df.groupby('Accident Detected').size()
accuracy_data = df.groupby('Angle Shift (degrees)')['Accuracy (%)'].mean().reset_index()

# Plotting Accident Detection Counts
plt.figure(figsize=(12, 6))

# Line graph for Accident Detection
plt.subplot(1, 2, 1)  # Create a subplot for accident detection
plt.plot(accident_counts.index, accident_counts.values, marker='o', color='blue', label='Counts')
plt.title('Accident Detection Counts')
plt.xlabel('Accident Detected')
plt.ylabel('Counts')
plt.xticks(rotation=0)
plt.grid()
plt.legend()

# Line graph for Average Accuracy by Angle Shift
plt.subplot(1, 2, 2)  # Create a subplot for accuracy
plt.plot(accuracy_data['Angle Shift (degrees)'], accuracy_data['Accuracy (%)'], marker='o', color='red', label='Average Accuracy')
plt.title('Average Accuracy by Angle Shift')
plt.xlabel('Angle Shift (degrees)')
plt.ylabel('Average Accuracy (%)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
'''
'''
import pandas as pd
import matplotlib.pyplot as plt

# Define the data for the table
data = {
    'Response Time Interval (seconds)': [
        '2.0 - 2.5', '2.0 - 2.5', 
        '2.5 - 3.0', '2.5 - 3.0', 
        '3.0 - 3.5', '3.0 - 3.5', 
        '3.5 - 4.0', '3.5 - 4.0', 
        '4.0 - 4.5', '4.0 - 4.5'
    ],
    'Impact Magnitude (3-15)': [
        15, 14, 
        13, 12, 
        11, 10, 
        9, 8, 
        6, 5
    ],
    'Count of Alerts': [
        5, 4, 
        3, 5, 
        6, 4, 
        7, 3, 
        2, 1
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Group by response time interval and calculate the average impact magnitude
avg_impact = df.groupby('Response Time Interval (seconds)').mean().reset_index()

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(avg_impact['Response Time Interval (seconds)'], avg_impact['Impact Magnitude (3-15)'], color='skyblue')

# Adding titles and labels
plt.title('Average Impact Magnitude by Alert Time Interval')
plt.xlabel('Alert Time Interval (seconds)')
plt.ylabel('Average Impact Magnitude (3-15)')
plt.xticks(rotation=45)  # Rotate x-ticks for better readability

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()  # Adjust layout to make room for rotated x-ticks
plt.show()
'''
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define the data for severity levels and response rates
data = {
    'Severity Level': ['High', 'Medium', 'Low'],
    'Response Rate (%)': [85, 75, 60]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(df['Severity Level'], df['Response Rate (%)'], color=['red', 'orange', 'green'])

# Adding titles and labels
plt.title('AI Model Response Rate by Severity Level')
plt.xlabel('Response Rate (%)')
plt.xlim(0, 100)  # Set x-axis limits from 0 to 100%

# Show grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Display the data in a table format
print("Response Rate Table:")
print(df)


'''
import matplotlib.pyplot as plt
import pandas as pd

# Define the updated data for overall system effectiveness
data = {
    'Metric': [
        'Total Accidents Detected',
        'Average Response Time (Seconds)',  # Updated metric
        'Total Incidents Resolved',
        'Average Severity Prediction Accuracy (%)',
        'Total User Queries to Chatbot'
    ],
    'Value': [50, 144, 45, 93, 380]  # 2.4 minutes converted to seconds
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.barh(df['Metric'], df['Value'], color='green')

# Adding titles and labels
plt.title('Overall System Effectiveness')
plt.xlabel('Value')
plt.xlim(0, max(df['Value']) + 50)  # Set x-axis limits

# Show grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Display the data in a table format
print("Overall System Effectiveness Table:")
print(df)