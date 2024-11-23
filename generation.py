import pandas as pd
import google.generativeai as genai
import os


os.environ["API_KEY"] = ''  
genai.configure(api_key=os.environ["API_KEY"])

def generate_gemini_summary_with_evolution(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Assuming the columns are named 'Review', 'Sentiment', and 'Date'
    reviews = df['Review'].tolist()
    sentiments = df['Sentiment'].tolist()
    dates = df['Date'].tolist()

    # Prepare the input text by combining dates, reviews, and sentiments
    combined_text = " ".join([f"Date: {date}, Review: {review}, Sentiment: {sentiment}" 
                              for date, review, sentiment in zip(dates, reviews, sentiments)])

    # Compose the prompt to ask for a summary of product evolution over time
    prompt = (
        "Please analyze the following user reviews over time and provide a detailed summary of how the product has evolved. Specifically, address the following points "
        "Trends in User Satisfaction: Highlight any noticeable changes in user sentiment and satisfaction levels over time. Describe how the overall reception of the product has improved or declined based on the reviews. Product Evolution: Provide insights into how the product has changed from its initial stages to the present. Explain any key improvements or issues that have been consistently mentioned by users. Product-Specific Suggestions for Improvement: Based on the reviews, provide detailed specific ideas for the manufacturer to enhance the product (Dont just make general suggestions). For example, if users mention performance issues related to graphics, suggest something like, 'Consider using an NVIDIA graphics card for better GPU performance.' Ensure that the suggestions are directly related to the product's features and are actionable for the manufacturer.:\n"
        f"{combined_text}"
    )

    # Send the prompt to Gemini for analysis
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    
    # Extract and print the response text
    final_summary = response.text
    print(final_summary)
    return final_summary

# Specify your Excel file path
file_path = 'fridgereviewssorted.xlsx'  # Replace with your actual file path
generate_gemini_summary_with_evolution(file_path)
