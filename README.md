# YouTube Comment Sentiment Analysis and Summarization

This web application analyzes YouTube video comments, classifies them as positive, neutral, or negative, and provides a summarized report of the negative comments. This tool is designed to help content creators understand the sentiment of their viewers and quickly identify common complaints or concerns.

## Features

- **YouTube Comment Extraction**: Fetches comments from a YouTube video using the provided link.
- **Sentiment Analysis**: Classifies comments into positive, neutral, or negative categories with an accuracy of 89.9% using Logistic Regression.
- **Negative Comment Summarization**: Uses the Llama 3 LLM model to summarize negative comments, helping users quickly understand viewer complaints.

## Technologies Used

- **Logistic Regression**: For sentiment classification of YouTube comments.
- **Llama 3 LLM Model**: For summarizing negative comments.
- **Streamlit**: For building the web application interface.
- **Python**: For backend processing and model integration.
- **YouTubeCommentDownloader**: For fetching comments from YouTube videos.
- **VADER Sentiment Analysis**: For initial sentiment scoring.

## Usage
Open the web application.
Paste the YouTube video link in the input field.
Click "Analyze" to fetch and analyze the comments.
View the percentage breakdown of positive, neutral, and negative comments.
Click "Summarize" to generate a summary of the negative comments.


## Contact
For any questions or suggestions, please feel free to reach out:

- Email: ghosh.pamir.education@gmail.com
- LinkedIn: [linkedin.com/in/pamirghosh](https://www.linkedin.com/in/pamirghosh/)
