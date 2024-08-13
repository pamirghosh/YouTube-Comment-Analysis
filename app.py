from youtube_comment_downloader import YoutubeCommentDownloader
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import spacy
import sentimentAnalysis
import joblib
import yt_dlp
import re

def findChannelName(channel_url):
    pattern = r'(@.+)'
    match = re.search(pattern, channel_url)
    return match.group(1)

def get_video_info(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        channel_url = info_dict.get('uploader_url', None)
        return findChannelName(channel_url)

def createSummery(comments):
    prompt='''
        You are a text summarizer.

        Task: Create summery text from a python list of negative comments.

        Description: You will be given a python list consist of negative comments form a youtube  video.
        You have to marge the comments and then have to summarize.

        Example:
            comments=['Improve your aduio quality','Mic is not good','Video quality is not good','Use a good mic']

            Summary:
                1. You have to improve audio and video quality
                2. Try to use good mic for better audio.

        ***Important***:
            Do not write any suggestions or solutions. Just write summary and nothing else.
        List of Comments:
        {comments}

        ***Output***:

    '''
    PROMPT = PromptTemplate(
            template=prompt,
            input_variables=["comments"]
        )
    chain = LLMChain(
        llm=llm,
        prompt=PROMPT,
        verbose=False
        )
    response = chain({"comments": comments})
    return response['text']
    
def analyze_comments(comments, author):
    _Total_=0
    _POS_=0
    _NUT_=0
    _NEG_=0
    negativeComment=[]
    for comment in comments:
        if author==comment['author']:
            continue
        user_input_with_emoji = sentimentAnalysis.convert_emojis_to_text(comment['text'])
        cleaned_input = sentimentAnalysis.preprocess(user_input_with_emoji, nlp)
        tfidf_input = tfidf.transform([cleaned_input])
        prediction = model.predict(tfidf_input)
        if prediction==2:
            _POS_+=1
        elif prediction==1:
            _NUT_+=1
        else:
            negativeComment.append(user_input_with_emoji)
            _NEG_+=1
        _Total_+=1
    return _Total_, _POS_, _NUT_, _NEG_, negativeComment

if 'negativeComment' not in st.session_state:
    st.session_state.negativeComment = []

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if 'summary_done' not in st.session_state:
    st.session_state.summary_done = False
if __name__ == "__main__":
    model, tfidf = joblib.load('YouTube_Sentiment_Analysis.pkl')
    nlp = spacy.load('en_core_web_sm')
    
    st.title("YouTube Comment Analysis")
    video_url = st.text_input("Paste YouTube Video URL:")
    
    if video_url:
        author = get_video_info(video_url)
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url)
    
    if not st.session_state.analysis_done and st.button("Analyze"):
        if video_url:
            with st.spinner("Analyzing comments..."):
                _Total_, _POS_, _NUT_, _NEG_, st.session_state.negativeComment = analyze_comments(comments, author)

            st.session_state.analysis_results = (_Total_, _POS_, _NUT_, _NEG_)
            st.session_state.analysis_done = True
            st.session_state.summary_done = False 
    if st.session_state.analysis_done:
        _Total_, _POS_, _NUT_, _NEG_ = st.session_state.analysis_results
        
        st.markdown(f"<h2 style='text-align: center;'>Analysis Results</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#2c3e50;'>Total comments (Excluding author): <strong>{_Total_}</strong></h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div style='text-align: center;'><h3 style='color:green;'>Positive Comments</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 24px; color:green;'>👍 {_POS_}<br>({_POS_ * 100 / _Total_:.2f}%)</div>", unsafe_allow_html=True)
            st.progress(_POS_ / _Total_)

        with col2:
            st.markdown(f"<div style='text-align: center;'><h3 style='color:gray;'>Neutral Comments</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 24px; color:gray;'>😐 {_NUT_}<br>({_NUT_ * 100 / _Total_:.2f}%)</div>", unsafe_allow_html=True)
            st.progress(_NUT_ / _Total_)

        with col3:
            st.markdown(f"<div style='text-align: center;'><h3 style='color:red;'>Negative Comments</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 24px; color:red;'>👎 {_NEG_}<br>({_NEG_ * 100 / _Total_:.2f}%)</div>", unsafe_allow_html=True)
            st.progress(_NEG_ / _Total_)
    if st.session_state.analysis_done and not st.session_state.summary_done: 
        if st.button("Summary"):
            st.session_state.summary_done = True
            llm = Ollama(model="llama3")
            summary = createSummery(st.session_state.negativeComment)
            cleaned_text = re.sub(r'^.*\n|Note:.*$', '', summary).strip()
            st.session_state.summary_text = cleaned_text
    if st.session_state.summary_done:
        with st.spinner("Analyzing comments..."):
            st.markdown(f"<h3>Summary of Negative Comments:</h3>", unsafe_allow_html=True)
            st.text(st.session_state.summary_text)