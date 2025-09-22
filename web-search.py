import streamlit as st
import time
from openai import OpenAI

# Initialize OpenAI client with Streamlit secrets
@st.cache_resource
def init_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except KeyError:
        st.error("❌ OpenAI API key not found! Please add OPENAI_API_KEY to your Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        st.stop()

client = init_openai_client()

# OpenAI API functions
def search_news_with_ai(query):
    """Use OpenAI web search to find real news with actual links"""
    try:
        response = client.responses.create(
            model="gpt-4.1-mini", 
            input=query,
            tools=[
                {
                    "type": "web_search"
                }
            ]
        )
        
        # Get the raw response text - keep it simple
        news_text = response.output[1].content[0].text
        return news_text
        
    except Exception as e:
        st.error(f"Error searching news: {str(e)}")
        return f"Error searching news: {str(e)}"

def get_sentiment_score_with_ai(news_text):
    """Use OpenAI to analyze sentiment and return score 1-5"""
    try:
        response = client.response.create(
            model="gpt-4.1-mini",
            instructions="You are a sentiment analyst. Analyze the given news content and rate the overall sentiment on a scale of 1-5 where: 1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive. Return ONLY the number (1, 2, 3, 4, or 5).",
            input=f"Rate the sentiment of this news content:\n\n{news_text}"
        )
        
        # Extract score
        score_text = response.choices[0].message.content.strip()
        score = int(''.join(filter(str.isdigit, score_text)))
        
        # Ensure score is between 1-5
        return max(1, min(5, score))
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return 3  # Return neutral if error

# Streamlit App Configuration
st.set_page_config(
    page_title="Web Search Mockup",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Web Search Mockup")

# Dropdown options
case_types = [
    'Korupsi', 
    'Skandal', 
    'Penipuan', 
    'Pelanggaran Regulasi',
    'Audit Bermasalah',
    'Konflik Kepentingan',
    'Kerugian Negara'
]

company_names = [
    'Perusahaan Unilever TBK',
    'PT Bank Central Asia TBK', 
    'PT Telkom Indonesia TBK',
    'PT Astra International TBK',
    'PT Bank Rakyat Indonesia TBK',
    'PT Indofood CBP Sukses Makmur TBK',
    'PT Gudang Garam TBK',
    'PT Bank Negara Indonesia TBK',
    'PT Adaro Energy TBK',
    'PT Vale Indonesia TBK',
    'PT Perkebunan Nusantara',
    'PTPN'
]

# Main selections
col1, col2 = st.columns(2)

with col1:
    x1 = st.selectbox("Select Case Type:", case_types, index=0)

with col2:
    x2 = st.selectbox("Select Company:", company_names, index=0)

search_button = st.button("🔍 Search News", type="primary", use_container_width=True)

# Results area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 News Results")
    
    if search_button:
        # Show loading spinner
        with st.spinner("Searching news and analyzing sentiment..."):
            time.sleep(1)  # Small delay for UX
            
            # Generate search query using x1 and x2 from user selections
            search_query = f"Berita {x1} {x2} di Indonesia, hanya listnya saja jangan pakai kata kata lain!"
            
            # Get news results using OpenAI web search
            news_text = search_news_with_ai(search_query)
            
            # Get sentiment score using OpenAI (in same run)
            sentiment_score = get_sentiment_score_with_ai(news_text)
            
            # Store in session state
            st.session_state.news_text = news_text
            st.session_state.sentiment_score = sentiment_score
            st.session_state.current_query = f"{x1} - {x2}"
    
    # Display news results if available
    if 'news_text' in st.session_state:
        # Just display the raw news text directly
        st.markdown(st.session_state.news_text)

with col2:
    st.header("📈 Sentiment Analysis")
    
    # Display sentiment score if available (automatically after search)
    if 'sentiment_score' in st.session_state:
        score = st.session_state.sentiment_score
        
        # Define color and label based on score
        if score == 1:
            color = "#ff4444"
            label = "Sangat Negatif"
        elif score == 2:
            color = "#ff8800"
            label = "Negatif" 
        elif score == 3:
            color = "#ffdd00"
            label = "Netral"
        elif score == 4:
            color = "#88ff00"
            label = "Positif"
        else:  # score == 5
            color = "#00ff44"
            label = "Sangat Positif"
        
        # Display score in a nice card
        st.markdown(f"""
        <div style="
            background-color: {color}20;
            padding: 20px;
            border-radius: 15px;
            border: 3px solid {color};
            text-align: center;
            margin: 20px 0;
        ">
            <h3 style="color: {color}; margin: 5px 0;">{label}</h3>
            <h1 style="color: {color}; margin: 5px 0;">{score}/5</h1>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Click search to get news and sentiment analysis")