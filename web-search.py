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
    """Use OpenAI to search/generate news points about the query"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a news researcher. Generate a list of realistic news headlines/points about the given query. Return only bullet points, no other text."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Parse response into list
        news_text = response.choices[0].message.content
        news_points = [line.strip().lstrip('•-*').strip() 
                      for line in news_text.split('\n') 
                      if line.strip() and not line.strip().startswith('#')]
        
        return [point for point in news_points if point]  # Remove empty strings
        
    except Exception as e:
        st.error(f"Error generating news: {str(e)}")
        return ["Error generating news. Please try again."]

def get_sentiment_score_with_ai(news_points):
    """Use OpenAI to analyze sentiment and return score 1-5"""
    try:
        news_text = "\n".join(news_points)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a sentiment analyst. Analyze the given news points and rate the overall sentiment on a scale of 1-5 where: 1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive. Return ONLY the number (1, 2, 3, 4, or 5)."
                },
                {
                    "role": "user", 
                    "content": f"Rate the sentiment of these news points:\n\n{news_text}"
                }
            ],
            max_tokens=10,
            temperature=0.1
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
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Sentiment Analyzer")
st.markdown("Analyze news sentiment for companies and case types in Indonesia")

# Dropdown options
case_types = [
    'Korupsi', 
    'Skandal', 
    'Penipuan', 
    'Pelanggaran Regulasi',
    'Audit Bermasalah',
    'Konflik Kepentingan'
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
    'PT Vale Indonesia TBK'
]

# Sidebar for selections
with st.sidebar:
    st.header("🔧 Configuration")
    
    x1 = st.selectbox(
        "Select Case Type:",
        case_types,
        index=0
    )
    
    x2 = st.selectbox(
        "Select Company:",
        company_names,
        index=0
    )
    
    search_button = st.button("🔍 Search News", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 News Results")
    
    if search_button:
        # Show loading spinner
        with st.spinner("Searching for news..."):
            time.sleep(2)  # Simulate API call delay
            
            # Generate search query
            search_query = f"Berita kasus {x1} {x2} di Indonesia, hanya listnya saja jangan pakai kata kata lain!"
            
            # Get news results using OpenAI
            news_points = search_news_with_ai(search_query)
            
            # Store in session state
            st.session_state.news_points = news_points
            st.session_state.current_query = f"{x1} - {x2}"
    
    # Display news results if available
    if 'news_points' in st.session_state:
        st.subheader(f"📊 Results for: {st.session_state.current_query}")
        
        # Display news points in a nice box
        with st.container():
            st.markdown("""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
                margin: 10px 0;
            ">
            """, unsafe_allow_html=True)
            
            for i, point in enumerate(st.session_state.news_points, 1):
                st.markdown(f"**{i}.** {point}")
            
            st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.header("📈 Sentiment Analysis")
    
    if 'news_points' in st.session_state:
        analyze_button = st.button("🧠 Analyze Sentiment", type="secondary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(1.5)  # Simulate API call delay
                
                # Get sentiment score using OpenAI
                score = get_sentiment_score_with_ai(st.session_state.news_points)
                st.session_state.sentiment_score = score
        
        # Display sentiment score if available
        if 'sentiment_score' in st.session_state:
            score = st.session_state.sentiment_score
            
            # Define color and label based on score
            if score == 1:
                color = "#ff4444"
                label = "Very Negative"
                emoji = "😡"
            elif score == 2:
                color = "#ff8800"
                label = "Negative" 
                emoji = "😟"
            elif score == 3:
                color = "#ffdd00"
                label = "Neutral"
                emoji = "😐"
            elif score == 4:
                color = "#88ff00"
                label = "Positive"
                emoji = "😊"
            else:  # score == 5
                color = "#00ff44"
                label = "Very Positive"
                emoji = "😍"
            
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
                <h2 style="color: {color}; margin: 0;">{emoji}</h2>
                <h3 style="color: {color}; margin: 5px 0;">{label}</h3>
                <h1 style="color: {color}; margin: 5px 0;">{score}/5</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(score / 5)
            
            st.success(f"Sentiment analysis complete! Score: {score}/5")
    
    else:
        st.info("👆 Please search for news first to analyze sentiment")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>News Sentiment Analyzer | Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Select Case Type**: Choose the type of case you want to search for
    2. **Select Company**: Choose the company you want to analyze
    3. **Search News**: Click the search button to find relevant news
    4. **Analyze Sentiment**: Click analyze to get sentiment score (1-5)
    
    **Score Meanings:**
    - 🔴 1-2: Negative sentiment (bad news)
    - 🟡 3: Neutral sentiment 
    - 🟢 4-5: Positive sentiment (good news)
    """)