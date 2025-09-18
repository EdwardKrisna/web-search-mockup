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
        
        # Get the response text
        news_text = response.output[1].content[0].text
        
        # Parse the response - handle different formats
        news_points = []
        lines = news_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Handle bullet points with links
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                # Remove bullet point
                content = line.lstrip('-•* ').strip()
                
                # Try to extract headline and URL
                if '([' in content and '])' in content:
                    # Format: headline ([website](url))
                    headline_part = content.split('([')[0].strip()
                    url_part = content.split('([')[1].split('])')[0]
                    
                    # Clean headline (remove markdown formatting)
                    headline = headline_part.replace('**', '').strip()
                    
                    # Extract URL
                    if '](http' in url_part:
                        url = url_part.split('](')[1]
                    elif url_part.startswith('http'):
                        url = url_part
                    else:
                        url = f"https://{url_part}"
                        
                elif '(http' in content and ')' in content:
                    # Format: headline (url)
                    headline = content.split('(http')[0].strip().replace('**', '')
                    url_start = content.find('(http') + 1
                    url_end = content.find(')', url_start)
                    url = content[url_start:url_end] if url_end > url_start else ""
                    
                else:
                    # No URL found, just use the content as headline
                    headline = content.replace('**', '').strip()
                    url = ""
                
                if headline:  # Only add if we have a headline
                    news_points.append({
                        'headline': headline,
                        'url': url
                    })
        
        # If no points found, return the raw text as a single point
        if not news_points:
            news_points = [{
                'headline': news_text.strip().replace('**', '')[:200] + '...' if len(news_text) > 200 else news_text.strip().replace('**', ''),
                'url': ""
            }]
            
        return news_points
        
    except Exception as e:
        st.error(f"Error searching news: {str(e)}")
        # Return error as a properly formatted point
        return [{"headline": f"Error searching news: {str(e)}", "url": ""}]

def get_sentiment_score_with_ai(news_points):
    """Use OpenAI to analyze sentiment and return score 1-5"""
    try:
        headlines_text = "\n".join([point['headline'] for point in news_points])
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a sentiment analyst. Analyze the given news headlines and rate the overall sentiment on a scale of 1-5 where: 1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive. Return ONLY the number (1, 2, 3, 4, or 5)."
                },
                {
                    "role": "user", 
                    "content": f"Rate the sentiment of these news headlines:\n\n{headlines_text}"
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
            
            # Generate search query using user selections
            search_query = f"Berita kasus {x1} {x2} di Indonesia, hanya listnya saja jangan pakai kata kata lain!"
            
            # Get news results using OpenAI
            news_points = search_news_with_ai(search_query)
            
            # Get sentiment score using OpenAI (in same run)
            sentiment_score = get_sentiment_score_with_ai(news_points)
            
            # Store in session state
            st.session_state.news_points = news_points
            st.session_state.sentiment_score = sentiment_score
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
                # Ensure point is a dictionary with the expected keys
                if isinstance(point, dict) and 'headline' in point:
                    st.markdown(f"**{i}.** {point['headline']}")
                    if point.get('url'):
                        st.markdown(f"   🔗 [Read more]({point['url']})")
                else:
                    # Fallback for unexpected format
                    st.markdown(f"**{i}.** {str(point)}")
                st.markdown("")  # Add spacing
            
            st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.header("📈 Sentiment Analysis")
    
    # Display sentiment score if available (automatically after search)
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
        
        st.success(f"Analysis complete! Score: {score}/5")
    
    else:
        st.info("👆 Click search to get news and sentiment analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>News Sentiment Analyzer | Powered by AI</p>
</div>
""", unsafe_allow_html=True)