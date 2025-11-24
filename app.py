import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, Optional, Tuple
import time

# Try to import transformers and PEFT for LoRA models (optional)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Detect deployment environment
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING", "").lower() == "true"
IS_CLOUD_DEPLOYMENT = IS_STREAMLIT_CLOUD

# Page configuration
st.set_page_config(
    page_title="🌾 Climate Resilience Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s;
        color: #1f1f1f;
        font-size: 16px;
        line-height: 1.6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        color: #1f1f1f;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
        color: #1f1f1f;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'location_data' not in st.session_state:
    st.session_state.location_data = {}
if 'soil_params' not in st.session_state:
    st.session_state.soil_params = {'N': 50, 'P': 50, 'K': 50, 'pH': 6.5}
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'lora_model' not in st.session_state:
    st.session_state.lora_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# API Keys - Replace with your own
DEFAULT_WEATHERAPI_KEY = "ENTER_YOUR_WEATHER_API_KEY"
DEFAULT_GROQ_API_KEY = "ENTER_YOUR_GROQ_API_KEY"

# Initialize API keys in session state
if 'weather_api_key' not in st.session_state:
    st.session_state.weather_api_key = DEFAULT_WEATHERAPI_KEY
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = DEFAULT_GROQ_API_KEY

# Title
st.markdown('<h1 class="main-header">🌾 Climate Resilience Chatbot for Farmers</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Agricultural Advisor with Real-Time Weather, Soil Data & Climate Adaptation Strategies</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys Section
    st.subheader("🔑 API Keys")
    
    weather_api_key_input = st.text_input(
        "🌤️ Weather API Key",
        value=st.session_state.weather_api_key,
        type="password",
        help="Get your free key from weatherapi.com"
    )
    if weather_api_key_input:
        st.session_state.weather_api_key = weather_api_key_input
    else:
        st.session_state.weather_api_key = DEFAULT_WEATHERAPI_KEY
    
    groq_api_key_input = st.text_input(
        "🤖 Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        help="Get your free key from console.groq.com"
    )
    if groq_api_key_input:
        st.session_state.groq_api_key = groq_api_key_input
    else:
        st.session_state.groq_api_key = DEFAULT_GROQ_API_KEY
    
    # API Status
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.weather_api_key:
            st.success("✅ Weather API")
        else:
            st.error("❌ Weather API")
    with col2:
        if st.session_state.groq_api_key:
            st.success("✅ Groq API")
        else:
            st.error("❌ Groq API")
    
    # Test API Keys button
    if st.button("🧪 Test API Keys", use_container_width=True):
        with st.spinner("Testing API keys..."):
            # Test Weather API
            try:
                import requests
                test_url = f"http://api.weatherapi.com/v1/current.json?key={st.session_state.weather_api_key}&q=London&aqi=no"
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code == 200:
                    st.success("✅ Weather API working!")
                else:
                    st.error(f"❌ Weather API failed (Status: {test_response.status_code})")
            except Exception as e:
                st.error(f"❌ Weather API error: {str(e)[:100]}")
            
            # Test Groq API
            try:
                import requests
                test_headers = {
                    "Authorization": f"Bearer {st.session_state.groq_api_key}",
                    "Content-Type": "application/json"
                }
                test_data = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Say 'test'"}],
                    "max_tokens": 10
                }
                test_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=test_headers,
                    json=test_data,
                    timeout=10
                )
                if test_response.status_code == 200:
                    st.success("✅ Groq API working!")
                else:
                    st.error(f"❌ Groq API failed (Status: {test_response.status_code})")
            except Exception as e:
                st.error(f"❌ Groq API error: {str(e)[:100]}")
    
    # Quick links to get API keys
    with st.expander("🔗 Get API Keys"):
        st.markdown("""
        **Weather API:**
        - Visit: [weatherapi.com](https://www.weatherapi.com/)
        - Sign up for free
        - Get 1M calls/month free
        
        **Groq API:**
        - Visit: [console.groq.com](https://console.groq.com/)
        - Sign up with Google/GitHub
        - Create API key
        - Free tier available
        """)
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 AI Model")
    
    if IS_CLOUD_DEPLOYMENT:
        st.info("🌐 Cloud: Using Groq API")
        use_local_model = False
    else:
        use_local_model = st.checkbox(
            "Use Local LoRA Model (Advanced)",
            value=False,
            help="Use fine-tuned LoRA adapter (requires GPU/CPU resources)"
        )
    
    if use_local_model and TRANSFORMERS_AVAILABLE:
        model_choice = st.selectbox(
            "LoRA Model",
            ["climate_advisor_lora", "climate_advisor_finetuned"],
            help="Select the LoRA adapter to use"
        )
        st.session_state.model_choice = model_choice
    else:
        model_choice = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            help="Choose AI model for recommendations"
        )
        st.session_state.model_choice = model_choice
        st.session_state.groq_model = model_choice
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum response length"
    )
    
    st.markdown("---")
    
    # Project Info
    st.subheader("📊 Project Info")
    st.info("""
    **Model:** TinyLlama-1.1B + LoRA
    
    **Dataset:** 2,200 samples
    
    **Crops:** 22 types
    
    **Training Loss:** 0.333 (79% reduction)
    
    **APIs:**
    - FreeWeather API ✅
    - Groq API ✅
    - NASA POWER API ✅
    """)

# Helper Functions
@st.cache_data(ttl=3600)
def get_weather_data(location: str, api_key: str) -> Optional[Dict]:
    """Fetch real-time weather data from WeatherAPI"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=yes"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data['current']['temp_c'],
                "humidity": data['current']['humidity'],
                "condition": data['current']['condition']['text'],
                "rainfall": data['current'].get('precip_mm', 0),
                "wind_speed": data['current']['wind_kph'],
                "wind_direction": data['current']['wind_degree'],
                "uv_index": data['current']['uv'],
                "pressure": data['current']['pressure_mb'],
                "feels_like": data['current']['feelslike_c'],
                "visibility": data['current']['vis_km'],
                "location": f"{data['location']['name']}, {data['location']['region']}, {data['location']['country']}",
                "lat": float(data['location']['lat']),
                "lon": float(data['location']['lon']),
                "timezone": data['location']['tz_id'],
                "last_updated": data['current']['last_updated']
            }
        else:
            st.error(f"Weather API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching weather: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_weather_forecast(location: str, api_key: str, days: int = 7) -> Optional[Dict]:
    """Fetch weather forecast from WeatherAPI"""
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days={days}&aqi=no&alerts=no"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            forecast_days = []
            for day in data['forecast']['forecastday']:
                forecast_days.append({
                    'date': day['date'],
                    'max_temp': day['day']['maxtemp_c'],
                    'min_temp': day['day']['mintemp_c'],
                    'avg_temp': day['day']['avgtemp_c'],
                    'condition': day['day']['condition']['text'],
                    'rainfall': day['day']['totalprecip_mm'],
                    'humidity': day['day']['avghumidity'],
                    'wind_speed': day['day']['maxwind_kph'],
                    'uv_index': day['day']['uv']
                })
            return {
                'location': f"{data['location']['name']}, {data['location']['region']}",
                'forecast': forecast_days
            }
        return None
    except Exception as e:
        st.warning(f"⚠️ Weather forecast unavailable: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def get_nasa_power_data(lat: float, lon: float) -> Optional[Dict]:
    """Fetch soil and climate data from NASA POWER API"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            properties = data.get('properties', {}).get('parameter', {})
            
            if properties:
                temp_data = properties.get('T2M', {})
                precip_data = properties.get('PRECTOTCORR', {})
                humidity_data = properties.get('RH2M', {})
                wind_data = properties.get('WS2M', {})
                solar_data = properties.get('ALLSKY_SFC_SW_DWN', {})
                
                recent_dates = sorted(temp_data.keys())[-7:] if temp_data else []
                
                if recent_dates:
                    avg_temp = np.mean([temp_data.get(d, 0) for d in recent_dates])
                    avg_precip = np.mean([precip_data.get(d, 0) for d in recent_dates])
                    avg_humidity = np.mean([humidity_data.get(d, 0) for d in recent_dates])
                    avg_wind = np.mean([wind_data.get(d, 0) for d in recent_dates])
                    avg_solar = np.mean([solar_data.get(d, 0) for d in recent_dates])
                    
                    return {
                        "avg_temperature_30d": avg_temp,
                        "avg_precipitation_30d": avg_precip,
                        "avg_humidity_30d": avg_humidity,
                        "avg_wind_speed_30d": avg_wind,
                        "avg_solar_radiation_30d": avg_solar,
                        "data_points": len(recent_dates),
                        "date_range": f"{recent_dates[0]} to {recent_dates[-1]}"
                    }
        return None
    except Exception as e:
        st.warning(f"⚠️ NASA POWER API unavailable: {str(e)}")
        return None

def load_lora_model(model_path: str):
    """Load LoRA adapter model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        full_path = os.path.join("models", model_path)
        
        if not os.path.exists(full_path):
            return None, None
        
        # Try loading tokenizer from base model (more reliable)
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(full_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            base_model = base_model.to(device)
        
        # Clean adapter config if needed
        config_path = os.path.join(full_path, "adapter_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                unsupported_fields = ['corda_config', 'eva_config', 'megatron_config', 'megatron_core']
                cleaned_config = {k: v for k, v in adapter_config.items() 
                                if k not in unsupported_fields and v is not None}
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_config, f, indent=2)
            except:
                pass
        
        model = PeftModel.from_pretrained(base_model, full_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        return None, None

def generate_with_lora(prompt: str, model, tokenizer, max_tokens: int = 1000) -> str:
    """Generate response using LoRA model"""
    try:
        formatted_prompt = f"<|system|>\nYou are an expert agricultural advisor.\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        device = next(model.parameters()).device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        return None

def get_groq_recommendation(prompt: str, model: str, temp: float, max_tokens: int, api_key: str) -> Optional[str]:
    """Get recommendation from Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert agricultural advisor specialized in climate-resilient farming. Analyze soil conditions, weather data, and recommend the most suitable crop with climate adaptation strategies. Provide practical, actionable advice for farmers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            st.error(f"Groq API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error getting recommendation: {str(e)}")
        return None

def create_weather_visualization(weather_data: Dict):
    """Create comprehensive weather visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'UV Index'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Temperature gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['temperature'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "°C"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 15], 'color': "lightblue"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "orange"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Humidity gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['humidity'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"}
            }
        ),
        row=1, col=2
    )
    
    # Wind Speed gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['wind_speed'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "km/h"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkgreen"}
            }
        ),
        row=2, col=1
    )
    
    # UV Index gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=weather_data['uv_index'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "UV Index"},
            gauge={
                'axis': {'range': [None, 11]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 3], 'color': "green"},
                    {'range': [3, 6], 'color': "yellow"},
                    {'range': [6, 8], 'color': "orange"},
                    {'range': [8, 11], 'color': "red"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Real-Time Weather Metrics")
    return fig

def create_soil_visualization(soil_params: Dict):
    """Create soil nutrients visualization"""
    nutrients = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    values = [soil_params['N'], soil_params['P'], soil_params['K']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=nutrients,
            y=values,
            marker_color=['#4CAF50', '#2196F3', '#FF9800'],
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Soil Nutrient Levels (kg/hectare)",
        xaxis_title="Nutrients",
        yaxis_title="Amount (kg/hectare)",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_ph_visualization(ph_value: float):
    """Create pH level visualization"""
    fig = go.Figure()
    
    ph_scale = np.arange(3, 11, 0.1)
    colors = ['red' if x < 5.5 else 'orange' if x < 6.5 else 'green' if x < 7.5 else 'orange' if x < 8.5 else 'red' for x in ph_scale]
    
    fig.add_trace(go.Scatter(
        x=ph_scale,
        y=[1]*len(ph_scale),
        mode='markers',
        marker=dict(size=10, color=colors),
        name='pH Scale'
    ))
    
    fig.add_trace(go.Scatter(
        x=[ph_value],
        y=[1],
        mode='markers',
        marker=dict(size=30, color='darkblue', symbol='diamond'),
        name=f'Current pH: {ph_value}'
    ))
    
    fig.update_layout(
        title="Soil pH Level",
        xaxis_title="pH Value",
        yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
        height=200,
        template="plotly_white"
    )
    
    return fig

def calculate_soil_health_score(soil_params: Dict) -> Dict:
    """Calculate soil health score based on NPK and pH"""
    n, p, k, ph = soil_params['N'], soil_params['P'], soil_params['K'], soil_params['pH']
    
    # Optimal ranges
    n_optimal = (40, 80)
    p_optimal = (20, 50)
    k_optimal = (100, 200)
    ph_optimal = (6.0, 7.5)
    
    # Calculate scores (0-100)
    def score_value(value, optimal_range, weight=1.0):
        min_val, max_val = optimal_range
        if min_val <= value <= max_val:
            return 100 * weight
        elif value < min_val:
            return max(0, (value / min_val) * 100 * weight)
        else:
            return max(0, (max_val / value) * 100 * weight)
    
    n_score = score_value(n, n_optimal, 0.25)
    p_score = score_value(p, p_optimal, 0.25)
    k_score = score_value(k, k_optimal, 0.25)
    ph_score = score_value(ph, ph_optimal, 0.25)
    
    total_score = n_score + p_score + k_score + ph_score
    
    # Determine health level
    if total_score >= 80:
        level = "Excellent"
        color = "green"
    elif total_score >= 60:
        level = "Good"
        color = "blue"
    elif total_score >= 40:
        level = "Fair"
        color = "orange"
    else:
        level = "Poor"
        color = "red"
    
    return {
        'total_score': round(total_score, 1),
        'level': level,
        'color': color,
        'breakdown': {
            'N': round(n_score, 1),
            'P': round(p_score, 1),
            'K': round(k_score, 1),
            'pH': round(ph_score, 1)
        },
        'recommendations': get_soil_recommendations(soil_params, total_score)
    }

def get_soil_recommendations(soil_params: Dict, score: float) -> list:
    """Get recommendations based on soil health score"""
    recommendations = []
    n, p, k, ph = soil_params['N'], soil_params['P'], soil_params['K'], soil_params['pH']
    
    if n < 40:
        recommendations.append("Add nitrogen-rich fertilizers (urea, ammonium sulfate)")
    elif n > 80:
        recommendations.append("Nitrogen levels are high - reduce nitrogen inputs")
    
    if p < 20:
        recommendations.append("Add phosphorus fertilizers (superphosphate, bone meal)")
    elif p > 50:
        recommendations.append("Phosphorus levels are adequate - maintain current levels")
    
    if k < 100:
        recommendations.append("Add potassium fertilizers (potash, wood ash)")
    elif k > 200:
        recommendations.append("Potassium levels are high - reduce potassium inputs")
    
    if ph < 6.0:
        recommendations.append("Soil is acidic - add lime to raise pH")
    elif ph > 7.5:
        recommendations.append("Soil is alkaline - add sulfur or organic matter to lower pH")
    
    if score < 40:
        recommendations.append("Consider soil testing and professional consultation")
        recommendations.append("Add organic matter (compost, manure) to improve overall soil health")
    
    return recommendations

# Main Interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chatbot", 
    "🌍 Location Analysis", 
    "📊 Visualizations", 
    "🌡️ Weather Forecast",
    "🌾 Crop Recommendations",
    "📜 History & Export"
])

with tab1:
    st.header("🤖 AI Agricultural Advisor Chatbot")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location = st.text_input(
            "📍 Enter Location",
            value="Delhi, India",
            help="Enter city name or coordinates (e.g., Delhi, Mumbai, Bangalore)"
        )
        
        if location:
            with st.spinner("🌤️ Fetching real-time weather data..."):
                weather_data = get_weather_data(location, st.session_state.weather_api_key)
            
            if weather_data:
                st.session_state.location_data = weather_data
                
                # Fetch NASA POWER data
                if 'lat' in weather_data and 'lon' in weather_data:
                    with st.spinner("🌍 Fetching soil and climate data from NASA..."):
                        nasa_data = get_nasa_power_data(weather_data['lat'], weather_data['lon'])
                        if nasa_data:
                            st.session_state.location_data.update(nasa_data)
    
    with col2:
        st.subheader("🌤️ Current Weather")
        if st.session_state.location_data:
            wd = st.session_state.location_data
            st.metric("Temperature", f"{wd.get('temperature', 'N/A')}°C")
            st.metric("Humidity", f"{wd.get('humidity', 'N/A')}%")
            st.metric("Condition", wd.get('condition', 'N/A'))
            st.metric("Rainfall", f"{wd.get('rainfall', 0)} mm")
        else:
            st.info("Enter location to see weather")
    
    st.markdown("---")
    
    # Soil Parameters
    st.subheader("🧪 Soil Parameters")
    col_n, col_p, col_k, col_ph = st.columns(4)
    
    with col_n:
        nitrogen = st.number_input("Nitrogen (N) kg/ha", min_value=0, max_value=150, value=50, step=5)
    with col_p:
        phosphorus = st.number_input("Phosphorus (P) kg/ha", min_value=0, max_value=150, value=50, step=5)
    with col_k:
        potassium = st.number_input("Potassium (K) kg/ha", min_value=0, max_value=200, value=50, step=5)
    with col_ph:
        ph = st.slider("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
    
    st.session_state.soil_params = {'N': nitrogen, 'P': phosphorus, 'K': potassium, 'pH': ph}
    
    st.markdown("---")
    
    # Chat Interface
    st.subheader("💬 Ask Your Question")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong></div>', unsafe_allow_html=True)
            st.markdown(message["content"])
    
    user_question = st.text_input(
        "Type your question here...",
        placeholder="e.g., What crops should I grow? How to improve soil fertility?",
        key="user_input"
    )
    
    col_ask, col_clear = st.columns([3, 1])
    
    with col_ask:
        ask_button = st.button("🚀 Ask AI Advisor", use_container_width=True, disabled=st.session_state.processing)
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and user_question and not st.session_state.processing:
        st.session_state.processing = True
        
        soil = st.session_state.soil_params
        context = f"""Location: {location}

Current Weather Conditions:
- Temperature: {st.session_state.location_data.get('temperature', 'N/A')}°C
- Humidity: {st.session_state.location_data.get('humidity', 'N/A')}%
- Rainfall: {st.session_state.location_data.get('rainfall', 0)} mm
- Condition: {st.session_state.location_data.get('condition', 'N/A')}
- Wind Speed: {st.session_state.location_data.get('wind_speed', 'N/A')} km/h
- UV Index: {st.session_state.location_data.get('uv_index', 'N/A')}

Soil Parameters:
- Nitrogen (N): {soil['N']} kg/hectare
- Phosphorus (P): {soil['P']} kg/hectare
- Potassium (K): {soil['K']} kg/hectare
- pH: {soil['pH']}

Question: {user_question}

Please provide a comprehensive, actionable answer based on the above conditions."""
        
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})
        
        with st.spinner("🤖 AI is thinking..."):
            if use_local_model and TRANSFORMERS_AVAILABLE and st.session_state.model_loaded:
                response = generate_with_lora(context, st.session_state.lora_model, st.session_state.tokenizer, max_tokens)
            else:
                groq_model = st.session_state.get('groq_model', 'llama-3.1-8b-instant')
                response = get_groq_recommendation(context, groq_model, temperature, max_tokens, st.session_state.groq_api_key)
        
        if response:
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
            # Auto-save to recommendation history
            if any(keyword in user_question.lower() for keyword in ['crop', 'grow', 'plant', 'recommend', 'suitable']):
                recommendation_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'location': location,
                    'question': user_question,
                    'weather': st.session_state.location_data.copy(),
                    'soil': soil.copy(),
                    'recommendation': response,
                    'model_used': st.session_state.get('groq_model', 'llama-3.1-8b-instant')
                }
                st.session_state.recommendation_history.append(recommendation_entry)
            
            st.session_state.processing = False
            st.rerun()
        else:
            st.error("Failed to get response. Please try again.")
            if st.session_state.chat_history and st.session_state.chat_history[-1].get('role') == 'user':
                st.session_state.chat_history.pop()
            st.session_state.processing = False

with tab2:
    st.header("🌍 Location-Based Analysis")
    
    if not st.session_state.location_data:
        st.info("👈 Go to Chatbot tab and enter a location first")
    else:
        wd = st.session_state.location_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Location Information")
            st.write(f"**Location:** {wd.get('location', 'N/A')}")
            st.write(f"**Coordinates:** {wd.get('lat', 'N/A')}, {wd.get('lon', 'N/A')}")
            st.write(f"**Timezone:** {wd.get('timezone', 'N/A')}")
            st.write(f"**Last Updated:** {wd.get('last_updated', 'N/A')}")
        
        with col2:
            st.subheader("🌤️ Weather Summary")
            st.metric("Temperature", f"{wd.get('temperature', 'N/A')}°C", f"Feels like {wd.get('feels_like', 'N/A')}°C")
            st.metric("Humidity", f"{wd.get('humidity', 'N/A')}%")
            st.metric("Pressure", f"{wd.get('pressure', 'N/A')} mb")
            st.metric("Visibility", f"{wd.get('visibility', 'N/A')} km")
        
        # Soil Health Score
        st.markdown("---")
        st.subheader("🧪 Soil Health Analysis")
        soil_score = calculate_soil_health_score(st.session_state.soil_params)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Health Score", f"{soil_score['total_score']}/100")
        with col2:
            st.metric("Health Level", soil_score['level'])
        with col3:
            score_color = {"Excellent": "🟢", "Good": "🔵", "Fair": "🟠", "Poor": "🔴"}
            st.write(f"**Status:** {score_color.get(soil_score['level'], '⚪')} {soil_score['level']}")
        
        # Quick recommendations
        if soil_score['recommendations']:
            with st.expander("💡 Quick Soil Improvement Tips"):
                for rec in soil_score['recommendations'][:3]:
                    st.write(f"• {rec}")
        
        # NASA POWER Data
        if 'avg_temperature_30d' in wd:
            st.markdown("---")
            st.subheader("🌍 NASA POWER Climate Data (30-day average)")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Avg Temp", f"{wd['avg_temperature_30d']:.1f}°C")
            with col2:
                st.metric("Avg Precip", f"{wd['avg_precipitation_30d']:.2f} mm")
            with col3:
                st.metric("Avg Humidity", f"{wd['avg_humidity_30d']:.1f}%")
            with col4:
                st.metric("Avg Wind", f"{wd['avg_wind_speed_30d']:.2f} m/s")
            with col5:
                st.metric("Solar Radiation", f"{wd['avg_solar_radiation_30d']:.2f} kWh/m²")
            
            st.caption(f"Data range: {wd.get('date_range', 'N/A')} ({wd.get('data_points', 0)} data points)")

with tab3:
    st.header("📊 Comprehensive Visualizations")
    
    if not st.session_state.location_data:
        st.info("👈 Enter location in Chatbot tab to see visualizations")
    else:
        wd = st.session_state.location_data
        
        # Weather Visualizations
        st.subheader("🌤️ Weather Metrics")
        weather_fig = create_weather_visualization(wd)
        st.plotly_chart(weather_fig, use_container_width=True)
        
        # Soil Visualizations
        st.subheader("🧪 Soil Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            soil_fig = create_soil_visualization(st.session_state.soil_params)
            st.plotly_chart(soil_fig, use_container_width=True)
        
        with col2:
            ph_fig = create_ph_visualization(st.session_state.soil_params['pH'])
            st.plotly_chart(ph_fig, use_container_width=True)
        
        # Additional metrics
        st.subheader("📈 Additional Metrics")
        
        metrics_data = {
            'Metric': ['Temperature', 'Humidity', 'Wind Speed', 'UV Index', 'Pressure', 'Visibility'],
            'Value': [
                wd.get('temperature', 0),
                wd.get('humidity', 0),
                wd.get('wind_speed', 0),
                wd.get('uv_index', 0),
                wd.get('pressure', 0),
                wd.get('visibility', 0)
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        fig_bar = px.bar(df_metrics, x='Metric', y='Value', color='Value', 
                        color_continuous_scale='Viridis', title="Weather Metrics Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.header("🌡️ 7-Day Weather Forecast")
    
    if not st.session_state.location_data:
        st.info("👈 Enter location in Chatbot tab to see forecast")
    else:
        location = st.session_state.location_data.get('location', 'Delhi, India')
        forecast_data = get_weather_forecast(location, st.session_state.weather_api_key, days=7)
        
        if forecast_data:
            st.subheader(f"📍 {forecast_data['location']}")
            
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            forecast_df['day_name'] = forecast_df['date'].dt.strftime('%a, %b %d')
            
            # Temperature chart
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=forecast_df['day_name'],
                y=forecast_df['max_temp'],
                name='Max Temp',
                line=dict(color='#ff6b6b', width=3),
                mode='lines+markers'
            ))
            fig_temp.add_trace(go.Scatter(
                x=forecast_df['day_name'],
                y=forecast_df['min_temp'],
                name='Min Temp',
                line=dict(color='#4ecdc4', width=3),
                mode='lines+markers',
                fill='tonexty',
                fillcolor='rgba(78, 205, 196, 0.2)'
            ))
            fig_temp.update_layout(
                title="Temperature Forecast (7 Days)",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Daily forecast cards
            st.subheader("📅 Daily Forecast Details")
            cols = st.columns(7)
            for idx, day in enumerate(forecast_data['forecast']):
                with cols[idx]:
                    st.metric(
                        day['date'].split('-')[2],
                        f"{day['max_temp']}°C",
                        f"Min: {day['min_temp']}°C"
                    )
                    st.caption(f"🌧️ {day['rainfall']}mm")
                    st.caption(f"💨 {day['wind_speed']} km/h")
                    st.caption(day['condition'])
        else:
            st.warning("Weather forecast data unavailable")

with tab5:
    st.header("🌾 Crop Recommendations & Soil Health")
    
    if not st.session_state.location_data:
        st.info("👈 Enter location and soil parameters in Chatbot tab")
    else:
        # Soil Health Score
        st.subheader("🧪 Soil Health Analysis")
        soil_score = calculate_soil_health_score(st.session_state.soil_params)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{soil_score['total_score']}/100", delta=soil_score['level'])
        with col2:
            st.metric("Health Level", soil_score['level'])
        with col3:
            st.metric("Status", "✅ Good" if soil_score['total_score'] >= 60 else "⚠️ Needs Improvement")
        
        # Score breakdown
        st.subheader("📊 Nutrient Breakdown")
        score_df = pd.DataFrame([
            {'Nutrient': 'Nitrogen (N)', 'Score': soil_score['breakdown']['N']},
            {'Nutrient': 'Phosphorus (P)', 'Score': soil_score['breakdown']['P']},
            {'Nutrient': 'Potassium (K)', 'Score': soil_score['breakdown']['K']},
            {'Nutrient': 'pH Level', 'Score': soil_score['breakdown']['pH']}
        ])
        
        fig_score = px.bar(
            score_df, 
            x='Nutrient', 
            y='Score',
            color='Score',
            color_continuous_scale='RdYlGn',
            title="Soil Nutrient Scores"
        )
        fig_score.update_layout(height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Recommendations
        if soil_score['recommendations']:
            st.subheader("💡 Soil Improvement Recommendations")
            for rec in soil_score['recommendations']:
                st.info(f"• {rec}")
        
        # Crop Recommendations from History
        st.markdown("---")
        st.subheader("🌾 Recommended Crops")
        
        if st.session_state.recommendation_history:
            # Get latest recommendation
            latest = st.session_state.recommendation_history[-1]
            st.markdown("**Latest AI Recommendation:**")
            with st.expander("View Full Recommendation", expanded=True):
                st.markdown(latest.get('recommendation', 'No recommendation available'))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Location:**", latest.get('location', 'N/A'))
                    st.write("**Date:**", latest.get('timestamp', 'N/A'))
                with col2:
                    st.write("**Weather:**", f"{latest.get('weather', {}).get('temperature', 'N/A')}°C")
                    st.write("**Soil pH:**", latest.get('soil', {}).get('pH', 'N/A'))
        else:
            st.info("💬 Ask about crops in the Chatbot tab to get recommendations!")

with tab6:
    st.header("📜 Recommendation History & Export")
    
    if st.session_state.recommendation_history:
        st.success(f"✅ {len(st.session_state.recommendation_history)} recommendations saved")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            json_data = json.dumps(st.session_state.recommendation_history, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            df_export = pd.DataFrame([
                {
                    'Timestamp': rec.get('timestamp', ''),
                    'Location': rec.get('location', ''),
                    'Question': rec.get('question', ''),
                    'Temperature': rec.get('weather', {}).get('temperature', ''),
                    'Humidity': rec.get('weather', {}).get('humidity', ''),
                    'N': rec.get('soil', {}).get('N', ''),
                    'P': rec.get('soil', {}).get('P', ''),
                    'K': rec.get('soil', {}).get('K', ''),
                    'pH': rec.get('soil', {}).get('pH', ''),
                    'Recommendation': rec.get('recommendation', '')[:200] + '...' if len(rec.get('recommendation', '')) > 200 else rec.get('recommendation', '')
                }
                for rec in st.session_state.recommendation_history
            ])
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Clear history
            if st.button("🗑️ Clear All History"):
                st.session_state.recommendation_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Display history
        st.subheader("📋 All Recommendations")
        for idx, rec in enumerate(reversed(st.session_state.recommendation_history), 1):
            with st.expander(f"#{len(st.session_state.recommendation_history) - idx + 1} - {rec.get('timestamp', 'N/A')} | {rec.get('location', 'N/A')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📍 Location:**", rec.get('location', 'N/A'))
                    st.write("**❓ Question:**", rec.get('question', 'N/A'))
                    st.write("**📅 Date:**", rec.get('timestamp', 'N/A'))
                    st.write("**🤖 Model:**", rec.get('model_used', 'N/A'))
                
                with col2:
                    weather = rec.get('weather', {})
                    st.write("**🌤️ Weather:**")
                    st.write(f"- Temp: {weather.get('temperature', 'N/A')}°C")
                    st.write(f"- Humidity: {weather.get('humidity', 'N/A')}%")
                    st.write(f"- Rainfall: {weather.get('rainfall', 0)}mm")
                    
                    soil = rec.get('soil', {})
                    st.write("**🧪 Soil:**")
                    st.write(f"- N: {soil.get('N', 'N/A')}, P: {soil.get('P', 'N/A')}, K: {soil.get('K', 'N/A')}")
                    st.write(f"- pH: {soil.get('pH', 'N/A')}")
                
                st.markdown("**💡 Recommendation:**")
                st.markdown(rec.get('recommendation', 'N/A'))
                st.markdown("---")
    else:
        st.info("📝 No recommendations yet. Start chatting in the Chatbot tab to generate recommendations!")
        st.markdown("""
        **How to generate recommendations:**
        1. Go to the **💬 Chatbot** tab
        2. Enter a location
        3. Set soil parameters
        4. Ask questions like:
           - "What crops should I grow?"
           - "What's the best crop for my soil?"
           - "Recommend suitable crops"
        
        Recommendations will be automatically saved here!
        """)

# Load LoRA model if requested
if use_local_model and TRANSFORMERS_AVAILABLE and not st.session_state.model_loaded:
    current_model = st.session_state.get('model_choice', 'climate_advisor_lora')
    with st.spinner("Loading LoRA model... This may take a few minutes..."):
        model, tokenizer = load_lora_model(current_model)
        if model and tokenizer:
            st.session_state.lora_model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("✅ LoRA model loaded!")
        else:
            st.warning("⚠️ LoRA model loading failed. Using Groq API instead.")
            use_local_model = False

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>🌾 Climate Resilience Chatbot for Farmers</strong></p>
    <p>Built with ❤️ | Powered by TinyLlama (LoRA) + Groq API + FreeWeather API + NASA POWER API</p>
    <p style="font-size: 12px; color: #999;">Research Project | K.R Mangalam University</p>
</div>
""", unsafe_allow_html=True)
