import streamlit as st
import json
import numpy as np
import faiss
import ollama
from sklearn.preprocessing import normalize
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
searching_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_rwq6ciql.json")  # Replace with your Lottie URL
success_animation = load_lottie_url("https://assets2.lottiefiles.com/private_files/lf30_m6j5igxb.json")  # Replace with your Lottie URL

# Initialize the Ollama embedding model (this might require an API key)
ollama_client = ollama.Client()  # Replace with your API key

# Load data
with open("careers_data.json") as f:
    careers_data = json.load(f)

# Function to generate embeddings using Ollama
def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            if "embedding" in response:
                embedding = np.array(response['embedding']).astype('float32')
                embeddings.append(embedding)
            else:
                st.warning(f"No embedding found for text: {text}")
        except Exception as e:
            st.error(f"Error generating embedding for text '{text}': {e}")
    return embeddings

# Function to recommend careers
def recommend_careers(response_text):
    try:
        response_embedding = get_embeddings(response_text)
        if not response_embedding or len(response_embedding) == 0:
            st.error("Failed to generate embedding for user response.")
            return []
        response_embedding = normalize(response_embedding[0].reshape(1, -1))[0]

        career_embeddings = []
        valid_careers = []
        for career in careers_data:
            embedding = get_embeddings(careers_data[career]["description"])
            if embedding:
                embedding = normalize(embedding[0].reshape(1, -1))[0]
                career_embeddings.append(embedding)
                valid_careers.append(career)

        if not career_embeddings:
            st.error("No career embeddings were generated.")
            return []

        dimension = career_embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(career_embeddings, dtype='float32'))

        distances, indices = index.search(response_embedding.reshape(1, -1), 5)

        recommendations = [
            (valid_careers[idx], max(0, 1 - distances[0][i]) * 100)
            for i, idx in enumerate(indices[0]) if idx != -1
        ]
        return recommendations
    except Exception as e:
        st.error(f"Error generating career recommendations: {e}")
        return []

# UI Design
st.set_page_config(page_title="Career Guidance Tool", layout="centered")
st.title("🎯 Career Guidance Psychometric Assessment")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Navigate to", ["Home", "Assessment", "Results", "Chatbot"])

# Home Page
if page == "Home":
    st.write("""
    Welcome to the enhanced **Career Guidance Tool**! 🎉  
    - Answer tailored questions to guide your career choices.  
    - Get recommendations and learning resources.  
    - Take the next step toward your dream job!
    """)
    st.image("career_banner.jpg", use_container_width=True)

# Assessment Page
elif page == "Assessment":
    st.subheader("📝 Fill out the assessment below:")
    with st.form("career_form"):
        responses = [
            st.text_input("What are your hobbies?", help="E.g., painting, hiking, coding"),
            st.text_input("Describe your ideal work environment.", help="E.g., remote, collaborative"),
            st.text_input("What tasks do you enjoy most?", help="E.g., problem-solving, planning"),
            st.text_input("What social causes are you passionate about?", help="E.g., climate action, education"),
            st.text_input("What type of extracurricular activities are you involved in?", help="E.g., sports, community service"),
            st.text_input("What skills are you currently developing?", help="E.g., data analysis, negotiation"),
        ]
        submitted = st.form_submit_button("Get Career Recommendations")

        if submitted:
            response_text = " ".join(responses)
            if response_text.strip():
                with st.spinner("Generating recommendations..."):
                    st_lottie(searching_animation, height=300)
                    recommendations = recommend_careers(response_text)
                    st.session_state['recommendations'] = recommendations
                st.success("Recommendations generated! 🎉")
                st_lottie(success_animation, height=200)
            else:
                st.warning("Please complete all fields before submitting.")

# Results Page
elif page == "Results":
    st.subheader("✨ Your Recommended Careers:")
    if 'recommendations' in st.session_state:
        for career, score in st.session_state['recommendations']:
            details = careers_data.get(career, {})
            with st.expander(f"{career} (Score: {score:.2f}%)"):
                st.markdown(f"**Description:** {details.get('description', 'N/A')}")
                st.markdown(f"**Key Skills:** {', '.join(details.get('skills', []))}")
                st.markdown("**Learning Resources:**")
                for resource in details.get('resources', []):
                    st.write(f"- [{resource['name']}]({resource['link']})")
    else:
        st.info("No recommendations yet. Complete the assessment first!")

# Chatbot Page
elif page == "Chatbot":
    st.subheader("🤖 Career Guidance Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input", help="Ask me about career guidance, skills, or anything!")
    if st.button("Send"):
        if user_input.strip():
            # Mock chatbot logic (can be enhanced with AI like GPT)
            bot_response = f"I'm here to help! For '{user_input}', let me guide you further."
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", bot_response))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"🟢 **{speaker}:** {message}")
