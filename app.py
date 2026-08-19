import streamlit as st
import requests

st.set_page_config(page_title="Ugandan Multilingual Translator", page_icon="🇺🇬")
st.title("Ugandan Multilingual Translator")

# Language choices mapping
LANGUAGES = {
    "English": "English",
    "Runyankole": "Runyankole",
    "Luganda": "Luganda",
    "Acholi": "Acholi",
    "Ateso": "Ateso",
    "Lugbara": "Lugbara"
}

col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("From:", list(LANGUAGES.keys()), index=0)
with col2:
    target_lang = st.selectbox("To:", list(LANGUAGES.keys()), index=1)

input_text = st.text_area("Enter text to translate:", height=120)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text before translating.")
    else:
        with st.spinner("Translating..."):
            try:
                # Sunbird API endpoints
                url = "https://api.sunbird.ai/tasks/translate"
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Attach token if you created one via Sunbird API
                if "SUNBIRD_API_KEY" in st.secrets:
                    headers["Authorization"] = f"Bearer {st.secrets['SUNBIRD_API_KEY']}"

                payload = {
                    "source_language": LANGUAGES[source_lang],
                    "target_language": LANGUAGES[target_lang],
                    "text": input_text
                }

                response = requests.post(url, json=payload, headers=headers, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    # Handles various payload key formats
                    translation = data.get("text") or data.get("translated_text") or str(data)
                    st.subheader("Translation:")
                    st.success(translation)
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
