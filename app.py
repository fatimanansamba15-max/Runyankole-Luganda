import streamlit as st
import requests

st.set_page_config(page_title="Ugandan Multilingual Translator", page_icon="🇺🇬")
st.title("Ugandan Multilingual Translator")

# FLORES-200 Language Codes used by Sunbird AI
LANGUAGES = {
    "English": "eng_Latn",
    "Runyankole": "nyn_Latn",
    "Luganda": "lug_Latn",
    "Acholi": "ach_Latn",
    "Ateso": "teo_Latn",
    "Lugbara": "lgg_Latn"
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
                # Direct API call to Sunbird's hosted translation server
                url = "https://api.sunbird.ai/tasks/process"
                payload = {
                    "action": "translate",
                    "text": input_text,
                    "source_language": LANGUAGES[source_lang],
                    "target_language": LANGUAGES[target_lang]
                }
                
                response = requests.post(url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    translation = response.json().get("text", "")
                    st.subheader("Translation:")
                    st.success(translation)
                else:
                    st.error(f"Translation API returned status code {response.status_code}. Try again shortly.")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
