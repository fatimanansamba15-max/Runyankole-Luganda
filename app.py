import streamlit as st
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Ugandan Multilingual Translator", page_icon="🇺🇬")
st.title("Ugandan Multilingual Translator")

MODEL_NAME = "Sunbird/translate-nllb-1.3b-salt"

@st.cache_resource
def load_translation_engine():
    """Loads and caches tokenizer and model weights into RAM."""
    # Use AutoTokenizer instead of NllbTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

with st.spinner("Loading translation engine into memory..."):
    tokenizer, model, device = load_translation_engine()

# Explicit Sunbird language token ID mappings
LANGUAGE_TOKENS = {
    "English": {"code": "eng", "token_id": 256047},
    "Runyankole": {"code": "nyn", "token_id": 256002},
    "Luganda": {"code": "lug", "token_id": 256110},
    "Acholi": {"code": "ach", "token_id": 256111},
    "Ateso": {"code": "teo", "token_id": 256006},
    "Lugbara": {"code": "lgg", "token_id": 256008},
}

col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("From:", list(LANGUAGE_TOKENS.keys()), index=0)
with col2:
    target_lang_name = st.selectbox("To:", list(LANGUAGE_TOKENS.keys()), index=1)

input_text = st.text_area("Enter text to translate:", height=120)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text before translating.")
    else:
        source_token_id = LANGUAGE_TOKENS[source_lang_name]["token_id"]
        target_token_id = LANGUAGE_TOKENS[target_lang_name]["token_id"]

        with st.spinner("Translating..."):
            try:
                # Tokenize input text
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                # Overwrite the first BOS token ID with the source language token ID
                inputs['input_ids'][0][0] = source_token_id
                
                # Generate translated tokens forcing target language BOS ID
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=128,
                    num_beams=5
                )
                
                # Decode output back into plain text
                result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.subheader("Translation:")
                st.success(result)
            except Exception as e:
                st.error(f"Translation Error: {str(e)}")
