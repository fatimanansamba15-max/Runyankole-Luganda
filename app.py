import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Ugandan Languages Translator", page_icon="🇺🇬")
st.title("Ugandan Multilingual Translator")

# Optimized distilled model for free Streamlit Cloud RAM limits
MODEL_NAME = "facebook/nllb-200-distilled-600M"

@st.cache_resource
def load_translation_engine():
    """Loads and caches tokenizer and model weights into RAM."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

with st.spinner("Loading translation engine into memory..."):
    tokenizer, model, device = load_translation_engine()

# Standard NLLB language codes
LANGUAGES = {
    "English": "eng_Latn",
    "Runyankole": "nyn_Latn",
    "Luganda": "lug_Latn"
}

col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("From:", list(LANGUAGES.keys()), index=0)
with col2:
    target_lang_name = st.selectbox("To:", list(LANGUAGES.keys()), index=1)

input_text = st.text_area("Enter text to translate:", height=120)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text before translating.")
    else:
        src_code = LANGUAGES[source_lang_name]
        tgt_code = LANGUAGES[target_lang_name]

        with st.spinner("Translating..."):
            try:
                # Set source language context
                tokenizer.src_lang = src_code
                
                # Tokenize input text
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                # Fetch target language token ID using convert_tokens_to_ids
                forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                # Generate translation
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_length=150,
                    num_beams=4
                )
                
                # Decode outputs back to text
                result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.subheader("Translation:")
                st.success(result)
            except Exception as e:
                st.error(f"Translation Error: {str(e)}")
