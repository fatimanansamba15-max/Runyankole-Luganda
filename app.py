import streamlit as st
import torch
import transformers

st.set_page_config(page_title="Ugandan Multilingual Translator", page_icon="🇺🇬")
st.title("Ugandan Multilingual Translator")

# 4-bit quantized Sunbird model fits within memory limits
MODEL_NAME = "Sunbird/translate-nllb-1.3b-salt-4bit"
TOKENIZER_NAME = "Sunbird/translate-nllb-1.3b-salt"

@st.cache_resource
def load_translation_engine():
    tokenizer = transformers.NllbTokenizer.from_pretrained(TOKENIZER_NAME)
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        load_in_4bit=True
    )
    return tokenizer, model

with st.spinner("Loading lightweight translation engine..."):
    tokenizer, model = load_translation_engine()

LANG_TOKENS = {
    "English": 256047,
    "Runyankole": 256002,
    "Luganda": 256110,
    "Acholi": 256111,
    "Ateso": 256006,
    "Lugbara": 256008,
}

col1, col2 = st.columns(2)
with col1:
    src_name = st.selectbox("From:", list(LANG_TOKENS.keys()), index=0)
with col2:
    tgt_name = st.selectbox("To:", list(LANG_TOKENS.keys()), index=1)

input_text = st.text_area("Enter text to translate:", height=120)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text first.")
    else:
        with st.spinner("Translating..."):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                inputs['input_ids'][0][0] = LANG_TOKENS[src_name]

                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=LANG_TOKENS[tgt_name],
                    max_length=128,
                    num_beams=3
                )
                
                result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                st.subheader("Translation:")
                st.success(result)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
