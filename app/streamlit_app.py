"""
streamlit_app.py — GUI for the bedtime-science model.

Run:  streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Bedtime Science", page_icon="🌙")
st.title("🌙 Bedtime Science")
st.caption("Paste a research abstract — get a bedtime story a five-year-old could follow.")

abstract = st.text_area("Research abstract", height=200,
                        placeholder="Paste an arXiv abstract here...")

col1, col2 = st.columns(2)
temperature = col1.slider("Temperature", 0.1, 1.5, 0.8, 0.1)
top_k = col2.slider("Top-k", 0, 100, 40, 5)

if st.button("✨ Tell me a story"):
    if not abstract.strip():
        st.warning("Please paste an abstract first.")
    else:
        # TODO: load trained model + tokenizer (cache with st.cache_resource)
        # TODO: story = generate(model, tokenizer, abstract, temperature=..., top_k=...)
        st.info("TODO: hook up the trained model here.")
        # st.success(story)
        # st.metric("Readability drop (grade levels)", readability_drop(abstract, story))
