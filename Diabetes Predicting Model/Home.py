import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.write("# Welcome to Health Assistant! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Health Assistant is AI based multiple disease predicting model Anyone can easily know their self 
    they are Heart disease petient or not, it's trust Worthy because it has made by Data Scientists 
    with Machine Learning.
    **👈 Select a demo from the sidebar** to see some examples
    of what Health Assistant can do!
    ### Want to connect with us ?
    - Contact on (https://www.linkedin.com/in/shubham-khatwase)
    - Github profile(https://github.com/shubhkhatwase)
    - Ask a question or Query so mail on (shubh7265@gmail.com)
    
"""
)