import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
import re

# DO_NOT_CHANGE########################################################
#######################################################################

st.set_page_config(
    page_title='NIQ APP | HOME',
    layout='centered',
    initial_sidebar_state='collapsed'
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: display;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

image = Image.open('images_main/NIQ_banner.png')

st.image(image, use_column_width='always', output_format='PNG')

selected = option_menu(
    menu_title=None,
    options=['Home', 'Sampling', 'Replacing'],
    icons=['house', 'calculator', 'archive'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important",
                      "background-color": "#fafafa"},
        "icon": {"color": "#31d1ff", "font-size": "15px"},
        "nav-link": {"color": "#31333F", "font-size": "15px",
                     "text-align": "centered",
                     "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"color": "#FFFFFF",
                              "background-color": "#090a47"},
    }
)

if selected == 'Home':
    st.header('List of available apps and sub apps')
    subhead_app_1 = '''
    <style>
    .subhead-item {
        backgroundcolor: transparent;
    }
    .subhead-item:hover {
        color: #2E6EF7;
    }
    </style>

    <a style='display: inline; text-align: left; color: #31333F
    ; text-decoration: none; '
    href="/Sampling" target="_self">
    <h3 class="subhead-item">
    Sampling app
    </h3>
    </a>
    '''
    st.write(subhead_app_1, unsafe_allow_html=True)
    st.latex(r'n = \frac{NZ^{2}pq}{e^{2}(N-1)+Z^{2}pq}')
    app_1_topic = 'Description'
    st.write(
        f'''<div style="text-align: left; color: #31d1ff;">
        {app_1_topic}</div>''',
        unsafe_allow_html=True)
    app_1_cap = f'''
    You can use it directly as a calculator
    or you can upload the Dataframe to obtain your sample size.
    After that you can sample your Dataframe or
    re-sample it if you like, and download it.
    Futhermore, you can remove items from the sampled Dataframe
    by selecting the columns that you are most interested to mantain
    the structure, and the stores to be replaced.
    The stores to be replaced can be uploaded too, if you want.
    '''
    st.caption(
        f'''
        <div style="text-align: justify;
        margin-top: 5px;
        ">{app_1_cap}</div>''',
        unsafe_allow_html=True)
    st.header('')

    subhead_app_2 = '''
    <style>
    .subhead-item {
        backgroundcolor: transparent;
    }
    .subhead-item:hover {
        color: #2E6EF7;
    }
    </style>

    <a style='display: inline; text-align: left; color: #31333F
    ; text-decoration: none; '
    href="/Replacing" target="_self">
    <h3 class="subhead-item">
    Replacing app
    </h3>
    </a>
    '''
    st.write(subhead_app_2, unsafe_allow_html=True)
    app_2_topic = 'Description'
    st.write(
        f'''<div style="text-align: left; color: #31d1ff;">
        {app_2_topic}</div>''',
        unsafe_allow_html=True)
    app_2_cap = f'''
    By uploading a Master Dataframe and a Working Dataframe
    you can remove items from the Working Dataframe
    by selecting the columns that you are most interested in to mantain
    the structure, and the items to be replaced with items from the
    Mater Dataframe.
    The items to be replaced can be uploaded too, if you want.
    '''
    st.caption(
        f'''
        <div style="text-align: justify;
        margin-top: 5px;
        ">{app_2_cap}</div>''',
        unsafe_allow_html=True)
    st.header('')

    subhead_app_3 = '''
    <style>
    .subhead-item {
        backgroundcolor: transparent;
    }
    .subhead-item:hover {
        color: #2E6EF7;
    }
    </style>

    <a style='display: inline; text-align: left; color: #31333F
    ; text-decoration: none; '
    href="/Random_sampling" target="_self">
    <h3 class="subhead-item">
    Random Sampling
    </h3>
    </a>
    '''
    st.write(subhead_app_3, unsafe_allow_html=True)
    app_3_topic = 'Description'
    st.write(
        f'''<div style="text-align: left; color: #31d1ff;">
        {app_3_topic}</div>''',
        unsafe_allow_html=True)
    app_3_cap = f'''
    By uploading a Dataframe you can sample it by providing the sample size value.
    This sample will take randomly the number provided.
    '''
    st.caption(
        f'''
        <div style="text-align: justify;
        margin-top: 5px;
        ">{app_3_cap}</div>''',
        unsafe_allow_html=True)
    st.header('')

    subhead_app_4 = '''
    <style>
    .subhead-item {
        backgroundcolor: transparent;
    }
    .subhead-item:hover {
        color: #2E6EF7;
    }
    </style>

    <a style='display: inline; text-align: left; color: #31333F
    ; text-decoration: none; '
    href="/Structure_sampling" target="_self">
    <h3 class="subhead-item">
    Structured Sampling
    </h3>
    </a>
    '''
    st.write(subhead_app_4, unsafe_allow_html=True)
    app_4_topic = 'Description'
    st.write(
        f'''<div style="text-align: left; color: #31d1ff;">
        {app_4_topic}</div>''',
        unsafe_allow_html=True)
    app_4_cap = f'''
    By uploading a Dataframe you can sample it by providing the sample size value.
    This sample will take the number provided along with the selected structure
    given by your needs. It also generates plots if you want.
    '''
    st.caption(
        f'''
        <div style="text-align: justify;
        margin-top: 5px;
        ">{app_4_cap}</div>''',
        unsafe_allow_html=True)
    st.header('')

    subhead_app_5 = '''
    <style>
    .subhead-item {
        backgroundcolor: transparent;
    }
    .subhead-item:hover {
        color: #2E6EF7;
    }
    </style>

    <a style='display: inline; text-align: left; color: #31333F
    ; text-decoration: none; '
    href="/Parameters_sampling" target="_self">
    <h3 class="subhead-item">
    Parameters Sampling
    </h3>
    </a>
    '''
    st.write(subhead_app_5, unsafe_allow_html=True)
    app_5_topic = 'Description'
    st.write(
        f'''<div style="text-align: left; color: #31d1ff;">
        {app_5_topic}</div>''',
        unsafe_allow_html=True)
    app_5_cap = f'''
    By uploading a Dataframe you can sample it by selecting one of the two
    main paramaters, and their specifications.
    '''
    st.caption(
        f'''
        <div style="text-align: justify;
        margin-top: 5px;
        ">{app_5_cap}</div>''',
        unsafe_allow_html=True)
    st.header('')

if selected == 'Sampling':
    switch_page('Sindex')

if selected == 'Replacing':
    switch_page('Replacing')


#######################################################################

ft = """
<style>
a:link , a:visited{
color: #808080;  /* theme's text color at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}
a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}
#page-container {
  position: relative;
  min-height: 10vh;
}
footer{
    visibility:hidden;
}
.footer {
position: relative;
left: 0;
top:230px;
bottom: 0;
width: 100%;
background-color: transparent;
color: #BFBFBF; /* theme's text color at 50 percent brightness*/
text-align: left; /* 'left', 'center' or 'right' if you want*/
}
</style>
<div id="page-container">
<div class="footer">
<p style='font-size: 0.875em;'>Developed by <a style='display: inline;
text-align:
left;' href="https://github.com/sape94" target="_blank">
<img src="https://i.postimg.cc/vBnHmZfF/innovation-logo.png"
alt="AI" height= "20"/><br>LatAm's Automation & Innovation Team.
</br></a>Version 1.3.0-b.1.</p>
</div>
</div>
"""
st.write(ft, unsafe_allow_html=True)
