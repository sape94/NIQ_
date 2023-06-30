import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
from app_modules import sampling_module as samp_mod
# from app_modules import replacing_module as repl_mod
import matplotlib.pyplot as plt
import numpy as np
from app_modules import niv_sample_selection as nss

# DO_NOT_CHANGE########################################################
#######################################################################

st.set_page_config(
    page_title='NIQ APP | Sampling',
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
    options=['Home', 'Sampling', 'Replacing', ''],
    icons=['house', 'calculator', 'archive', 'arrow-left-circle-fill'],
    menu_icon='cast',
    default_index=1,
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
    switch_page('NIQ p app')

if selected == '':
    switch_page('Sindex')

if selected == 'Sampling':
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

    with st.expander('Expand this section to upload your Dataframe. When you finish you can collapse it again.'):
        st.write(
            'Upload the CSV file that contains the working Dataframe:')
        uploaded_file = st.file_uploader("Choose a file",
                                         type=['csv'],
                                         key='gral_settings_df'
                                         )
        if uploaded_file is not None:
        
            o_df = pd.read_csv(uploaded_file, encoding='UTF8')
            
            file_name_df = uploaded_file.name.replace('.csv', '')
           
            st.write(o_df)

    st.markdown('')

    if uploaded_file is None:
        st.caption('<p style="color: #2e6ef7;">Please upload a Dataframe to continue.</p>',
                   unsafe_allow_html=True)

    if uploaded_file is not None:
        # st.write(r'Type the **sample size**, $n$:')
        col_pre_input, col_input_n, col_input_n_2 = st.columns(
            [1, 5, 1], gap='medium')

        with col_input_n:
            n_s = st.number_input(
                r'Type the **sample size**, $n$:', min_value=1)
            n = int(n_s)

        st.write('')
        st.write('')
        with st.expander('Expand this section to show your **Randomly Sampled Dataframe**:'):
            sampled_df = o_df.sample(n=n)
            if st.button(':inbox_tray: Press here to re-sample :inbox_tray:'):
                o_df = o_df.sample(frac=1)
                sampled_df = o_df.sample(n=n)

            st.write(sampled_df)
            sampled_df_csv = sampled_df.to_csv(index=False)

            coldoss, coldos = st.columns(2, gap='medium')

            with coldos:
                st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                   data=sampled_df_csv,
                                   file_name=f'SAMPLED_{file_name_df}.csv',
                                   mime='text/csv')

            st.write('')
            st.write('Don\'t forget to **download** your sampled Dataframe.')
            st.write(
                'If you want to remove stores from the sampled Dataframe use our:')
            subhead_app_2 = '''
            <style>
            .subhead-item_2 {
                color: #2E6EF7;
                backgroundcolor: transparent;
            }
            .subhead-item_2:hover {
                color: #164fc9;
            }
            </style>

            <a style='display: inline; text-align: center; color: #31333F
            ; text-decoration: none; '
            href="/Replacing" target="_self">
            <h5 class="subhead-item_2">
            Replacing app
            </h5>
            </a>
            '''
            st.write(subhead_app_2, unsafe_allow_html=True)
            st.write('')
            st.write('')


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
