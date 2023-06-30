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
        # o_df = pd.DataFrame()

        st.write('')
        st.write('')
    if uploaded_file is not None:
        with st.expander('Expand this section to continue.'):
            # sampled_df = o_df.sample(n=n)
            pre_par_df = o_df.copy()
            st.write('')
            param_method_ans = st.radio('Which **method** would you like to use?',
                                        ('Parameters\' structure preserving.',
                                            'ACV and Stores parameters maximizing.'))
            if param_method_ans == 'Parameters\' structure preserving.':
                st.write('')
                st.write(
                    'By which **parameters** the most relevant cities in the universe will be selected?')
                param_list = ['ACV', 'Stores']
                par_samp_list = st.multiselect(
                    'This would be the most relevant cities in the universe.', param_list, max_selections=1)
                if par_samp_list == []:
                    st.caption('<p style="color: #2e6ef7;">You must select the parameters if you want to continue.</p>',
                               unsafe_allow_html=True)
                if par_samp_list != []:
                    st.write('')
                    st.write(
                        f'Select the {par_samp_list[0]} percentage that the principal cities will cover:')
                    input_percent = st.slider(
                        r'Select the percentage (%):', 0, 100, 50)
                    input_val = input_percent / 100
                    niv_design = nss.NIV_Sample_Selection(data=pre_par_df, parameter_acv='0.5', parameter_stores='0.5',
                                                          structure='Cities', reduction=par_samp_list[0], cities_weight=input_val)
                    stc_niv = niv_design.structure_preserving_sample()
                    # text_closet = niv_design.a_text()
                    # st.write(
                    #    f'The closest value to the selected one is {text_closet}, and the {par_samp_list[0]} is concentrated in the following Dataframe:')
                    st.write(stc_niv)
                    stc_niv_df = stc_niv.to_csv(index=False)
                    coldos_par_acv_1, coldos_par_acv_2 = st.columns(
                        2, gap='medium')

                    with coldos_par_acv_2:
                        st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                           data=stc_niv_df,
                                           file_name=f'STRUCTURE_PRESERVING_ACV_{file_name_df}.csv',
                                           mime='text/csv')

            if param_method_ans == 'ACV and Stores parameters maximizing.':
                st.write('')
                st.write(
                    'Select the **structure** that the Sample will preserve:')
                par_struct_list = ['Cities', 'Universe']
                par_struct_samp_list = st.multiselect(
                    'This would determine the structure of this Sample.', par_struct_list, max_selections=1)
                if par_struct_samp_list == []:
                    st.caption('<p style="color: #2e6ef7;">You must select the structure if you want to continue.</p>',
                               unsafe_allow_html=True)
                if par_struct_samp_list != []:
                    struct_val = par_struct_samp_list[0].lower()
                    st.write('')
                    st.write(
                        'Select the respective percentage that the Sample will target.')
                    param_sample_col_1, param_sample_col_2 = st.columns(
                        2, gap='medium')
                    with param_sample_col_1:
                        acv_percent = st.slider(
                            r'ACV (%):', 0, 100, 50)
                        acv_val = acv_percent / 100
                    with param_sample_col_2:
                        stores_percent = st.slider(
                            r'Stores (%):', 0, 100, 50)
                        stores_val = stores_percent / 100
                    niv_design = nss.NIV_Sample_Selection(
                        data=pre_par_df, parameter_acv=acv_val, parameter_stores=stores_val, structure=par_struct_samp_list[0], reduction='ACV', cities_weight='0.5')
                    max_acv_niv = niv_design.acv_maximizing_sample()

                    st.write(max_acv_niv)
                    max_acv_niv_df = max_acv_niv.to_csv(index=False)
                    coldos_par_acv_1, coldos_par_acv_2 = st.columns(
                        2, gap='medium')

                    with coldos_par_acv_2:
                        st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                           data=max_acv_niv_df,
                                           file_name=f'MAX_ACV_{file_name_df}.csv',
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
