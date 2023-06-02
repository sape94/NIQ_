import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
from app_modules import replacing_module as repl_mod

# DO_NOT_CHANGE########################################################
#######################################################################

st.set_page_config(
    page_title='NIQ APP | Replacing',
    layout='centered',
    initial_sidebar_state='collapsed'
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
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
    default_index=2,
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


@st.cache_data
def cache_df(df):
    df = df.copy()
    c_df = df
    return c_df


def cache_df_2(df):
    df_2 = df.copy()
    c_df_2 = df_2
    return c_df_2


if selected == 'Home':
    switch_page('NIQ p app')

if selected == 'Sampling':
    switch_page('Sindex')

if selected == 'Replacing':
    st.subheader('Replacing app')
    st.write(
        'Please provide Dataframes such that the Master Dataframe contains the Working Dataframe.')
    st.write('')
    with st.expander('Upload the CSV or XLSX file that contains the  **Master Dataframe**:'):
        uploaded_file = st.file_uploader("Choose a file",
                                         type=['csv', 'xlsx'],
                                         key='master_df'
                                         )
        if uploaded_file is not None:
            try:
                o_df = pd.read_csv(uploaded_file, encoding='UTF8')
            except:
                o_df = pd.read_excel(uploaded_file, encoding='UTF8')

            try:
                file_name_df = uploaded_file.name.replace('.csv', '')
            except:
                try:
                    file_name_df = uploaded_file.name.replace('.xlsx', '')
                except:
                    pass
            o_df = cache_df(o_df)
            st.write(o_df)
        st.caption('You can collapse this section if you want.')
    st.markdown('')
    with st.expander('Upload the CSV or XLSX file that contains the  **Working Dataframe**:'):
        uploaded_file_2 = st.file_uploader("Choose a file",
                                           type=['csv', 'xlsx'],
                                           key='working_df'
                                           )
        if uploaded_file_2 is not None:
            try:
                no_df = pd.read_csv(uploaded_file_2, encoding='UTF8')
            except:
                no_df = pd.read_excel(uploaded_file_2, encoding='UTF8')

            try:
                file_name_df = uploaded_file_2.name.replace('.csv', '')
            except:
                try:
                    file_name_df = uploaded_file_2.name.replace('.xlsx', '')
                except:
                    pass
            no_df = cache_df_2(no_df)
            st.write(no_df)
        st.caption('You can collapse this section if you want.')

    if uploaded_file is None and uploaded_file_2 is None:
        st.write('')
        st.write(
            'Please upload the **Master Dataframe** and the **Working Dataframe**.')
    if uploaded_file is not None and uploaded_file_2 is None:
        st.write('')
        st.write('Please upload the **Working Dataframe**.')
    if uploaded_file is None and uploaded_file_2 is not None:
        st.write('')
        st.write('Please upload the **Master Dataframe**.')
    if uploaded_file is not None and uploaded_file_2 is not None:
        st.write('')
        df_cols = no_df.columns.to_list()
        st.write('Select the **sorting** column:')
        sort_col_list = st.multiselect(
            'This would be the column that determines the most important feature of the items.', df_cols, max_selections=1)
        if sort_col_list == []:
            st.caption('<p style="color: #2e6ef7;">You must select the most important column if you want a proper performance.</p>',
                       unsafe_allow_html=True)
            sort_col = ''
        if sort_col_list != []:
            sort_col = sort_col_list[0]

        st.write('')
        st.write('Select column to **identify** the items:')
        df_cols_3 = [col for col in df_cols if col not in sort_col]
        id_rmv_list = st.multiselect(
            'This would be up to three columns that will determine the grouping characteristics.', df_cols_3, max_selections=1)
        if id_rmv_list == []:
            st.caption('<p style="color: #2e6ef7;">You must select one column if you want a proper performance.</p>',
                       unsafe_allow_html=True)
            id_rmv = f'{df_cols_3[0]}'
        if id_rmv_list != []:
            id_rmv = f'{id_rmv_list[0]}'

            st.write('')
            st.write('Select columns by priority to get the **structure**:')
            df_cols_2 = [col for col in df_cols if col not in [id_rmv]]
            filter_list = st.multiselect(
                'This would be up to three columns that will determine the grouping characteristics.', df_cols_2, max_selections=3)
            if filter_list == []:
                st.caption('<p style="color: #2e6ef7;">You must select at least one column if you want a proper performance to mantain the Datframe structure.</p>',
                           unsafe_allow_html=True)
                est_col = [df_cols_2[0]]
            if filter_list != []:
                est_col = filter_list

            rmv_ans = st.radio(f'Provide the stores by {id_rmv} that you want to remove:',
                               ('Do it by selecting items from a list.',
                                'Upload a Dataframe that contains the items you want to remove.'))
            if rmv_ans == 'Do it by selecting items from a list.':
                rmv_list = st.multiselect(
                    f'Select the stores that you want to remove by {id_rmv}:', no_df[[f'{id_rmv}']])
                if rmv_list == []:
                    st.caption('<p style="color: #2e6ef7;">You dataframe is the same as before.</p>',
                               unsafe_allow_html=True)

            if rmv_ans == 'Upload a Dataframe that contains the items you want to remove.':
                up_rmv_file = st.file_uploader("Choose a file",
                                               type=['csv', 'xlsx'],
                                               key='settings_df'
                                               )

                if up_rmv_file is not None:
                    st.caption(
                        f'This dataframe must contain the column selected as identifier, otherwise an error message would appear.')
                    to_rmv_df = pd.read_csv(up_rmv_file)
                    file_name_df = up_rmv_file.name
                    rmv_list = to_rmv_df[id_rmv].to_list()
                else:
                    rmv_list = []
                    st.caption('<p style="color: #2e6ef7;">You dataframe is the same as before.</p>',
                               unsafe_allow_html=True)
            n_s_df = repl_mod.DataFrameReplacer(fulldf=o_df,
                                                fracdf=no_df,
                                                sort_col=sort_col).add_sts(est_col=est_col,
                                                                           rmv_list=rmv_list,
                                                                           id_rmv=id_rmv)

            new_sampled_df_csv = n_s_df.to_csv(index=False)

            st.write(n_s_df)

            col7, col8 = st.columns(2, gap='medium')

            with col8:
                st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                   data=new_sampled_df_csv,
                                   file_name=f'REPLACED_{file_name_df}.csv',
                                   mime='text/csv')

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
</br></a></p>
</div>
</div>
"""
st.write(ft, unsafe_allow_html=True)
