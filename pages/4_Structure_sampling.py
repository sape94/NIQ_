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
    href="/Structured_sampling" target="_self">
    <h3 class="subhead-item">
    Structured Sampling
    </h3>
    </a>
    '''
    st.write(subhead_app_4, unsafe_allow_html=True)

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
        with st.expander('Expand this section to continue.'):
            N = o_df.shape[0]
            o_df_cols = o_df.columns.to_list()
            st.write('Select the **identifier** column:')
            sort_col_list = st.multiselect(
                'This would be the column that uniquely identifies the items of the Dataframe.', o_df_cols, max_selections=1)
            if sort_col_list == []:
                st.caption('<p style="color: #2e6ef7;">You should select the identifier column if you want your Dataframe to be sorted.</p>',
                           unsafe_allow_html=True)
                sort_col = ''
                p_df = o_df
                grouped_df = p_df
                pivot_df = p_df
                pivot_df_2 = pivot_df
            if sort_col_list != []:
                sort_col = sort_col_list[0]
                p_df = o_df.sort_values(sort_col)

            st.write('')
            st.write('Select the **structure** parameters column(s):')
            p_df_cols = [
                item for item in o_df_cols if item != sort_col]
            par_col_list = st.multiselect(
                'This would be the parameters that will define the structure of the Dataframe for stratification.', p_df_cols)
            if par_col_list == []:
                st.caption('<p style="color: #2e6ef7;">You must select the parameters column(s) if you want to continue.</p>',
                           unsafe_allow_html=True)
                grouped_df = p_df
                pivot_df = grouped_df
                pivot_df_2 = pivot_df
            if par_col_list != []:
                grouped_df = p_df.groupby(
                    par_col_list).size().reset_index(name='Count')

                st.write('')
                feature_op_ans = st.radio('Do you want to select a feature column?',
                                          ('No.',
                                           'Yes.'))
                if feature_op_ans == 'No.':
                    pivot_df = grouped_df
                    w_pivot_df = pivot_df.copy()
                    for column in pivot_df.columns:
                        if column == 'Count':
                            weight_column_name = f'Weight(%)'
                            s_s_col_name = f'Sample_size_by_weight'
                            column_feature = column
                            w_pivot_df[column_feature] = pivot_df[column]
                            w_pivot_df[weight_column_name] = np.round(
                                (pivot_df[column] / N) * 100, 4)
                            w_pivot_df[s_s_col_name] = np.round(
                                (pivot_df[column] / N) * n)
                    st.write(
                        'Weighted Dataframe pivot given the selected structure:')
                    st.write(w_pivot_df)
                    pivot_structure_sampled_df_csv = w_pivot_df.to_csv(
                        index=False)
                    coldos_pivot_1, coldos_pivot_2 = st.columns(
                        2, gap='medium')

                    with coldos_pivot_2:
                        st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                           data=pivot_structure_sampled_df_csv,
                                           file_name=f'PIVOT_STRUCTURE_{file_name_df}.csv',
                                           mime='text/csv')
                    pre_est_samp_df = pd.DataFrame(columns=o_df_cols)
                    o_l_df = o_df.copy()
                    st.write('')
                    for _, row in w_pivot_df.iterrows():
                        temp_indices_list = []
                        n_temp = int(row['Sample_size_by_weight'])
                        filt_w_pivot_df = o_l_df.loc[(
                            o_l_df[par_col_list] == row[par_col_list]).all(axis=1)]
                        incomplete_est_samp_rows = filt_w_pivot_df.sample(
                            n=n_temp)
                        pre_est_samp_df = pd.concat(
                            [pre_est_samp_df, incomplete_est_samp_rows])
                        temp_indices_list.extend(
                            incomplete_est_samp_rows.index.tolist())
                        o_l_df = o_l_df.drop(temp_indices_list)
                    o_merge_pre_est_df = pd.merge(
                        o_df, pre_est_samp_df, how='left', indicator=True)
                    o_minus_pre_est_df = o_merge_pre_est_df[o_merge_pre_est_df['_merge'] == 'left_only'].drop(
                        columns='_merge')
                    a_auxiliar_df = w_pivot_df['Sample_size_by_weight'].value_counts(
                    ).sort_index().reset_index()
                    a_auxiliar_df.columns = [
                        'Sample_size_by_weight', 'Count']
                    d_auxiliar_df = a_auxiliar_df.sort_values(
                        'Sample_size_by_weight', ascending=False)
                    aux_eval = pre_est_samp_df.shape[0] - n
                    if aux_eval < 0:
                        sub_a_auxiliar = pd.DataFrame(
                            columns=a_auxiliar_df.columns)
                        temp_a_auxiliar_count = 0
                        for _, row in a_auxiliar_df.iterrows():
                            count = row['Count']
                            temp_a_auxiliar_count = temp_a_auxiliar_count + count
                            sub_a_auxiliar = pd.concat(
                                [sub_a_auxiliar, row.to_frame().T])
                            if temp_a_auxiliar_count >= abs(aux_eval):
                                break
                        sub_a_auxiliar = sub_a_auxiliar.reset_index(
                            drop=True)
                        values_a_filter = sub_a_auxiliar['Sample_size_by_weight'].tolist(
                        )
                        a_filt_w_pivot = w_pivot_df[w_pivot_df['Sample_size_by_weight'].isin(
                            values_a_filter)].sample(n=abs(aux_eval))
                        to_fill_a_w_pivot = pd.DataFrame()
                        for _, row in a_filt_w_pivot.iterrows():
                            temp_fill = o_minus_pre_est_df.loc[(
                                o_minus_pre_est_df[par_col_list] == row[par_col_list]).all(axis=1)].sample(n=1)
                            to_fill_a_w_pivot = pd.concat(
                                [to_fill_a_w_pivot, temp_fill])
                        est_samp_df = pd.concat(
                            [pre_est_samp_df, to_fill_a_w_pivot])
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        # st.write(est_samp_df.shape)
                        structure_sampled_l_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_l_1, coldos_est_l_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_l_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_l_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')
                    if aux_eval > 0:
                        sub_d_auxiliar = pd.DataFrame(
                            columns=d_auxiliar_df.columns)
                        temp_d_auxiliar_count = 0
                        for _, row in d_auxiliar_df.iterrows():
                            count = row['Count']
                            temp_d_auxiliar_count = temp_d_auxiliar_count + count
                            sub_d_auxiliar = pd.concat(
                                [sub_d_auxiliar, row.to_frame().T])
                            if temp_d_auxiliar_count >= abs(aux_eval):
                                break
                        sub_d_auxiliar = sub_d_auxiliar.reset_index(
                            drop=True)
                        values_d_filter = sub_d_auxiliar['Sample_size_by_weight'].tolist(
                        )
                        d_filt_w_pivot = w_pivot_df[w_pivot_df['Sample_size_by_weight'].isin(
                            values_d_filter)].sample(n=abs(aux_eval))
                        to_rem_d_w_pivot = pd.DataFrame()
                        for _, row in d_filt_w_pivot.iterrows():
                            temp_rem = pre_est_samp_df.loc[(
                                pre_est_samp_df[par_col_list] == row[par_col_list]).all(axis=1)].sample(n=1)
                            to_rem_d_w_pivot = pd.concat(
                                [to_rem_d_w_pivot, temp_rem])
                        pre_merge_to_rem = pd.merge(
                            pre_est_samp_df, to_rem_d_w_pivot, how='left', indicator=True)
                        est_samp_df = pre_merge_to_rem[pre_merge_to_rem['_merge'] == 'left_only'].drop(
                            columns='_merge')
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        structure_sampled_g_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_g_1, coldos_est_g_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_g_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_g_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')
                    if aux_eval == 0:
                        est_samp_df = pre_est_samp_df
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        structure_sampled_e_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_e_1, coldos_est_e_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_e_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_e_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')
                if feature_op_ans == 'Yes.':
                    st.write('Select the **feature** column:')
                    feat_col_list = st.multiselect(
                        'This would be the column that determines the features of the Dataframe.', par_col_list, max_selections=1)
                    if feat_col_list == []:
                        st.caption('<p style="color: #2e6ef7;">You must select the feature column if you want to continue.</p>',
                                   unsafe_allow_html=True)
                        pivot_df = grouped_df
                        pivot_df_2 = pivot_df
                    if feat_col_list != []:
                        features_ans = st.radio('Do you want to select only specific features?',
                                                ('No.',
                                                    'Yes.'))
                        feature = feat_col_list[0]
                        par_featureless = [
                            item for item in par_col_list if item != feature]
                        features_list = grouped_df[feature].unique(
                        ).tolist()
                        pivot_cols = par_featureless + features_list
                        if features_ans == 'No.':
                            pivot_sum = 0  # Verificación de suma
                            pivot_df = grouped_df.pivot_table(
                                index=par_featureless, columns=feature, values='Count', fill_value=0).reset_index()
                            pivot_df = pivot_df.reindex(
                                columns=pivot_cols)
                            pivot_df_2 = pivot_df[par_featureless].copy(
                            )
                            temp_pivot_df = pivot_df.copy()
                            pre_label = ', '.join(
                                str(item for item in par_featureless))
                            temp_pivot_df[pre_label] = temp_pivot_df.apply(
                                lambda row: ', '.join(str(row[item]) for item in par_featureless), axis=1)
                            for column in pivot_df.columns:
                                if column not in par_featureless:
                                    weight_column_name = f'{column}_weight(%)'
                                    s_s_col_name = f'{column}_sample_size_by_weight'
                                    column_feature = column
                                    pivot_df_2[column_feature] = pivot_df[column]
                                    pivot_df_2[weight_column_name] = np.round(
                                        (pivot_df[column] / N) * 100, 4)
                                    pivot_df_2[s_s_col_name] = np.round(
                                        (pivot_df[column] / N) * n)
                                    pivot_sum = pivot_sum + \
                                        pivot_df_2[s_s_col_name].sum()
                        if features_ans == 'Yes.':
                            st.write(
                                'Select the **features** of interest:')
                            interest_features_list = st.multiselect(
                                'This would be the interesting features of the Dataframe.', features_list)
                            if interest_features_list == []:
                                st.caption('<p style="color: #2e6ef7;">You must select at least one feature.</p>',
                                           unsafe_allow_html=True)
                                pivot_df = grouped_df.pivot_table(
                                    index=par_featureless, columns=feature, values='Count', fill_value=0).reset_index()
                                pivot_df = pivot_df.reindex(
                                    columns=pivot_cols)
                                pivot_df_2 = pivot_df[par_featureless].copy(
                                )
                                temp_pivot_df = pivot_df.copy()
                                pre_label = ', '.join(
                                    str(item for item in par_featureless))
                                temp_pivot_df[pre_label] = temp_pivot_df.apply(
                                    lambda row: ', '.join(str(row[item]) for item in par_featureless), axis=1)
                                for column in pivot_df.columns:
                                    if column not in par_featureless:
                                        weight_column_name = f'{column}_weight(%)'
                                        s_s_col_name = f'{column}_sample_size_by_weight'
                                        column_feature = column
                                        pivot_df_2[column_feature] = pivot_df[column]
                                        pivot_df_2[weight_column_name] = np.round(
                                            (pivot_df[column] / N) * 100, 4)
                                        pivot_df_2[s_s_col_name] = np.round(
                                            (pivot_df[column] / N) * n)
                            if interest_features_list != []:
                                pivot_df = grouped_df.pivot_table(
                                    index=par_featureless, columns=feature, values='Count', fill_value=0).reset_index()
                                pivot_cols_interest = par_featureless + interest_features_list
                                pivot_df = pivot_df.reindex(
                                    columns=pivot_cols_interest)
                                pivot_df_2 = pivot_df[par_featureless].copy(
                                )
                                temp_pivot_df = pivot_df.copy()
                                pre_label = ', '.join(
                                    str(item for item in par_featureless))
                                temp_pivot_df[pre_label] = temp_pivot_df.apply(
                                    lambda row: ', '.join(str(row[item]) for item in par_featureless), axis=1)
                                st.write('')
                                plot_ans = st.radio('Do you want to plot the weight percentage of the interesting features?',
                                                    ('No.',
                                                        'Yes.'))
                                if plot_ans == 'No.':
                                    for column in pivot_df.columns:
                                        if column not in par_featureless:
                                            weight_column_name = f'{column}_weight(%)'
                                            s_s_col_name = f'{column}_sample_size_by_weight'
                                            column_feature = column
                                            pivot_df_2[column_feature] = pivot_df[column]
                                            pivot_df_2[weight_column_name] = np.round(
                                                (pivot_df[column] / N) * 100, 4)
                                            pivot_df_2[s_s_col_name] = np.round(
                                                (pivot_df[column] / N) * n)
                                if plot_ans == 'Yes.':
                                    st.caption(
                                        'You can download the figures by using the right click method.')
                                    for column in pivot_df.columns:
                                        if column not in par_featureless:
                                            weight_column_name = f'{column}_weight(%)'
                                            s_s_col_name = f'{column}_sample_size_by_weight'
                                            column_feature = column
                                            pivot_df_2[column_feature] = pivot_df[column]
                                            pivot_df_2[weight_column_name] = np.round(
                                                (pivot_df[column] / N) * 100, 4)
                                            pivot_df_2[s_s_col_name] = np.round(
                                                (pivot_df[column] / N) * n)
                                            labels = temp_pivot_df[pre_label]
                                            weights = pivot_df_2[weight_column_name]
                                            color_blue = '#2E6EF7'  # niq_blue
                                            color_orange = '#F05E19'  # niq_orange
                                            fig, ax = plt.subplots()
                                            ax.bar(
                                                labels, weights, color=color_blue)
                                            ax.set_title(column)
                                            ax.set_xlabel(
                                                ', '.join(str(item) for item in par_featureless))
                                            ax.set_ylabel(
                                                'Weight(%)')
                                            ax.tick_params(
                                                axis='x', labelsize=5)
                                            ax.tick_params(
                                                axis='y', labelsize=8)
                                            plt.xticks(rotation=90)
                                            st.pyplot(fig)

                        st.write(
                            'Weighted Dataframe pivot given the selected structure and feature column:')
                        st.write(pivot_df_2)
                        pivot_featured_df_csv = pivot_df_2.to_csv(
                            index=False)
                        coldos_est_feat_1, coldos_est_feat_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_feat_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=pivot_featured_df_csv,
                                               file_name=f'PIVOT_FEATURE_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')

                    pivot_df = grouped_df
                    w_pivot_df = pivot_df.copy()
                    for column in pivot_df.columns:
                        if column == 'Count':
                            weight_column_name = f'Weight(%)'
                            s_s_col_name = f'Sample_size_by_weight'
                            column_feature = column
                            w_pivot_df[column_feature] = pivot_df[column]
                            w_pivot_df[weight_column_name] = np.round(
                                (pivot_df[column] / N) * 100, 4)
                            w_pivot_df[s_s_col_name] = np.round(
                                (pivot_df[column] / N) * n)
                    st.write(
                        'Weighted Dataframe pivot given the selected structure:')
                    st.write(w_pivot_df)
                    pivot_structure_sampled_df_csv = w_pivot_df.to_csv(
                        index=False)
                    coldos_pivot_1, coldos_pivot_2 = st.columns(
                        2, gap='medium')

                    with coldos_pivot_2:
                        st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                           data=pivot_structure_sampled_df_csv,
                                           file_name=f'PIVOT_STRUCTURE_{file_name_df}.csv',
                                           mime='text/csv')
                    pre_est_samp_df = pd.DataFrame(columns=o_df_cols)
                    o_l_df = o_df.copy()
                    st.write('')
                    for _, row in w_pivot_df.iterrows():
                        temp_indices_list = []
                        n_temp = int(row['Sample_size_by_weight'])
                        filt_w_pivot_df = o_l_df.loc[(
                            o_l_df[par_col_list] == row[par_col_list]).all(axis=1)]
                        incomplete_est_samp_rows = filt_w_pivot_df.sample(
                            n=n_temp)
                        pre_est_samp_df = pd.concat(
                            [pre_est_samp_df, incomplete_est_samp_rows])
                        temp_indices_list.extend(
                            incomplete_est_samp_rows.index.tolist())
                        o_l_df = o_l_df.drop(temp_indices_list)
                    o_merge_pre_est_df = pd.merge(
                        o_df, pre_est_samp_df, how='left', indicator=True)
                    o_minus_pre_est_df = o_merge_pre_est_df[o_merge_pre_est_df['_merge'] == 'left_only'].drop(
                        columns='_merge')
                    a_auxiliar_df = w_pivot_df['Sample_size_by_weight'].value_counts(
                    ).sort_index().reset_index()
                    a_auxiliar_df.columns = [
                        'Sample_size_by_weight', 'Count']
                    d_auxiliar_df = a_auxiliar_df.sort_values(
                        'Sample_size_by_weight', ascending=False)
                    aux_eval = pre_est_samp_df.shape[0] - n
                    if aux_eval < 0:
                        sub_a_auxiliar = pd.DataFrame(
                            columns=a_auxiliar_df.columns)
                        temp_a_auxiliar_count = 0
                        for _, row in a_auxiliar_df.iterrows():
                            count = row['Count']
                            temp_a_auxiliar_count = temp_a_auxiliar_count + count
                            sub_a_auxiliar = pd.concat(
                                [sub_a_auxiliar, row.to_frame().T])
                            if temp_a_auxiliar_count >= abs(aux_eval):
                                break
                        sub_a_auxiliar = sub_a_auxiliar.reset_index(
                            drop=True)
                        values_a_filter = sub_a_auxiliar['Sample_size_by_weight'].tolist(
                        )
                        a_filt_w_pivot = w_pivot_df[w_pivot_df['Sample_size_by_weight'].isin(
                            values_a_filter)].sample(n=abs(aux_eval))
                        to_fill_a_w_pivot = pd.DataFrame()
                        for _, row in a_filt_w_pivot.iterrows():
                            temp_fill = o_minus_pre_est_df.loc[(
                                o_minus_pre_est_df[par_col_list] == row[par_col_list]).all(axis=1)].sample(n=1)
                            to_fill_a_w_pivot = pd.concat(
                                [to_fill_a_w_pivot, temp_fill])
                        est_samp_df = pd.concat(
                            [pre_est_samp_df, to_fill_a_w_pivot])
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        structure_sampled_l_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_l_1, coldos_est_l_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_l_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_l_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')
                    if aux_eval > 0:
                        sub_d_auxiliar = pd.DataFrame(
                            columns=d_auxiliar_df.columns)
                        temp_d_auxiliar_count = 0
                        for _, row in d_auxiliar_df.iterrows():
                            count = row['Count']
                            temp_d_auxiliar_count = temp_d_auxiliar_count + count
                            sub_d_auxiliar = pd.concat(
                                [sub_d_auxiliar, row.to_frame().T])
                            if temp_d_auxiliar_count >= abs(aux_eval):
                                break
                        sub_d_auxiliar = sub_d_auxiliar.reset_index(
                            drop=True)
                        values_d_filter = sub_d_auxiliar['Sample_size_by_weight'].tolist(
                        )
                        d_filt_w_pivot = w_pivot_df[w_pivot_df['Sample_size_by_weight'].isin(
                            values_d_filter)].sample(n=abs(aux_eval))
                        to_rem_d_w_pivot = pd.DataFrame()
                        for _, row in d_filt_w_pivot.iterrows():
                            temp_rem = pre_est_samp_df.loc[(
                                pre_est_samp_df[par_col_list] == row[par_col_list]).all(axis=1)].sample(n=1)
                            to_rem_d_w_pivot = pd.concat(
                                [to_rem_d_w_pivot, temp_rem])
                        pre_merge_to_rem = pd.merge(
                            pre_est_samp_df, to_rem_d_w_pivot, how='left', indicator=True)
                        est_samp_df = pre_merge_to_rem[pre_merge_to_rem['_merge'] == 'left_only'].drop(
                            columns='_merge')
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        structure_sampled_g_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_g_1, coldos_est_g_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_g_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_g_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
                                               mime='text/csv')
                    if aux_eval == 0:
                        est_samp_df = pre_est_samp_df
                        st.write(
                            'Sampled Dataframe given the selected structure:')
                        st.write(est_samp_df)
                        structure_sampled_e_df_csv = est_samp_df.to_csv(
                            index=False)
                        coldos_est_e_1, coldos_est_e_2 = st.columns(
                            2, gap='medium')

                        with coldos_est_e_2:
                            st.download_button(label=':floppy_disk: Download Dataframe as CSV :floppy_disk:',
                                               data=structure_sampled_e_df_csv,
                                               file_name=f'SAMPLED_STRUCTURE_{file_name_df}.csv',
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
</br></a>Version 1.4.1-b.1.</p>
</div>
</div>
"""
st.write(ft, unsafe_allow_html=True)
