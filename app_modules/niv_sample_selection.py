import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
from pandas.api.types import is_numeric_dtype as pind
warnings.filterwarnings("ignore")


class Universe_Structure:

    def __init__(self, data: pd.DataFrame):

        self.data = data

        self.necessary_columns = {'SHO_EXTERNAL_CODE': 'External store code.',
                                  'SHO_ID': 'Store ID.',
                                  'ACV': 'Store ACV.',
                                  'Player_ID': 'Nielsen ID for the commercial group or main retailer.',
                                  'Player': 'Name of the commercial group or main retailer.',
                                  'Subplayer_ID': 'Nielsen ID for the members of the commercial group, \n'
                                                  'chains or formats offered by the main retailer.',
                                  'Subplayer': 'Name of the members of the commercial group, \n'
                                               'chains or formats offered by the main retailer.',
                                  'City_ID': 'Nielsen Code for the city.',
                                  'City': 'Name of the city.',
                                  'State_ID': 'Nielsen Code for the state.',
                                  'State': 'Name of the state.'}

        self.player_info_cols = ['Player_ID',
                                 'Player', 'Subplayer_ID', 'Subplayer']
        self.state_info_cols = ['State_ID', 'State']
        self.city_info_cols = ['City_ID', 'City']
        self.full_geo_info_cols = self.state_info_cols + self.city_info_cols
        self.detailed_info_cols = self.full_geo_info_cols + self.player_info_cols

        self.case_settings = {'retailer': self.player_info_cols,
                              'state': self.state_info_cols,
                              'city': self.city_info_cols,
                              'detailed': self.detailed_info_cols}

    def players_help(self):

        print(
            'A player can group multiple retail chains or format. \n'
            'In the example bellow, Player 1 groups two subplayers, \n'
            'while Player 2 is its own subplayer. \n')
        players_definition = [
            [1, 'Player 1', 101, 'Subplayer 1'],
            [1, 'Player 1', 102, 'Subplayer 2'],
            [2, 'Player 2', 2, 'Player 2']
        ]
        players_headers = ['Player_ID', 'Player', 'Subplayer_ID', 'Subplayer']

        return print(tabulate(players_definition, headers=players_headers))

    def columns_check(self):

        data_cols = self.data.columns.unique().tolist()
        missing_cols = [
            req_col for req_col in self.necessary_columns.keys() if req_col not in data_cols]
        n_missing = len(missing_cols)
        player_info_missing = set(self.player_info_cols) & set(missing_cols)

        if (n_missing != 0):

            if (player_info_missing):

                print(
                    f""" The following {n_missing} columns are either missing from your dataframe, or are not properly labeled. \n""")
                for missing_col in missing_cols:
                    print(
                        f' -- {missing_col}: {self.necessary_columns[missing_col]}\n')
                print(
                    "For more information about players and subplayers, type 'players_help()'.")

                return False

            else:

                print(
                    f""" The following {n_missing} columns are either missing from your data frame, or not properly labeled. \n""")

                for missing_col in missing_cols:

                    print(
                        f' -- {missing_col}: {self.necessary_columns[missing_col]}\n')

                return False
        else:

            if pind(self.data.ACV):
                return True
            else:
                try:
                    self.data['ACV'] = self.data.ACV.astype(int)
                    return True
                except ValueError as err:
                    print('The ACV column of the dataframe should be numeric. \n'
                          f'Unexpected {err=}.')
                    return False
                except Exception as err:
                    print(f'Unexpected {err=}, {type(err)=}')
                    return False

    def acv_structure(self, structure_case: str, universe_acv: int):

        weight_col_name = f'{structure_case.title()} ACV Weight (%)'
        cumsum_col_name = f'{structure_case.title()} ACV Cumm Sum (%)'

        group_cols = self.case_settings[structure_case]
        reduced_cols = group_cols + ['ACV']
        summary_cols = reduced_cols + [weight_col_name, cumsum_col_name]

        structure_df = self.data.copy()
        structure_df = structure_df[reduced_cols]
        acv_gpd_stc = structure_df.groupby(group_cols).sum()
        acv_gpd_stc.reset_index(inplace=True)

        acv_gpd_stc[weight_col_name] = np.round(
            (acv_gpd_stc['ACV'] / universe_acv) * 100, 2)
        acv_gpd_stc.sort_values(
            weight_col_name, ascending=False, inplace=True)
        acv_gpd_stc[cumsum_col_name] = np.round(
            (acv_gpd_stc['ACV'].cumsum() / universe_acv) * 100, 2)

        acv_gpd_stc.reset_index(drop=True, inplace=True)
        acv_gpd_stc = acv_gpd_stc[summary_cols]

        return acv_gpd_stc

    def sample_structure(self, structure_case: str, universe_acv: int, universe_n: int):

        acv = self.acv_structure(structure_case, universe_acv)

        acv_weight_col_name = f'{structure_case.title()} ACV Weight (%)'
        acv_cumsum_col_name = f'{structure_case.title()} ACV Cumm Sum (%)'

        n_weight_col_name = f'{structure_case.title()} Stores Weight (%)'
        n_cumsum_col_name = f'{structure_case.title()} Stores Cumm Sum (%)'

        group_cols = self.case_settings[structure_case]
        reduced_cols = group_cols + ['SHO_ID']
        summary_cols = reduced_cols + \
            ['ACV', acv_weight_col_name, acv_cumsum_col_name,
                n_weight_col_name, n_cumsum_col_name]

        structure_df = self.data.copy()
        structure_df = structure_df[reduced_cols]
        n_gpd_stc = structure_df.groupby(group_cols).count()
        n_gpd_stc.reset_index(inplace=True)

        n_gpd_stc[n_weight_col_name] = np.round(
            (n_gpd_stc['SHO_ID'] / universe_n) * 100, 2)

        gpd_stc = pd.merge(acv, n_gpd_stc, on=group_cols)

        gpd_stc[n_cumsum_col_name] = np.round(
            (gpd_stc['SHO_ID'].cumsum() / universe_n) * 100, 2)

        gpd_stc.reset_index(drop=True, inplace=True)
        gpd_stc = gpd_stc[summary_cols]
        gpd_stc.rename(columns={'SHO_ID': 'Store_Count'}, inplace=True)

        return gpd_stc

    def get_structure(self):

        data_check = self.columns_check()

        if data_check:

            self.universe_acv = self.data.ACV.sum()
            self.universe_n = self.data.SHO_ID.count()

            self.retail_structure = self.sample_structure(
                'retailer', self.universe_acv, self.universe_n)
            self.state_structure = self.sample_structure(
                'state', self.universe_acv, self.universe_n)
            self.city_structure = self.sample_structure(
                'city', self.universe_acv, self.universe_n)
            self.detailed_structure = self.sample_structure(
                'detailed', self.universe_acv, self.universe_n)

            structure_df_ls = [self.retail_structure,
                               self.state_structure,
                               self.city_structure,
                               self.detailed_structure]

            return structure_df_ls

        else:
            pass


class NIV_Structure_Design:

    def __init__(self, data: pd.DataFrame, parameter_acv, parameter_stores, structure, reduction, cities_weight):
        # self.parameter = parameter
        self.parameter_acv = parameter_acv
        self.parameter_stores = parameter_stores
        self.structure = structure
        self.reduction = reduction
        self.cities_weight = cities_weight

        self.data = data

        _ustc = Universe_Structure(self.data)
        self.ustc_df_ls = _ustc.get_structure()

        self.retail_stc = _ustc.retail_structure
        self.state_stc = _ustc.state_structure
        self.city_stc = _ustc.city_structure

    def get_closest(self, values_ls: list, constant_value: int):
        def abs_diff_func(list_value): return abs(constant_value - list_value)
        closest_value = min(values_ls, key=abs_diff_func)
        return closest_value

    def set_target_parameters(self, criteria: str, parameter):
        self.criteria = criteria
        # parameter = input(
        #    f'Please type the {criteria} % that the Sample will target.')
        self.parameter = parameter
        # while True:
        #    try:
        #        if (0 <= float(parameter) <= 1):
        #            break
        #        else:
        #            print(
        #                'The introduced parameter should be a numeric value between 0 and 1.')
        #            parameter = input(
        #                f'Please type the {criteria} % that the Sample will target.')
        #    except:
        #        print(
        #            'The introduced parameter should be a numeric value between 0 and 1.')
        #        parameter = input(
        #            f'Please type the {criteria} % that the Sample will target.')

        return float(parameter)

    def set_structure_preservation(self, structure):
        structure_preservation_modes = ['cities', 'universe']
        # structure = str(
        #    input(
        #        """
        #        Please type the structure ('cities' or 'universe') you would like the sample to preserve. \n
        #        For more information type 'help'.
        #        """
        #    )
        # )
        structure = self.structure
        while True:
            if structure.lower() in structure_preservation_modes:
                break
            # elif structure.lower() == 'help':
            #    print("\n -- Cities: Preserving structure by cities will try to maintain the weight of each chain, store and ACV-wise, by city.")
            #    print("   -- Universe: Preserving structure by universe will try to maintain the weight of each chain, store and ACV-wise, on the provided universe.\n")
            #    structure = str(
            #        input(
            #            """
            #        Please type the structure ('cities' or 'universe') you would like the sample to preserve. \n
            #        For more information type 'help'.
            #        """
            #        )
            #    )
            # else:
            #    print(
            #        """The requested criteria for the selection has no yet been implemented.""")
            #    structure = str(
            #        input(
            #            """
            #        Please type the structure ('cities' or 'universe') you would like the sample to preserve. \n
            #        For more information type 'help'.
            #        """
            #        )
            #    )
        return structure

    def set_reduction_method(self, reduction):
        reduction_modes = ['acv', 'stores']
        # reduction = str(
        #    input(
        #        """
        #        How would you like to select the most relevant cities in the universe? ('ACV' or 'Stores') \n
        #        """
        #    )
        # )
        reduction = self.reduction
        while True:
            if reduction.lower() in reduction_modes:
                break
            # else:
            #    print(
            #        """The requested criteria for the selection has no yet been implemented.""")
            #    reduction = str(
            #        input(
            #            """
            #            How would you like to select the most relevant cities in the universe? ('ACV' or 'Stores') \n
            #            """
            #        )
            #    )

        if reduction.lower() == 'acv':
            reduction_title = 'ACV'
        elif reduction.lower() == 'stores':
            reduction_title = 'Stores'

        return reduction_title

    def set_principal_cities(self, criteria, cities_weight):
        self.criteria = criteria
        # cities_weight = input(
        #    f'Please type the {criteria} % that the principal cities will cover.')
        cities_weight = self.cities_weight
        # while True:
        #    try:
        #        if (0 <= float(cities_weight) <= 1):
        #            break
        #        else:
        #            print(
        #                'The introduced parameter should be a numeric value between 0 and 1.')
        #            cities_weight = input(
        #                f'Please type the {criteria} % that the principal cities will cover.')
        #    except:
        #        print(
        #            'The introduced parameter should be a numeric value between 0 and 1.')
        #        cities_weight = input(
        #            f'Please type the {criteria} % that the principal cities will cover.')
        return float(cities_weight)

    def get_principal_cities(self, reduction, cities_weight):
        self.reduction = reduction
        self.cities_weight = cities_weight

        self.criteria_title = self.set_reduction_method(
            reduction=self.reduction)
        self.cities_weight = self.set_principal_cities(
            self.criteria_title, cities_weight=self.cities_weight)

        city_rep = self.city_stc[f'City {self.criteria_title} Cumm Sum (%)'].unique(
        ).tolist()
        closest_value = self.get_closest(
            city_rep, self.cities_weight * 100)
        selected_cities_df = self.city_stc[self.city_stc[
            f'City {self.criteria_title} Cumm Sum (%)'] <= closest_value]
        selected_cities_ls = selected_cities_df.City.unique().tolist()

        print(
            f'{closest_value}% of the provided sample\'s {self.criteria_title} is concentrated in the cities:')

        for city in selected_cities_ls:
            if "_" in city:
                proper_city_name = city.replace('_', ' ').title()
            else:
                proper_city_name = city.title()

            print(f'  -- {proper_city_name}', sep="\n")

        return selected_cities_df

    def target_parameters_df(self, parameter_acv, parameter_stores, structure, reduction, cities_weight):
        # self.parameter = parameter
        self.parameter_acv = parameter_acv
        self.parameter_stores = parameter_stores
        self.reduction = reduction
        self.cities_weight = cities_weight

        selected_cities_df = self.get_principal_cities(
            reduction=self.reduction, cities_weight=self.cities_weight)
        structure = self.set_structure_preservation(structure=self.structure)
        target_acv = self.set_target_parameters(
            criteria='ACV', parameter=self.parameter_acv)
        target_stores = self.set_target_parameters(
            criteria='Stores', parameter=self.parameter_stores)

        print("\n The current sample will:")
        print(f"    -- Preserve structure by {structure.lower()}.")
        print(f"    -- Target {np.round(target_acv * 100, 2)} % ACV.")
        print(f"    -- Target {np.round(target_stores * 100, 2)} % Stores.")

        store_universe = selected_cities_df['Store_Count'].sum()
        acv_universe = selected_cities_df['ACV'].sum()

        u_store_sample = np.round(store_universe * target_stores, 0)
        u_acv_sample = np.round(acv_universe * target_acv, 0)

        selected_cities = selected_cities_df['City_ID'].unique().tolist()

        cities_data = self.data.copy()
        cities_data = cities_data[cities_data.City_ID.isin(selected_cities)]

        _sustc = Universe_Structure(cities_data)
        ssustc_df_ls = _sustc.get_structure()
        city_sample_stc = _sustc.city_structure

        if structure.lower() == 'cities':
            city_sample_stc['Target Stores (City)'] = city_sample_stc['Store_Count'] * \
                target_stores
            city_sample_stc['Target ACV (City)'] = city_sample_stc['ACV'] * \
                target_acv
        else:
            city_sample_stc['Target Stores (City)'] = (
                city_sample_stc['City Stores Weight (%)'] / 100) * u_store_sample
            city_sample_stc['Target ACV (City)'] = (
                city_sample_stc['City ACV Weight (%)'] / 100) * u_acv_sample

        city_sample_stc = city_sample_stc.round(
            {'Target Stores (City)': 0, 'Target ACV (City)': 0})

        return city_sample_stc

    def new_sample_structure(self, parameter_acv, parameter_stores, structure, reduction, cities_weight):
        self.parameter_acv = parameter_acv
        self.parameter_stores = parameter_stores
        self.structure = structure
        self.reduction = reduction
        self.cities_weight = cities_weight

        selected_cities_df = self.target_parameters_df(
            parameter_acv=self.parameter_acv, parameter_stores=self.parameter_stores, structure=self.structure, reduction=self.reduction, cities_weight=self.cities_weight)
        selected_cities = selected_cities_df['City_ID'].unique().tolist()
        parameter_columns = ['City_ID',
                             'Target Stores (City)', 'Target ACV (City)']

        cities_parameters = selected_cities_df.copy()
        cities_parameters = cities_parameters[parameter_columns]

        cities_data = self.data.copy()
        cities_data = cities_data[cities_data.City_ID.isin(selected_cities)]

        _sustc = Universe_Structure(cities_data)
        ssustc_df_ls = _sustc.get_structure()
        detailed_sample_stc = _sustc.detailed_structure

        working_df = pd.merge(detailed_sample_stc, cities_parameters,
                              on='City_ID')
        working_df.drop(columns=['Detailed ACV Cumm Sum (%)',
                                 'Detailed Stores Cumm Sum (%)'],
                        inplace=True)

        city_df_ls = [working_df[working_df.City_ID == city]
                      for city in working_df.City_ID.unique().tolist()]

        for df in city_df_ls:
            df['City ACV Weight (Chains)'] = (
                df['ACV'] / df['ACV'].sum()) * 100
            df['City Stores Weight (Chains)'] = (
                df['Store_Count'] / df['Store_Count'].sum()) * 100
            df['Target ACV (Chains)'] = (
                df['City ACV Weight (Chains)'] / 100) * df['Target ACV (City)']
            df['Target Stores (Chains)'] = (
                df['City Stores Weight (Chains)'] / 100) * df['Target Stores (City)']

        cities_df = pd.concat(city_df_ls)

        cities_df = cities_df.round(
            {'Target Stores (Chains)': 0, 'Target ACV (Chains)': 0})

        summary_cols = ['State_ID', 'State', 'City_ID', 'City',
                        'Player_ID', 'Player', 'Subplayer_ID', 'Subplayer',
                        'Target Stores (Chains)', 'Target ACV (Chains)']

        cities_df = cities_df[summary_cols]

        return cities_df


class NIV_Sample_Selection:

    def __init__(self, data: pd.DataFrame, parameter_acv, parameter_stores, structure, reduction, cities_weight):
        # self.parameter = parameter
        self.parameter_acv = parameter_acv
        self.parameter_stores = parameter_stores
        self.structure = structure
        self.reduction = reduction
        self.cities_weight = cities_weight
        # def __init__(self, data: pd.DataFrame):

        self.complete_data = data

        _nstd = NIV_Structure_Design(
            self.complete_data, parameter_acv=self.parameter_acv, parameter_stores=self.parameter_stores, structure=self.structure, reduction=self.reduction, cities_weight=self.cities_weight)
        self.new_structure = _nstd.new_sample_structure(
            parameter_acv=self.parameter_acv, parameter_stores=self.parameter_stores, structure=self.structure, reduction=self.reduction, cities_weight=self.cities_weight)
        self.cities = self.new_structure.City.unique().tolist()
        self.cities_id = self.new_structure.City_ID.unique().tolist()
        self.n_stores = self.new_structure['Target Stores (Chains)'].sum()

        self.cities_data_stc = self.new_structure[self.new_structure.City_ID.isin(
            self.cities_id)]
        self.cities_data = self.complete_data[self.complete_data.City_ID.isin(
            self.cities_id)]

    def structure_preserving_sample(self):
        nsdf = self.cities_data.loc[self.cities_data['City_ID'].isin(
            self.cities_id)]
        subplayer_ls = []
        for ciudad, subdf in nsdf.groupby('City_ID'):
            for subplayer, sbpdf in subdf.groupby('Subplayer_ID'):
                n = int(
                    self.new_structure.loc[
                        (self.new_structure['City_ID'] == ciudad) &
                        (self.new_structure['Subplayer_ID'] == subplayer)
                    ]['Target Stores (Chains)'].values[0]
                )
                sbpdf = sbpdf.nlargest(n, 'ACV')
                subplayer_ls.append(sbpdf)
        structure_case_niv = pd.concat(subplayer_ls, ignore_index=True)
        return structure_case_niv

    def acv_maximizing_sample(self):
        nsdf = self.cities_data.loc[self.cities_data['City_ID'].isin(
            self.cities_id)]
        nsdf = nsdf.nlargest(int(self.n_stores), 'ACV')
        nsdf.reset_index(drop=True, inplace=True)
        return nsdf
