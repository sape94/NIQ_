import pandas as pd


class DataFrameReplacer:

    def __init__(self, fulldf, fracdf, sort_col: str = 'ACV'):
        o_df = fulldf.copy()
        s_df = fracdf.copy()
        col_n = list(o_df.columns.values)
        original_rows = o_df.index.values.tolist()
        used_rows = s_df.index.values.tolist()
        unused_rows = [item for item in original_rows
                       if item not in used_rows]
        ws_df = o_df.filter(items=unused_rows, axis=0)
        if sort_col == '':
            ws_df = ws_df
        else:
            ws_df = ws_df.sort_values(by=sort_col, ascending=False)

        self.fulldf = fulldf
        self.fracdf = fracdf
        self.o_df = o_df
        self.s_df = s_df
        self.col_n = col_n
        self.ws_df = ws_df

    def rmv_sts(self, est_col: list, rmv_list: list,
                id_rmv: str = 'SHO_ID'):
        rmv_s_df = self.s_df.copy()
        in_s_df = self.s_df.copy()
        rmv_s_df = rmv_s_df[rmv_s_df[id_rmv].isin(rmv_list)]
        in_s_df = in_s_df[~in_s_df[id_rmv].isin(rmv_list)]
        stc_rmv_df = rmv_s_df[est_col]
        stc_dic = stc_rmv_df.to_dict('list')

        self.est_col = est_col
        self.rmv_s_df = rmv_s_df
        self.in_s_df = in_s_df
        self.stc_rmv_df = stc_rmv_df
        self.stc_dic = stc_dic

        return self.in_s_df

    def add_sts(self, est_col: list, rmv_list: list,
                id_rmv: str = 'SHO_ID'):
        rmv_s_df = self.s_df.copy()
        in_s_df = self.s_df.copy()
        rmv_s_df = rmv_s_df[rmv_s_df[id_rmv].isin(rmv_list)]
        in_s_df = in_s_df[~in_s_df[id_rmv].isin(rmv_list)]
        stc_rmv_df = rmv_s_df[est_col]
        stc_dic = stc_rmv_df.to_dict('list')

        self.est_col = est_col
        self.rmv_s_df = rmv_s_df
        self.in_s_df = in_s_df
        self.stc_rmv_df = stc_rmv_df
        self.stc_dic = stc_dic

    #    return self.in_s_df

    # def add_sts(self):
        add_df = pd.DataFrame(columns=self.col_n)
        l = len(self.est_col)

        for row in range(0, self.rmv_s_df.shape[0]):
            temp_df = self.ws_df.copy()
            for i in range(l):
                crit_a = temp_df[self.est_col[i]]
                crit_b = self.stc_dic[self.est_col[i]][row]
                temp_df = temp_df[crit_a == crit_b]
                if temp_df.empty:
                    break
            if temp_df.empty:
                ta_df = self.ws_df.iloc[0]
            else:
                ta_df = temp_df.iloc[0]
            add_df = pd.concat([add_df, ta_df.to_frame().transpose()])
        n_s_df = pd.concat([self.in_s_df, add_df])

        self.add_df = add_df
        self.n_s_df = n_s_df

        return self.n_s_df
