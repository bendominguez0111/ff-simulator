import pandas as pd
import nflfastpy as nfl
import numpy as np
import warnings
import time
from scipy.stats import gamma
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as BS
from sklearn.linear_model import LinearRegression
import requests

warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', None)

class Simulator:

    def __init__(self, n_simulations=100, stdout=False, week_num=10, year=2020):
        self.n_simulations = n_simulations
        self.week_num = week_num
        self.year = year
        self.stdout = stdout

    def grab_weekly_results(self):

        df = self.load_pbp_data()
        df = df.loc[df['week'].isin(range(1, self.week_num))]
        df['gsis_id'] = df['receiver_player_id'].apply(nfl.utils.convert_to_gsis_id)

        weekly_fpts = df.groupby(['gsis_id', 'receiver_player_name', 'posteam', 'game_id'], as_index=False)[['yards_gained', 'pass_touchdown', 'pass_attempt', 'complete_pass']].sum().assign(fpts = lambda x: x.yards_gained*0.1 + x.pass_touchdown*6 + x.complete_pass)

        roster_data = self.load_roster_data()
        injuries = self.grab_player_injuries()
        rookies = self.grab_rookies()

        injured_players = roster_data.merge(injuries, on=['Player', 'Pos'])[['Player', 'gsis_id']]

        #removing injured players
        weekly_fpts = weekly_fpts.loc[(~weekly_fpts['gsis_id'].isin(injured_players['gsis_id']))]
        # removing non-WRs
        weekly_fpts = weekly_fpts.loc[(weekly_fpts['gsis_id'].isin(roster_data['gsis_id'])) | (weekly_fpts['receiver_player_name'].isin(rookies))]

        return weekly_fpts

    def load_pbp_data(self):
        return nfl.load_pbp_data(self.year)

    @staticmethod
    def load_roster_data():
        roster_data = nfl.load_roster_data()
        roster_data = roster_data.loc[(roster_data['team.season'] == 2019) & (roster_data['teamPlayers.positionGroup'] == 'WR'), ['teamPlayers.displayName', 'teamPlayers.gsisId', 'teamPlayers.positionGroup']]
        roster_data = roster_data.rename(columns={'teamPlayers.displayName': 'Player', 'teamPlayers.positionGroup': 'Pos', 'teamPlayers.gsisId': 'gsis_id'})

        return roster_data

    @staticmethod
    def grab_player_injuries():
        url = 'https://www.pro-football-reference.com/players/injuries.htm'
        table = str(BS(requests.get(url).content, 'html.parser').find('table'))
        injuries = pd.read_html(table)[0]
        injuries = injuries.loc[injuries['Class'].isin(['I-R'])]
        return injuries

    @staticmethod
    def grab_rookies():
        rookies_url = 'https://www.pro-football-reference.com/years/2020/draft.htm'
        rookies = pd.read_html(str(BS(requests.get(rookies_url).content, 'html.parser').find('table')))[0]
        rookies.columns = rookies.columns.droplevel(level=0)
        rookies = rookies.loc[rookies['Pos'] == 'WR']
        rookies['Player'] = rookies['Player'].apply(lambda x: '.'.join(
            [x.split()[0][0], x.split()[1]]
        ))
        rookies = rookies['Player']
        return rookies

    def run_simulation(self):
        weekly_fpts = self.grab_weekly_results()
        player_id_table = weekly_fpts.groupby('gsis_id', as_index=False)[['receiver_player_name', 'posteam']].first()
        players = weekly_fpts['gsis_id'].unique()
        weeks_remaining = 17 - self.week_num

        #train a linear regression model
        X = weekly_fpts['pass_attempt'].values.reshape(-1, 1)
        y = weekly_fpts['fpts'].values

        lm = LinearRegression().fit(X, y)

        sim_df = pd.DataFrame({}, columns=['gsis_id', 'sim_num', 'fpts'])
        i = 1
        start = time.time()
        for simulation in range(1, self.n_simulations+1):
            if self.stdout and i % 25 == 0:
                print('running simulation #{i}'.format(i=i))
                print(f'time elapsed: {time.time() - start} seconds')
            i+=1
            for player_id in players:
                mean = np.random.choice(weekly_fpts.loc[weekly_fpts['gsis_id'] == player_id]['pass_attempt'], size=100, replace=True).mean()
                std = np.random.choice(weekly_fpts.loc[weekly_fpts['gsis_id'] == player_id]['pass_attempt'], size=100, replace=True).std()
                #gamma distribution model parameters
                shape = (mean/std)**2
                scale = (std**2)/mean
                #find a random value from a fitted gamma distribution
                targets = np.random.gamma(shape, scale, size=(1, weeks_remaining)).reshape(-1, 1)

                if np.isnan(targets).any():
                    continue

                fpts = lm.predict(targets).mean()

                row = pd.DataFrame({'gsis_id': [player_id], 'sim_num': [simulation],'fpts': [fpts]})
                sim_df = pd.concat([sim_df, row])

        sim_df = sim_df.dropna()

        sim_df = sim_df.merge(player_id_table, on='gsis_id')

        sims = [sim[-1] for sim in sim_df.groupby('sim_num')]

        final_sim_df = pd.DataFrame()

        for i, sim in enumerate(sims):
            sim[f'rank_sim_{i+1}'] = sim['fpts'].rank(ascending=False)
            sim = sim[['gsis_id', 'receiver_player_name', 'posteam', f'rank_sim_{i+1}']]
            if final_sim_df.empty:
                final_sim_df = sim
                continue
            else:
                final_sim_df = final_sim_df.merge(sim, on=['gsis_id', 'receiver_player_name', 'posteam'])

        data = {
            'gsis_id': [],
            'receiver_player_name': [],
            'posteam': [],
            '#1 WR': [],
            'Top 5 WR': [],
            'Top 10 WR': [],
            'Top 25 WR': [],
            'Top 50 WR': []
        }

        for _, row in final_sim_df.iterrows():
            data['gsis_id'].append(row['gsis_id'])
            data['receiver_player_name'].append(row['receiver_player_name'])
            data['posteam'].append(row['posteam'])
            ranks = [int(rank) for rank in row[3:].values.tolist()]

            number_one = 0
            top_5 = 0
            top_10 = 0
            top_25 = 0
            top_50 = 0

            for rank in ranks: 
                if rank == 1:
                    number_one += 1
                if rank <= 5:
                    top_5 += 1
                if rank <= 10:
                    top_10 += 1
                if rank <= 25:
                    top_25 += 1
                if rank <= 50:
                    top_50 += 1
            
            data['#1 WR'].append(number_one / self.n_simulations)
            data['Top 5 WR'].append(top_5 / self.n_simulations)
            data['Top 10 WR'].append(top_10 / self.n_simulations)
            data['Top 25 WR'].append(top_25 / self.n_simulations)
            data['Top 50 WR'].append(top_50 / self.n_simulations)

        percent_ranks_df = pd.DataFrame(data)
        sim_df = sim_df.groupby('gsis_id', as_index=False).mean().merge(percent_ranks_df, on='gsis_id').sort_values(by='fpts', ascending=False).head(50)

        return sim_df

simulator = Simulator(n_simulations=1000, stdout=True, week_num=10)

simulator.run_simulation().to_csv('simulations.csv', index=True)