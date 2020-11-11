from flask import Flask, render_template
import pandas as pd
import nflfastpy as nfl
import numpy as np

app = Flask(__name__)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

@app.route('/')
def index():
    df = pd.read_csv('simulations.csv').iloc[:, 1:]
    df = df.rename(columns={
        'receiver_player_name': 'Receiver',
        'posteam': 'Team',
        'fpts': 'FP'
    })

    team_logo_data = nfl.load_team_logo_data().rename(columns={
        'team_abbr': 'Team',
        'team_logo_wikipedia': 'Logo'
    })[['Team', 'Logo']]

    df = df.merge(team_logo_data, on='Team')
    df['Team'] = df['Logo'].apply(lambda x: f'<img src="{x}" width="60px">')
    
    headshots = nfl.load_roster_data()[['teamPlayers.gsisId', 'teamPlayers.headshot_url', 'team.season']]
    
    headshots = headshots.loc[headshots['team.season'] == 2019, ['teamPlayers.gsisId', 'teamPlayers.headshot_url']].rename(columns={
        'teamPlayers.headshot_url': 'headshot_url',
        'teamPlayers.gsisId': 'gsis_id'
    })

    df = df.merge(headshots, on='gsis_id', how='left')

    def format_headshot(row):
        headshot = row['headshot_url']
        receiver = row['Receiver']
        if headshot is np.nan:
            headshot = 'https://sportsfly.cbsistatic.com/fly-959/bundles/sportsmediacss/images/player/headshot-default.png'
        return f'<img src={headshot} width="50px">' + '<br>' + f'<p style="font-weight: bold">{receiver}</p>'

    df['Receiver'] = df.apply(format_headshot, axis=1)

    df = df.drop(columns=['gsis_id', 'Logo'])
    df = df.sort_values(by='FP', ascending=False)

    rank_columns = ['#1 WR', 'Top 5 WR', 'Top 10 WR', 'Top 25 WR', 'Top 50 WR']
    formatters = {
        'FP': '{:.2f}'
    }

    for col in rank_columns:
        df[col] = df[col] * 100
        formatters[col] = '{:.2f}%'
    df = df[['Receiver', 'Team', 'FP'] + rank_columns]
    df = df.style.hide_index().format(formatters).render()

    return render_template('index.html', df=df)

if __name__ == '__main__':
    app.run(debug=True)