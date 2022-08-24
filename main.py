from pydoc import plain
from typing import final
import pandas as pd
import numpy as np
import pickle
import streamlit as st

data = pd.read_csv("all_data_clean.csv")

player_names = data.Player.unique()

predictors = ['MPpg', 'PTSpg', 'FGpg', '3Ppg', 'FTpg', 'ORBpg', 'DRBpg', 'TRBpg',
       'ASTpg', 'STLpg', 'BLKpg', 'TOVpg', 'PFpg', 'Age', 'Year_index',
       'Team_index', 'FT', 'FT%', 'Games', 'MP', 'FG%']

@st.cache(allow_output_mutation = True)
def load_model():
    models = {}
    with open('rookie-models2.bin', 'rb') as f_in:
        models['lr'],models['lgb'],models['xgb'],models['cats'],models['ada'],models['gbm'],models['rf'] = pickle.load(f_in)
        return models

models = load_model()


show_cols = ['Player','Team','Year','Career'] + ['Games','MPpg', 'PTSpg','ASTpg','TRBpg']

def predict_single(name,models,predictors):
    df = data[data.Player == name]
    #display(df)
    #df['Year_index'] = df['Year'] - 1979
    #df['Team_index'] = df['Team'].map(team_mapper)
    final_predictions = (
         (models['lr'].predict_proba(df[predictors])[:,1]) +
        (models['lgb'].predict_proba(df[predictors])[:,1]) +
        (models['xgb'].predict_proba(df[predictors])[:,1]) +
        (models['cats'].predict_proba(df[predictors])[:,1])+ 
        (models['ada'].predict_proba(df[predictors])[:,1])+
         (models['gbm'].predict_proba(df[predictors])[:,1])+
         (models['rf'].predict_proba(df[predictors])[:,1])
    )/7
    final_preds = (final_predictions>=0.5)*1
    return df ,final_predictions,final_preds



def submit():
    st.title('NBA Rookie Career Greater than five years prediction')
    st.image("""rose.jpg""")
    st.header('Search Player Name')

    rookie = st.selectbox(
     'Player name :', player_names)

    df ,final_predictions,final_preds =predict_single(name=rookie,models=models,predictors=predictors)
    
    st.dataframe(df[show_cols])

    if st.button('Predict Career greater 5 years in the NBA'):
        #price = predict(carat, cut, color, clarity, depth, table, x, y, z)
        if final_predictions > 0.5:
            st.success(f'{rookie} is gonna last more than 5 years in the NBA')  
            st.success(f'Probability : {np.round(final_predictions,3)*100}%')
        else:
            st.error(f'{rookie} may not  last more than 5 years in the NBA')  
            st.error(f'Probability {np.round(final_predictions,3)*100}%')

if __name__ == '__main__':
    submit()