#Import packages

import pickle
#Streamlit
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import dvc.api
import lightgbm
from matplotlib.image import imread
import datetime
import plotly.graph_objects as go
import webbrowser
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 


st.set_page_config(layout="wide")

def load_data():


	# Chargement du modèle Lgbm Github repository	
	pickle_in = dvc.api.read('lgbm.pkl',
    repo='https://github.com/om300/projet7_livrable.git',
    mode='rb')
	lgbm=pickle.loads(pickle_in)

	#Chargement pour le Dashboard
	db_test=pd.read_csv('https://github.com/om300/projet7_livrable/blob/0783ef223e233d254c8a31017814229f276d9014/data_app.csv?raw=true')
	
	#Creation d'une variable Age 
	db_test['YEARS_BIRTH']=(db_test['DAYS_BIRTH']/-365).apply(lambda x: int(x))
	
	#Reset index 
	db_test=db_test.reset_index(drop=True)
	
	#Charegement des donnees de test 
	df_test=pd.read_csv('https://github.com/om300/projet7_livrable/blob/0783ef223e233d254c8a31017814229f276d9014/data_test.csv?raw=true')
	
	#Chargement du logo pour le Dashboard
	logo=imread("https://github.com/om300/projet7_livrable/blob/0783ef223e233d254c8a31017814229f276d9014/Logo_OpenClassrooms.png?raw=true")


	#Calcul des SHAP values pour la partie explicabilite
	#importation du modele pré-entrainé lgbm
	explainer = shap.TreeExplainer(lgbm)
	
	#Appliquer sur le jeux de test
	shap_values = explainer.shap_values(df_test)[1]
	
	#Afficher le resultat
	exp_value=explainer.expected_value[1]
	
	return db_test,df_test,shap_values,lgbm,exp_value,logo


   # ##################################################
    # Interactive Dashboard
    # ##################################################

# read csv from a github repo
#df = pd.read_csv("https://github.com/om300/projet7_livrable/blob/ab7c5ee3522df5de9ad4c88afbf28b02acb93b07/data.csv?raw=true")

#st.markdown("<h1 style='text-align: center; color: green;'>Projet 7: Openclassrooms</h1>", unsafe_allow_html=True)

#st.markdown("<h1 style='text-align: center; color: grey;'>Implémentez un modèle de scoring</h1>", unsafe_allow_html=True)
# dashboard title

#st.subheader('Real-Time / Live Data Science Dashboard')

# top-level filters 

#NAME_CONTRACT_TYPE_filter = st.selectbox("Select the NAME CONTRACT TYPE", pd.unique(df['NAME_CONTRACT_TYPE']))

# creating a single-element container.
#placeholder = st.empty()

# dataframe filter 

#df = df[df['NAME_CONTRACT_TYPE']==NAME_CONTRACT_TYPE_filter]


# near real-time / live feed simulation 

'''for seconds in range(2306):
#while True: 
    
    df['Age'] = df['Age'] 
    df['balance_new'] = df['AMT_CREDIT'] 

    # creating KPIs 
    avg_age = np.mean(df['Age']) 

    count_married = int(df[(df["Married"]=='Yes')]['Married'].count())
    
    balance = np.mean(df['AMT_CREDIT'])
    
    AMT_INCOME_TOTAL = np.mean(df['AMT_INCOME_TOTAL'])
    
    AMT_GOODS_PRICE = np.mean(df['AMT_GOODS_PRICE'])
    
    #AMT_ANNUITY: Loan annuity / regular EMI to the bank
    AMT_ANNUITY = np.mean(df['AMT_ANNUITY'])


    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3 ,kpi4, kpi5, kpi6 = st.columns(6)

        # fill in those three columns with respective metrics or KPIs 
      # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Age ⏳", value=round(avg_age))
        kpi2.metric(label="Married Count 💍", value= int(count_married))
        kpi3.metric(label="Amount of credit ＄", value= f"$ {round(balance,2)} ")
        kpi4.metric(label="Amount Income ＄", value= int(AMT_INCOME_TOTAL))
        #For consumer loans it is the price of the goods for which the loan is given 
        kpi5.metric(label="AMT_GOODS_PRICE ＄", value= int(AMT_GOODS_PRICE))
        #AMT_ANNUITY: Loan annuity / regular EMI to the bank
        kpi6.metric(label="AMT_ANNUITY ⏳", value= int(AMT_ANNUITY))


        
        # create two columns for charts 

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.density_heatmap(data_frame=df, y = 'Age', x = 'Married')
            st.write(fig)
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame = df, x = 'Age')
            st.write(fig2)
        #st.markdown("### Detailed Data View")
        #st.dataframe(df)
        #time.sleep(20)
    #placeholder.empty()'''

 

    # ##################################################
    # API
    # ##################################################

#quand un identifiant correct a été saisi on appelle l'API

    #Appel de l'API : 

   # '''API_url = "http://127.0.0.1:5000/" + id_input

    #with st.spinner('Chargement du score du client...'):
        #json_url = urlopen(API_url)

        #API_data = json.loads(json_url.read())
        #classe_predite = API_data['prediction']
        #if classe_predite == 1:
            #etat = 'client à risque'
        #else:
            #etat = 'client peu risqué'
        #proba = 1-API_data['proba'] 

        #affichage de la prédiction
        #prediction = API_data['proba']
        #classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
        #classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        #chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')''''



   # ##################################################
    # Interest Rate 
    # ##################################################


#

#^IRX: 13 Week treasury Bill,
#^FVX : Treasury Yield 5 Year,
#^TNX : Treasury Yield 10 Year,
#^TYX : Treasury Yield 30 Year
#^IQHGCPI : CPI Inflation Tracker Index
#st.title('Finance Dashboard')

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: green;'>Projet 7: Openclassrooms</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: grey;'>Implémentez un modèle de scoring</h1>", unsafe_allow_html=True)

st.subheader("Taux d'intérêt : Taux des bons du Trésor américain")

tickers = ('^IRX','^FVX','^TNX','^TYX','^IQHGCPI')

dropdown = st.multiselect('Choisissez votre actif',
                         tickers)

start = st.date_input('début', value= pd.to_datetime('2018-01-01'))
end = st.date_input('fin', value= pd.to_datetime('today'))

def relativeret(df):
    rel = df.pct_change()
    cumret = (1+rel).cumprod() - 1
    cumret = cumret.fillna(0)
    return cumret


if len(dropdown) > 0:
    #df = yf.download(dropdown,start,end)['Adj Close']
    df = relativeret(yf.download(dropdown,start,end)['Adj Close'])
    st.header('Returns of {}'.format(dropdown))
    st.line_chart(df)


link='U.S. CONSUMER PRICE INDEX, YEAR-OVER-YEAR CHANGE SINCE 1970 ,check out this [link](https://flo.uri.sh/visualisation/6180194/embed?auto=1)'
st.markdown(link,unsafe_allow_html=True)

link2='Impact of the Interest rates charged on your credit ,check out this [link](https://www.cnbc.com/2022/03/17/how-to-start-paying-down-your-debt-now.html)'
st.markdown(link2,unsafe_allow_html=True)




    # ##################################################
    # Tableau clientèle
    # ##################################################

def tab_client(db_test):


	st.subheader('Tableau clientèle')
	row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row_spacer5 = st.columns([1,.1,1,.1,1,.1,4])

	# filtres via selectbox pour selectionner
	#Code_Gender ,afficher le sexe du client sélectionner
	sex=row0_1.selectbox("Sexe",['All']+db_test['CODE_GENDER'].unique().tolist())
	
	#Age du client selon les disponibilité dans la base de donnée
	age=row0_1.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))

	#Statut familian (Married ,not Married etc...)
	fam=row0_2.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
	
	#Nombre d'enfant 
	child=row0_2.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
	
	#Statut professionnel
	pro=row0_3.selectbox("Statut pro.",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
	
	#Niveau d'études
	stud=row0_3.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())

	#Affichage des variable selon les filtres définis
	db_display=db_test[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
	
	db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
	
	db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
	
	db_display['AMT_INCOME_TOTAL']=db_test['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
	
	db_display['AMT_CREDIT']=db_test['AMT_CREDIT'].apply(lambda x: int(x))
	
	db_display['AMT_ANNUITY']=db_test['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

	db_display=filter(db_display,'CODE_GENDER',sex)
	db_display=filter(db_display,'YEARS_BIRTH',age)
	db_display=filter(db_display,'NAME_FAMILY_STATUS',fam)
	db_display=filter(db_display,'CNT_CHILDREN',child)
	db_display=filter(db_display,'NAME_INCOME_TYPE',pro)
	db_display=filter(db_display,'NAME_EDUCATION_TYPE',stud)

	st.dataframe(db_display)
	st.markdown("**Total clients correspondants: **"+str(len(db_display)))

def filter(df,col,value):
	'''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
	if value!='All':
		db_filtered=df.loc[df[col]==value]
	else:
		db_filtered=df
	return db_filtered



    # ##################################################
    # Score visualisation
    # ##################################################

#Onglet Score visualisation
def score_viz(lgbm,df_test,client,exp_value,shap_values):
	st.subheader('Visualisation score')

	score,result=prediction(lgbm,df_test,client)

	fig = go.Figure(go.Indicator(
	mode = "gauge+number+delta",
    value = score,
    number = {'font':{'size':50}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': result.tolist(), 'font': {'size': 28, 'color':color(result)}},
    delta = {'reference': 0.50, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
    gauge = {
        'axis': {'range': [0,1], 'tickcolor': color(result)},
        'bar': {'color': color(result)},
        'steps': [
            {'range': [0,0.50], 'color': 'lightgreen'},
            {'range': [0.50,1], 'color': 'lightcoral'}],
        'threshold': {
            'line': {'color': "black", 'width': 5},
            'thickness': 1,
            'value': 0.50}}))


	st.plotly_chart(fig)

	st_shap(shap.force_plot(exp_value, shap_values[client], features = df_test.iloc[client], feature_names=df_test.columns, figsize=(12,5)))
#Fonction permettant de prédire la capacité du client à rembourser son emprunt,les paramètres sont le modèle, le dataframe et l'ID du client
def prediction(model,df_test,id):
	y_pred=model.predict_proba(df_test)[id,1]
	decision=np.where(y_pred>0.50,"Rejected","Approved")
	return y_pred,decision

#Définition de la couleur selon la prédiction
def color(pred):
	if pred=='Approved':
		col='Green'
	else :
		col='Red'
	return col

#Fonction permettant l'affichage de graphique shap values
def st_shap(plot, height=None):
	"""Fonction permettant l'affichage de graphique shap values"""
	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
	components.html(shap_html, height=height)

#Sélection d'un client via une selectbox
def get_client(db_test):
	client=st.sidebar.selectbox('Client',db_test['SK_ID_CURR'])
	idx_client=db_test.index[db_test['SK_ID_CURR']==client][0]
	return client,idx_client

#Affichage des infos du client sélectionné dans la barre latérale
def infos_client(db_test,client,idx_client):
	st.sidebar.markdown("**ID client: **"+str(client))
	st.sidebar.markdown("**Sexe: **"+db_test.loc[idx_client,'CODE_GENDER'])
	st.sidebar.markdown("**Statut familial: **"+db_test.loc[idx_client,'NAME_FAMILY_STATUS'])
	st.sidebar.markdown("**Enfants: **"+str(db_test.loc[idx_client,'CNT_CHILDREN']))
	st.sidebar.markdown("**Age: **"+str(db_test.loc[idx_client,'YEARS_BIRTH']))	
	st.sidebar.markdown("**Statut pro.: **"+db_test.loc[idx_client,'NAME_INCOME_TYPE'])
	st.sidebar.markdown("**Niveau d'études: **"+db_test.loc[idx_client,'NAME_EDUCATION_TYPE'])



    # ##################################################
    # Comparaison clientèle
    # ##################################################

#Fonction principale de l'onglet 'Comparaison clientèle
def comparaison(df_test,db_test,idx_client):
	st.subheader('Comparaison clientèle')
	idx_neigh,total=get_neigh(df_test,idx_client)
	db_neigh=db_test.loc[idx_neigh,['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','TARGET']]
	db_neigh['AMT_INCOME_TOTAL']=db_neigh['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
	db_neigh['AMT_CREDIT']=db_neigh['AMT_CREDIT'].apply(lambda x: int(x))
	db_neigh['AMT_ANNUITY']=db_neigh['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

	if total:
		display_charts(db_test,idx_client)

	else:
		display_charts(db_neigh,idx_client)


#Calcul des voisins les plus proches du client sélectionner ,Sélection du nombre de voisins par un slider.Retourne les proches voisins et un booléen indiquant la clientèle globale ou non
def get_neigh(df,idx_client):
	row1,row_spacer1,row2,row_spacer2 = st.columns([1,.1,.3,3])
	size=row1.slider("Taille du groupe de comparaison",min_value=10,max_value=1000,value=500)
	row2.write('')
	total=row2.button(label="Clientèle globale")
	neigh= NearestNeighbors(n_neighbors=size)
	neigh.fit(df)
	k_neigh=neigh.kneighbors(df.loc[idx_client].values.reshape(1,-1),return_distance=False)[0]
	k_neigh=np.sort(k_neigh)
	return k_neigh,total

#Affichae des graphes de comparaison pour le client sélectionné
def display_charts(df,client):
	row1_1,row1_2,row1_3 = st.columns(3)
	st.write('')
	row2_10,row2_2,row2_3 = st.columns(3)
	
	chart_kde("Répartition de l'age",row1_1,df,'YEARS_BIRTH',client)
	chart_kde("Répartition des revenus",row1_2,df,'AMT_INCOME_TOTAL',client)
	chart_bar("Répartition du nombre d'enfants",row1_3,df,'CNT_CHILDREN',client)

	chart_bar("Répartition du statut professionel",row2_10,df,'NAME_INCOME_TYPE',client)
	chart_bar("Répartition du niveau d'études",row2_2,df,'NAME_EDUCATION_TYPE',client)
	chart_bar("Répartition du type de logement",row2_3,df,'NAME_HOUSING_TYPE',client)
	st.dataframe(df[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])

#Définition des graphes KDE avec une ligne verticale indiquant la position du client
def chart_kde(title,row,df,col,client):
	with row:
		st.subheader(title)
		fig,ax = plt.subplots()
		sns.kdeplot(df.loc[df['TARGET']==0,col],color='green', label = 'Target == 0')
		sns.kdeplot(df.loc[df['TARGET']==1,col],color='red', label = 'Target == 1')
		plt.axvline(x=df.loc[client,col],ymax=0.95,color='black')
		plt.legend()
		st.pyplot(fig)

#Définition des graphes barres avec une ligne horizontale indiquant la position du client
def chart_bar(title,row,df,col,client):
	with row:
		st.subheader(title)
		fig,ax = plt.subplots()
		data=df[['TARGET',col]]
		if data[col].dtypes!='object':
			data[col]=data[col].astype('str')

			data1=round(data[col].loc[data['TARGET']==1].value_counts()/data[col].loc[data['TARGET']==1].value_counts().sum()*100,2)
			data0=round(data[col].loc[data['TARGET']==0].value_counts()/data[col].loc[data['TARGET']==0].value_counts().sum()*100,2)
			data=pd.concat([pd.DataFrame({"Pourcentage":data0,'TARGET':0}),pd.DataFrame({'Pourcentage':data1,'TARGET':1})]).reset_index().rename(columns={'index':col})
			sns.barplot(data=data,x='Pourcentage', y=col, hue='TARGET', palette=['green','red'], order=sorted(data[col].unique()));
			
			data[col]=data[col].astype('int64')

			plt.axhline(y=sorted(data[col].unique()).index(df.loc[client,col]),xmax=0.95,color='black',linewidth=4)
			st.pyplot(fig)
		else:

			data1=round(data[col].loc[data['TARGET']==1].value_counts()/data[col].loc[data['TARGET']==1].value_counts().sum()*100,2)
			data0=round(data[col].loc[data['TARGET']==0].value_counts()/data[col].loc[data['TARGET']==0].value_counts().sum()*100,2)
			data=pd.concat([pd.DataFrame({"Pourcentage":data0,'TARGET':0}),pd.DataFrame({'Pourcentage':data1,'TARGET':1})]).reset_index().rename(columns={'index':col})
			sns.barplot(data=data,x='Pourcentage', y=col, hue='TARGET', palette=['green','red'], order=sorted(data[col].unique()));
			
			plt.axhline(y=sorted(data[col].unique()).index(df.loc[client,col]),xmax=0.95,color='black',linewidth=4)
			st.pyplot(fig)

#Fonction principale permettant l'affichage de la fenêtre latérale avec les 5 onglets
def main():
	db_test,df_test,shap_values,lgbm,exp_value,logo=load_data()
	st.sidebar.image(logo)
	Headers = ["Taux d'intérêt : Taux des bons du Trésor américain / Inflation",
	    "Tableau clientèle",
	    "Visualisation score",
	    "Comparaison clientèle"
	]
	
	st.sidebar.write('')
	st.sidebar.write('')

	st.sidebar.title('SOMMAIRE')
	selection = st.sidebar.radio("Go to", Headers)

	if selection=="Tableau clientèle":
		tab_client(db_test)
	if selection=="Visualisation score":
		client,idx_client=get_client(db_test)
		infos_client(db_test,client,idx_client)
		score_viz(lgbm,df_test,idx_client,exp_value,shap_values)
	if selection=="Comparaison clientèle":
		client,idx_client=get_client(db_test)
		infos_client(db_test,client,idx_client)
		comparaison(df_test,db_test,idx_client)

if __name__ == '__main__':
	main()



