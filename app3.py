import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

###################################################
##     initieren variabelen in sessionstate    ####
###################################################

if 'leeftijd' not in st.session_state: # toevoegen leeftijd aan sessionstate als deze nog niet is ingevult
    st.session_state.leeftijd = 0

if 'planning_df' not in st.session_state: # toevoegen aan planning aan sessionstate als deze nog niet is ingevult
    st.session_state.planning_df = pd.DataFrame({'Week': range(1, 6),
                     'Hr-zone': [0]*5,
                     'Stappen doel': [0]*5
                     })

########################
##     functies     ####
########################

##### functie om de ruis uit de ecg te filteren #####
@st.cache_data
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    # nyquist is een waarde die de helft van de meetfrequentie neemt, dit is wat accuraat gemeten kan worden
    nyq = 0.5 * fs 
    # high en lowcut omzetten in verhouding tot nyquist
    low = lowcut / nyq
    high = highcut / nyq 
    # met de functie butter met btype band wordt de bandpass filter toegepast, dit behoudt alleen de fq tussen de high en lowcut
    # de order geeft aan hoe scherp de overgang van cutoff naar doorgelaten fq is
    b, a = butter(order, [low, high], btype='band') 
    return filtfilt(b, a, data)

##### lowpass filter om EDR te krijgen #####
@st.cache_data
def EDR_filter(data, cutoff, fs, order=3):
    # nyquist is een waarde die de helft van de meetfrequentie neemt, dit is wat accuraat gemeten kan worden
    nyq = 0.5 * fs
    # de cutoff maken voor de filter
    normal_cutoff = cutoff / nyq
    # in deze functie wordt btype 'low' gebruikt om de lowpass filter toe te passen. dit filter de hogere frequenties en behoudt de lagere
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

##### piek detectie functie #####
@st.cache_data
def pieken_detecteren_dynamisch(df, naam_column, drempel_percentage, interval, minimale_drempel):
    pieken_index = [] #lijst voor index pieken
    
    # for loop die aan de hand van het interval de pieken detecteerd,
    # dit zorgt ervoor dat zelfs ruis die niet uit de data gefilterd is niet zorgt voor problemen met detectie
    for start_idx in range(0, len(df), interval): # door df loopen met interval, en start index geven
        end_idx = min(start_idx + interval, len(df)) # eind index vinden door start idx +interval
        segment = df.iloc[start_idx:end_idx] #sefment selecteren

        # Dynamische drempel op basis van max, maar met een minimale ondergrens
        drempelwaarde = max(segment[naam_column].max() * drempel_percentage, minimale_drempel)
        
        # pieken detecteren aan de hand van drempelwaarde
        pieken = find_peaks(segment[naam_column], height=drempelwaarde, distance=10)[0]
        pieken_index.extend(segment.index[pieken])
    
    # df teruggeven met de pieken 
    return df.iloc[pieken_index]

##### functie voor rmssd voor simulatie bpm #####
@st.cache_data
def simulate_rmssd(bpm_series):
    rr_intervals = 60 / bpm_series
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 100  
    return rmssd

##### simulatie bpm hrv voor trendlijn #####
@st.cache_data
def bpm_hrv_sim():
    # Simuleer timestamps per minuut voor 30 dagen
    n_dagen = 60
    n_per_day = 24 * 60  # aantal minuten per dag
    timestamps = pd.date_range(start="2025-05-01", periods=n_dagen * n_per_day, freq="T")

    # Simuleer BPM: basislijn met schommelingen
    np.random.seed(42)
    bpm = 70 + 8 * np.sin(np.linspace(0, 10 * np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))

    # Zet in DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'bpm': bpm})
    df['dag'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W')
    df['maand'] = df['timestamp'].dt.to_period('M')

    # Bereken daggemiddelden + hrv per dag
    dag_gem = df.groupby('dag').agg(
        bpm_gem=('bpm', 'mean'),
        rmssd=('bpm', lambda x: simulate_rmssd(x.values))
    ).reset_index()

    # Idem voor weken
    week_gem = df.groupby('week').agg(
        bpm_gem=('bpm', 'mean'),
        rmssd=('bpm', lambda x: simulate_rmssd(x.values))
    ).reset_index()
    week_gem['week'] = week_gem['week'].astype(str)
    
    # Idem voor maanden
    maand_gem = df.groupby('maand').agg(
        bpm_gem=('bpm', 'mean'),
        rmssd=('bpm', lambda x: simulate_rmssd(x.values))
    ).reset_index()
    maand_gem['maand'] = maand_gem['maand'].astype(str)
    
    return dag_gem, week_gem, maand_gem


##### data preperatie #####
@st.cache_data
def prep_data(df):
    df['timestamp'] = (df['time'] - df['time'].iloc[0]) /1000 #tijd normaliseren
    
    
    df = df.rename(columns={
        ' ECG': 'sample',
        ' accX': 'x',
        ' accY': 'y',
        ' accZ': 'z'
        })
    
    # tijd groeperen per minuut, dit kan later ook per uur of per dag
    df["minuut"] = (df["timestamp"] // 60).astype(int)
    
    ##### ECG prep #####
    df['filtered'] = bandpass_filter(df['sample'], 5, 49, fs=100)
    
    rpieken_drempel = 0.5 
    interval = 250
    rminimum = 0.2
    df_ecg_pieken = pieken_detecteren_dynamisch(df,'filtered', rpieken_drempel,interval, rminimum)

    
    df_ecg_pieken['tijd_tussen_pieken'] =  df_ecg_pieken['timestamp'].diff() # tijd tussen pieken berekenen
    df_ecg_pieken['bpm'] = 60 / df_ecg_pieken['tijd_tussen_pieken'] # bpm berekenen
    df_ecg_pieken['gemiddelde_tijd'] = df_ecg_pieken['timestamp'].rolling(window=2).mean() # gemiddele tijd zodat het klopt tov bpm

    # max hartslag berekenen en daarop filteren
    df_ecg_pieken = df_ecg_pieken[(df_ecg_pieken['bpm'] < (220 - st.session_state.leeftijd) ) ] 

    
    df_ecg_pieken['bpm_vloeiend'] = df_ecg_pieken['bpm'].rolling(window=7,min_periods=2).mean() # hartslag filteren om grafiek pretiger te maken
    df_ecg_pieken['timestamp_vloeiend'] = df_ecg_pieken['gemiddelde_tijd'].rolling(window=7,min_periods=2).mean() # tijd op zelfde manier filteren
    
    ##### acc #####
    
    #Samengesteld signaal maken
    df['samengesteld_signaal'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    # signaal filteren aan de hand van bandpass om ruis weg te halen
    df['samengesteld_filtered'] = bandpass_filter(df['samengesteld_signaal'], lowcut=0.5, highcut=5, fs=100) 

    # Data normaliseren voor stappenteller
    df['genormaliseerd'] = df['samengesteld_filtered']-df['samengesteld_filtered'].mean()


    # stappen detecteren
    interval = 250
    stappenteller_drempel = 0.5
    min_drempel=0.4
    df_stappen = pieken_detecteren_dynamisch(df, 'genormaliseerd', stappenteller_drempel, interval, min_drempel)

    # Cumulatieve stappenteller
    df['is_een_stap'] = df.index.isin(df_stappen.index)
    df['stappenteller'] = df['is_een_stap'].cumsum()
    
    # df met stappen per minuut
    df_spm = df.groupby("minuut")["is_een_stap"].sum().reset_index()

    # Zet de index naar de minuten voor bar chart
    df_spm = df_spm.set_index("minuut")
    df_spm['stap_per_min'] = df_spm['is_een_stap'].cumsum()
    
    ##### edr #####
    # EDR wordt opgezet aan de hand van ecg filteren zodat alleen waarden onder 0.5hz worden gezien
    EDR = EDR_filter(df['sample'], 0.5, fs=100)
    df['EDR'] = EDR - EDR.mean() #EDR normaliseren
    return df, df_ecg_pieken, df_spm

############################
##     sidebar maken    ####
############################

with st.sidebar:
    st.title('Opties')
    knop = st.toggle('uitgebreide mode') # knop voor uitgebreide mode
    with st.popover('Instellingen planning'): # popover waar zorgverlener planning en gezondheidswaardes kan invoeren
        #wachtwoordbeveiliging voor popover zodat alleen de zorgverlener dit kan toevoegen
        wachtwoord = st.text_input("Voer wachtwoord in", 
                                   type="password", 
                                   key="wachtwoord_popover"
                                   )
        correct_wachtwoord = "geheim123"
        
        if wachtwoord == correct_wachtwoord:
            st.title("Persoonlijke Trainingsplanning")
            # leeftijd input
            st.session_state.leeftijd = st.number_input('Leeftijd client invullen',
                                       min_value=1,
                                       max_value=120,
                                       value= 20,
                                       step=1
                                       )
            # input duur traject
            weken = st.number_input("Hoeveel weken wil je plannen?", 
                                    min_value=1, 
                                    max_value=52, 
                                    value=5, 
                                    step=1
                                    )
            
            planning_maken = pd.DataFrame({'Week': range(1, weken + 1),
                             'Hr-zone': [-999]*weken,
                             'Stappen doel': [-999]*weken
                             })
    
            st.write("Vul hieronder je hr-zone en stappendoel per week in:")
            # input max hr zone en stappen doel voor elke week
            editable_df = st.data_editor(planning_maken, num_rows="fixed")
            
            # opslaan planning
            if st.button("Planning opslaan"):
                st.success("Planning opgeslagen!")
                
                # Sla de planning op in session_state
                st.session_state["planning_df"] = editable_df
        else:
            st.error("Onjuist wachtwoord.") # error als ww onjuist is
    rangeselectie = st.segmented_control('Range selectie', 
                                         ['Dag','Week','Maand'], 
                                         selection_mode = 'single', 
                                         default='Dag')

#################################################
##     data inladen en waarden berekenen     ####
#################################################
# bestand inladen
df_data = pd.read_csv('data_meting3.txt', delimiter=',' )

# data prep functie initiatie
df_ecg_acc, df_ecg_pieken_es, stappen_per_minuut = prep_data(df_data)

# bpm vinden voor live bpm count
laatste_bpm = round(df_ecg_pieken_es["bpm_vloeiend"].iloc[-1])
delta_bpm = round(df_ecg_pieken_es["bpm_vloeiend"].iloc[-2] - df_ecg_pieken_es["bpm_vloeiend"].iloc[-1])

#hr zones berekenen
max_hr_es = 220 - st.session_state.leeftijd

zones = pd.DataFrame({'Zone': [0.50 * max_hr_es,
                               0.60 * max_hr_es,
                               0.70 * max_hr_es,
                               0.80 * max_hr_es,
                               0.90 * max_hr_es,
                               max_hr_es],
                      'Zone naam': ["Zone 1",
                                    "zone 2",
                                    "Zone 3",
                                    "Zone 4",
                                    "Zone 5",
                                    "Max"] })

# tot stappen voor live stappen teller
tot_stappen=df_ecg_acc['stappenteller'].max()

# rmssd voor hrv meter
rmssd = np.sqrt(np.mean(np.square(np.diff(df_ecg_pieken_es['tijd_tussen_pieken'])))) * 100

dag_trend, week_trend, maand_trend = bpm_hrv_sim()

#############################
##     figuren maken     ####
#############################

##### bpm figuur #####
fig_bpm = go.Figure()

# Lijn voor vloeiende hartslag
fig_bpm.add_trace(go.Scatter(
    x=df_ecg_pieken_es['timestamp_vloeiend'],
    y=df_ecg_pieken_es['bpm_vloeiend'],
    name='Hartslag over tijd',
    line=dict(color='crimson'),
    hovertemplate='BPM: %{y}<extra></extra>'
))

# Zones als horizontale lijnen
for i in range(len(zones)):
    fig_bpm.add_hline(
        y=zones['Zone'][i],
        line_dash='dash',
        opacity=0.7,
        annotation_text=zones['Zone naam'][i],
        annotation_position='top left'
    )

# Layout aanpassen
fig_bpm.update_layout(
    title='Visualisatie van hartslag over tijd',
    xaxis_title='Tijd (sec)',
    yaxis_title='BPM',
    yaxis=dict(range=[-1, 220])
)

##### ecg figuur #####
fig_ecg = go.Figure()

# lijn ecg toevoegen
fig_ecg.add_trace(go.Scatter(
    x=df_ecg_acc['timestamp'],
    y=df_ecg_acc['filtered'],
    name='ECG'
    ))

# rode punten op de pieken
fig_ecg.add_trace(go.Scatter(
    x=df_ecg_pieken_es['timestamp'],
    y=df_ecg_pieken_es['filtered'],
    mode='markers',
    marker=dict(color='crimson', size=6),
    name='R-piek detectie'
    ))

# layout aanpassen met slider
fig_ecg.update_layout(
    xaxis_title="Tijd (sec)",
    yaxis=dict(visible=False),
    xaxis=dict(
        rangeslider=dict(visible=True),
        range=[df_ecg_acc['timestamp'].max()-5, df_ecg_acc['timestamp'].max()],
        type="linear"),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.1,
        xanchor='center',
        x=0.5    
    ))


##### stappen figuur #####
fig_stappen = go.Figure()

# barchart maken
fig_stappen.add_trace(go.Bar(
    x = stappen_per_minuut.index,
    y = stappen_per_minuut['is_een_stap'],
    name = 'stappen per minuut',
    hovertemplate='stappen: %{y}<extra></extra>'
    ))

# als uitgebreide mode aan staat
if knop == True:
    # lijn met cumultatieve stappenteller
    fig_stappen.add_trace(go.Scatter(
        x = stappen_per_minuut.index,
        y = stappen_per_minuut['stap_per_min'],
        name = 'Totaal aantal stappen',
        marker=dict(color='orange'),
        hovertemplate='stappen: %{y}<extra></extra>'
        ))
    
    # layout aanpassen
    fig_stappen.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.1,
            xanchor='center',
            x=0.5    
        ),
        xaxis_title='Tijd (min)',
        yaxis_title='stappen'
    )

# als uitgebreide mode uit staat
else:
    # Lyout aanpassen
    fig_stappen.update_layout(
        title='Visualisatie van stappen over tijd',
        xaxis_title='Tijd (min)',
        yaxis_title='stappen',
        yaxis=dict(range=[-1,stappen_per_minuut['is_een_stap'].max() +10 ])
    )

##### EDR figuur #####
fig_edr = go.Figure()

# EDR grafiek
fig_edr.add_trace(go.Scatter(
    x=df_ecg_acc['timestamp'],
    y=df_ecg_acc['EDR'],
    name='EDR'
    ))

# layout aanpassen met slider
fig_edr.update_layout(
    title = 'Ademhalingsgrafiek (EDR)',
    xaxis_title="Tijd (sec)",
    yaxis=dict(visible=False),
    xaxis=dict(
        rangeslider=dict(visible=True),
        range=[df_ecg_acc['timestamp'].max()-60, df_ecg_acc['timestamp'].max()],
        type="linear"
    ))

##### HRV figuur #####
fig2 = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=rmssd,
    delta={'reference': 50, 'increasing': {'color': "red"}},
    title= {'text': "HRV", 'font': {'size': 20}},
    gauge={
        'axis': {'range': [0, 100],
        'tickvals': [0, 25, 50, 75],  # Ticks op overgangspunten
        'ticktext': ["0", "25", "50", "75"]},
        # 'bar': {'color': "crimson"},
        'steps': [
            {'range': [0, 25], 'color': "red"},
            {'range': [25, 50], 'color': "orange"}, 
            {'range': [50, 75], 'color': "lightgreen"}, 
            {'range': [75, 100], 'color': "yellow"} 
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': 50
        }
    }
))

##### bpm trend figuren #####
#dag
fig_dag_bpm =  go.Figure()

fig_dag_bpm.add_trace(go.Scatter(
    x=dag_trend['dag'],
    y=dag_trend['bpm_gem']
    ))

fig_dag_bpm.add_trace(go.Scatter(
    x=dag_trend['dag'],
    y=dag_trend['bpm_gem'],
    mode='markers',
    marker=dict(color='crimson', size=3)
    ))

fig_dag_bpm.update_layout(
    title='BPM trend per dag',
    xaxis_title='Dag',
    yaxis_title='BPM',
    yaxis=dict(range=[-1, 220]),
    height=350,
    showlegend = False
    )

#week
fig_week_bpm =  go.Figure()

fig_week_bpm.add_trace(go.Scatter(
    x=week_trend['week'],
    y=week_trend['bpm_gem']
    ))

fig_week_bpm.add_trace(go.Scatter(
    x=week_trend['week'],
    y=week_trend['bpm_gem'],
    mode='markers',
    marker=dict(color='crimson', size=3)
    ))

fig_week_bpm.update_layout(
    title='BPM trend per week',
    xaxis_title='Week',
    yaxis_title='BPM',
    yaxis=dict(range=[-1, 220]),
    height=350,
    showlegend = False
    )

#maand
fig_maand_bpm =  go.Figure()

fig_maand_bpm.add_trace(go.Scatter(
    x=maand_trend['maand'],
    y=maand_trend['bpm_gem']
    ))

fig_maand_bpm.add_trace(go.Scatter(
    x=maand_trend['maand'],
    y=maand_trend['bpm_gem'],
    mode='markers',
    marker=dict(color='crimson', size=3)
    ))

fig_maand_bpm.update_layout(
    title='BPM trend per maand',
    xaxis_title='Maand',
    yaxis_title='BPM',
    yaxis=dict(range=[-1, 220]),
    height=350,
    showlegend = False
    )

##### HRV trend figuren
#dag
fig_dag_hrv = go.Figure()

fig_dag_hrv.add_trace(go.Scatter(
    x=dag_trend['dag'],
    y=dag_trend['rmssd']))

fig_dag_hrv.add_trace(go.Scatter(
    x=dag_trend['dag'],
    y=dag_trend['rmssd'],
    mode='markers',
    marker=dict(color='green', size=3)
    ))

fig_dag_hrv.update_layout(
    title='HRV trend per dag',
    xaxis_title='Dag',
    yaxis_title='HRV',
    yaxis=dict(range=[-1, 100]),
    height=250,
    showlegend = False
    )

#week
fig_week_hrv = go.Figure()

fig_week_hrv.add_trace(go.Scatter(
    x=week_trend['week'],
    y=week_trend['rmssd']))

fig_week_hrv.add_trace(go.Scatter(
    x=week_trend['week'],
    y=week_trend['rmssd'],
    mode='markers',
    marker=dict(color='green', size=3)
    ))

fig_week_hrv.update_layout(
    title='HRV trend per week',
    xaxis_title='Week',
    yaxis_title='HRV',
    yaxis=dict(range=[-1, 100]),
    height=250,
    showlegend = False
    )

#maand
fig_maand_hrv = go.Figure()

fig_maand_hrv.add_trace(go.Scatter(
    x=maand_trend['maand'],
    y=maand_trend['rmssd']))

fig_maand_hrv.add_trace(go.Scatter(
    x=maand_trend['maand'],
    y=maand_trend['rmssd'],
    mode='markers',
    marker=dict(color='green', size=3)
    ))

fig_maand_hrv.update_layout(
    title='HRV trend per maand',
    xaxis_title='Maand',
    yaxis_title='HRV',
    yaxis=dict(range=[-1, 100]),
    height=250,
    showlegend = False
    )

############################
##     indeling app     ####
############################

##### colomen voor de header van de app #####
cola,colb,colc = st.columns([3,1,1]) # geeft 3:1:1 verhouding voor indeling kolommen
with cola:
    st.header('Gezondheidswaarden') 

with colb:
    keuze = st.selectbox('Week', st.session_state['planning_df']['Week']) # weekselectie voor planning
    
with colc:
    subcol1, subcol2 = st.columns([2, 1])
    with subcol2:
        # alarmknop toevoegen
        if st.button("🔔",type='primary', use_container_width=True):
            subcol1.header("ALARM")

##### tabladen toevoegen  #####
tab1, tab2 = st.tabs(['Dashboard', 'Help'])

##### waardes uit weekselectie van planning halen #####
week_index = st.session_state['planning_df']['Week'] == keuze #omzetten van keuze naar indx
stappen_doel = st.session_state['planning_df'].loc[week_index, 'Stappen doel'].values[0] # stappen doel
nog_tot_doel = stappen_doel - tot_stappen
# teksten maken voor tekst bij stappenteller
if nog_tot_doel>0:
    delta_stap = f"nog {nog_tot_doel} stappen tot je doel"
else:
    delta_stap = f"{nog_tot_doel} stappen over je doel heen"

hr_zone_doel = st.session_state['planning_df'].loc[week_index, 'Hr-zone'].values[0] # HR zone selecite
zone_max = zones.loc[hr_zone_doel,'Zone'].astype(int) # max hartslagzone vinden

##### tablat 1 #####
with tab1:
    # opdelen in 3 rijen
    row1 = st.container()
    row2 = st.container()
    row3 = st.container()
    
    with row1:
        col1, col2 = st.columns(2)
        
        with col1:
            cold, cole = st.columns([1,2])
            with cold:
                st.header("Hartslag")
            with cole:
                st.write('') #witregels
                st.write('')
                st.write(f'Streefwaarde: onder {zone_max} BPM') #hr zone doel neerzeten
            # live BPM
            st.metric(label='', 
                      value=f"{laatste_bpm} BPM (slagen per minuut)", 
                      delta=f"{delta_bpm} BPM", 
                      delta_color="off"
                      )
            st.plotly_chart(fig_bpm) # bpm figuur
        
        
        with col2:
            colf, colg = st.columns([1,1])
            with colf: 
                st.header("Stappenteller")
            with colg:
                st.write('') # witregels
                st.write('')
                st.write(f'Doel: {stappen_doel} stappen') # stappendoel neerzetten
            # live stappenteller
            st.metric(label='',
                      value=f'{tot_stappen} stappen',
                      delta = delta_stap,
                      delta_color='off'
                      )
            
            st.plotly_chart(fig_stappen) # stappen figuur
        
        
            
    with row2:
        col3, col4 = st.columns(2)
        
        with col3:
            st.title('ECG')
            st.write('Een ECG meet de elektrische activiteit van het hart en laat zien hoe het ritme en de hartslagen in de tijd verlopen.')
            st.plotly_chart(fig_ecg) #ecg figuur
            

        with col4:
            st.title('Planning')
            st.write('Dit is de planning die voor u is opgesteld.')
            st.dataframe(st.session_state["planning_df"], use_container_width=True) # planning
    
    # als uitgebreide mode aan staat kan de EDR en HRV gezien worden
    if knop == True:
        with row3:
            col5, col6 = st.columns(2)
            
            with col5:
                st.title('Ademhalingsgrafiek')
                st.write('Via kleine variaties in het ECG-signaal kan ook het ademhalingspatroon worden afgeleid, zonder aparte sensor.')
                st.info('Let op, dit werkt het best als u stil zit')
                st.plotly_chart(fig_edr) #EDR figuur

            with col6:
                st.title('HRV (Hartritmevariabiliteit)')
                st.write('HRV geeft de variatie in tijd tussen hartslagen weer en is een maat voor balans tussen inspanning en herstel.')
                st.plotly_chart(fig2) #hrv figuur
    st.info('Als u meer wilt weten over de gezondheidswaarden kijk dan bij de kennisclips op de help pagina')

##### tablat 2 #####
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    # kennisclips
    with col1:
        st.header("Kennisclips")
        st.text('Wil je meer weten over hartslag?')
        st.video("https://www.youtube.com/watch?v=WZI-fYAYzZw")
    
    with col2:
        st.header('')
        st.text('Wil je meer weten over hartslagzones?')
        st.video("https://www.youtube.com/watch?v=Mh5oNMDCucQ&pp=ygUNaGFydHNsYWd6b25lcw%3D%3D")
            
    with col3:
        st.header('')
        st.text('Ademhalingsoefening')
        st.video("https://youtu.be/5RDG99jFKBY?si=SO7cFeu6rSWg1lp4")
    
    if knop == True:
        with col4:
            st.header('')
            st.text('Wil je meer weten over HRV?')
            st.video("https://youtu.be/zUyuUoU7lAQ?si=3AzaetMN8u7gqaX5&t=7")
    
    # chatbot
    st.title('Chatbot')
    prompt = st.chat_input("Typ hier je vraag")
    if prompt:
        st.write(f"Gebruiker: {prompt}")
            
##### sidebar trend figuren #####
with st.sidebar:
    # BPM figuren
    if rangeselectie == 'Dag':
        st.plotly_chart(fig_dag_bpm)
    elif rangeselectie == 'Week':
        st.plotly_chart(fig_week_bpm)
    else:
        st.plotly_chart(fig_maand_bpm)
        
    # HRV figuren als knop aan staat
    if knop == True:
        if rangeselectie == 'Dag':
            st.plotly_chart(fig_dag_hrv)
        elif rangeselectie == 'Week':
            st.plotly_chart(fig_week_hrv)
        else:
            st.plotly_chart(fig_maand_hrv)
            
            
