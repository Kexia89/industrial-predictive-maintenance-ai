"""Panel industrial de demostración; no controla maquinaria ni recibe sensores reales."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier
from features import RAW, transform
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Kexia | Industrial AI Monitor',page_icon='⚙️',layout='wide')
st.markdown('''<style>.block-container{max-width:1350px;padding-top:2rem} [data-testid="stMetric"]{background:#172636;border:1px solid #294052;border-radius:12px;padding:18px} .brand{letter-spacing:3px;color:#38BDF8;font-size:13px;font-weight:700} </style>''',unsafe_allow_html=True)
st.markdown('<div class="brand">KEXIA / INDUSTRIAL INTELLIGENCE</div>',unsafe_allow_html=True)
st.title('Industrial AI Monitor')
st.caption('XGBoost · Detección experimental de fallas · AI4I 2020')
@st.cache_resource
def load_model(model_path, modified):
    model=XGBClassifier(); model.load_model(model_path); return model
try:
    path=ROOT/'models/xgboost.json'; meta=json.loads((ROOT/'models/metadata.json').read_text()); model=load_model(str(path),path.stat().st_mtime_ns)
except (FileNotFoundError,ValueError) as e:
    st.error('No se encontró un modelo válido. Ejecuta python train_model.py primero.'); st.stop()
threshold=meta['threshold']
def predict(df):
    x=transform(df)[meta['feature_names']]; return model.predict_proba(x)[:,1]
def outside(df):
    return [c for c,(lo,hi) in meta['ranges'].items() if ((pd.to_numeric(df[c])<lo)|(pd.to_numeric(df[c])>hi)).any()]
st.sidebar.header('Centro de operaciones')
page=st.sidebar.radio('Módulo',['Evaluación manual','Análisis por lote','Rendimiento del modelo'])
st.sidebar.caption(f'Umbral operativo (validación): {threshold:.3f}')
st.sidebar.info('Prototipo con datos sintéticos. Sin conexión a sensores ni control de equipos.')
if page=='Evaluación manual':
    st.subheader('Lecturas operativas')
    with st.form('readings'):
        a,b,c=st.columns(3)
        with a:
            typ=st.selectbox('Tipo de producto',['L','M','H'],index=1)
            air=st.number_input('Temperatura ambiental (K)',min_value=0.,value=300.,step=.1)
        with b:
            process=st.number_input('Temperatura del proceso (K)',min_value=0.,value=310.2,step=.1)
            rpm=st.number_input('Velocidad (RPM)',min_value=0.,value=1450.,step=10.)
        with c:
            torque=st.number_input('Torque (Nm)',min_value=0.,value=54.8,step=.1)
            wear=st.number_input('Desgaste (min)',min_value=0.,value=218.,step=1.)
        submitted=st.form_submit_button('Evaluar condiciones',type='primary')
    if submitted or 'last' not in st.session_state:
        frame=pd.DataFrame([[typ,air,process,rpm,torque,wear]],columns=RAW)
        score=float(predict(frame)[0]); st.session_state.last=(frame,score)
        if submitted:
            history=st.session_state.get('history',[]); history.append({'Evaluación':len(history)+1,'Puntuación':score}); st.session_state.history=history
    frame,score=st.session_state.last
    cols=st.columns(4)
    for col,label,val in zip(cols,['Temperatura proceso','Velocidad','Torque','Desgaste'],[f'{frame.iloc[0][RAW[2]]:.1f} K',f'{frame.iloc[0][RAW[3]]:,.0f} RPM',f'{frame.iloc[0][RAW[4]]:.1f} Nm',f'{frame.iloc[0][RAW[5]]:.0f} min']): col.metric(label,val)
    flags=outside(frame)
    if flags: st.warning('Lecturas fuera del rango de entrenamiento: '+', '.join(flags))
    left,right=st.columns([1,1]); alert=score>=threshold
    with left:
        fig=go.Figure(go.Indicator(mode='gauge+number',value=score*100,number={'suffix':'%'},title={'text':'Puntuación de falla'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#F59E0B' if alert else '#38BDF8'},'steps':[{'range':[0,threshold*100],'color':'#183B3A'},{'range':[threshold*100,100],'color':'#49352A'}],'threshold':{'line':{'color':'#FFFFFF','width':3},'value':threshold*100}})); fig.update_layout(height=300,margin=dict(l=30,r=30,t=55,b=15),paper_bgcolor='rgba(0,0,0,0)',font_color='#E6EDF5'); st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader('Estado de la evaluación')
        if alert: st.warning('Alerta del modelo · revisar lecturas con personal cualificado.')
        else: st.success('Sin alerta según el umbral del modelo. No garantiza ausencia de falla.')
        st.metric('Potencia mecánica estimada',f"{transform(frame)['mechanical_power_w'].iloc[0]/1000:.2f} kW")
        st.caption('La puntuación no es una probabilidad calibrada ni una estimación de vida útil restante.')
    st.subheader('Historial de evaluaciones de esta sesión')
    hist=pd.DataFrame(st.session_state.get('history',[]))
    if len(hist):
        chart=go.Figure(go.Scatter(x=hist['Evaluación'],y=hist['Puntuación'],mode='lines+markers',line={'color':'#38BDF8'})); chart.add_hline(y=threshold,line_dash='dash',annotation_text='Umbral'); chart.update_layout(yaxis_range=[0,1],height=280,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#E6EDF5'); st.plotly_chart(chart,use_container_width=True)
    else: st.info('Evalúa nuevas lecturas para construir el historial.')
elif page=='Análisis por lote':
    st.subheader('Evaluación de registros CSV')
    source=st.file_uploader('Cargar CSV con las seis columnas AI4I',type='csv')
    demo=st.checkbox('Usar registros del conjunto de prueba guardado',value=False)
    if source is not None or demo:
        try:
            df=pd.read_csv(source if source is not None else ROOT/'results/test_predictions.csv'); scores=predict(df); result=df[RAW].copy(); result['score']=scores; result['alerta']=(scores>=threshold).astype(int)
            if 'Machine failure' in df: result['etiqueta_real']=df['Machine failure']
            elif 'y_true' in df: result['etiqueta_real']=df['y_true']
            flags=outside(df)
            if flags: st.warning('Hay lecturas fuera del rango de entrenamiento: '+', '.join(flags))
            a,b,c=st.columns(3); a.metric('Registros',len(result)); b.metric('Alertas',int(result.alerta.sum())); c.metric('Sin alerta',int((1-result.alerta).sum())); st.dataframe(result.sort_values('score',ascending=False),use_container_width=True); st.download_button('Descargar predicciones',result.to_csv(index=False).encode(),'predicciones.csv','text/csv')
        except Exception as e: st.error(f'No se pudo procesar el CSV: {e}')
else:
    st.subheader('Evaluación en conjunto de prueba')
    try:
        m=json.loads((ROOT/'results/metrics.json').read_text()); cols=st.columns(4)
        for col,k,label in zip(cols,['average_precision','roc_auc','f1','recall'],['Average Precision','ROC-AUC','F1','Recall']): col.metric(label,f'{m[k]:.2%}')
        st.caption(f"{m['n_test']} registros · prevalencia {m['prevalence']:.2%} · umbral seleccionado en validación")
        a,b=st.columns(2)
        for col,name in [(a,'precision_recall'),(b,'roc'),(a,'confusion_matrix'),(b,'feature_importance')]: col.image(str(ROOT/f'results/{name}.png'),use_container_width=True)
        st.json(m)
    except FileNotFoundError: st.info('Ejecuta el entrenamiento para generar métricas y gráficas.')
st.divider(); st.caption('Kexia · Demostración de investigación aplicada. No sustituye inspección, mantenimiento ni sistemas de seguridad industrial.')
