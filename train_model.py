"""Entrena XGBoost y exporta resultados reproducibles; ejecutar una vez."""
import argparse, hashlib, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, brier_score_loss)
from features import RAW, TARGET, transform
ROOT = Path(__file__).resolve().parent

def metrics(y,s,t):
    p = (s>=t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(average_precision=average_precision_score(y,s), roc_auc=roc_auc_score(y,s), accuracy=accuracy_score(y,p), balanced_accuracy=balanced_accuracy_score(y,p), precision=precision_score(y,p,zero_division=0), recall=recall_score(y,p,zero_division=0), f1=f1_score(y,p,zero_division=0), mcc=matthews_corrcoef(y,p), specificity=tn/(tn+fp), brier=brier_score_loss(y,s), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(t))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/ai4i2020.csv')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    path=args.data
    if not path.exists() and path==ROOT/'data/ai4i2020.csv': path=ROOT/'ai4i2020.csv'
    df=pd.read_csv(path); X=transform(df); y=df[TARGET].to_numpy()
    if set(np.unique(y))!={0,1}: raise ValueError('Se requieren etiquetas binarias 0 y 1.')
    train,tmp=train_test_split(np.arange(len(df)),test_size=.4,stratify=y,random_state=args.seed)
    val,test=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=args.seed)
    models=ROOT/'models'; out=ROOT/'results'; models.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    # Pequeña búsqueda exclusivamente sobre validación. No se submuestrea.
    ratio=float((y[train]==0).sum()/(y[train]==1).sum())
    candidates=[]; best=None; best_ap=-1
    start=time.perf_counter()
    for depth in [3,5]:
        for weight in [1.,ratio**.5,ratio]:
            model=XGBClassifier(n_estimators=500,max_depth=depth,learning_rate=.04,subsample=.9,colsample_bytree=.9,reg_lambda=2,scale_pos_weight=weight,eval_metric='aucpr',early_stopping_rounds=35,tree_method='hist',n_jobs=4,random_state=args.seed)
            t0=time.perf_counter(); model.fit(X.iloc[train],y[train],eval_set=[(X.iloc[val],y[val])],verbose=False)
            score=model.predict_proba(X.iloc[val])[:,1]; ap=average_precision_score(y[val],score)
            candidates.append(dict(depth=depth,weight=weight,validation_ap=ap,best_iteration=model.best_iteration,seconds=time.perf_counter()-t0))
            if ap>best_ap: best_ap=ap; best=model
    duration=time.perf_counter()-start
    vs=best.predict_proba(X.iloc[val])[:,1]
    p,r,t=precision_recall_curve(y[val],vs); f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12); threshold=float(t[np.argmax(f)])
    t0=time.perf_counter(); ts=best.predict_proba(X.iloc[test])[:,1]; inference=time.perf_counter()-t0
    report=metrics(y[test],ts,threshold); report.update(train_seconds=duration,inference_seconds=inference,n_test=len(test),prevalence=float(y[test].mean()))
    best.save_model(models/'xgboost.json')
    meta=dict(threshold=threshold,feature_names=list(X.columns),seed=args.seed,validation_ap=best_ap,versions=dict(python=platform.python_version(),xgboost=xgboost.__version__,sklearn=sklearn.__version__),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split_sizes=dict(train=len(train),validation=len(val),test=len(test)),selection='Maximum validation AP; threshold maximizes validation F1',ranges={c:[float(df.iloc[train][c].min()),float(df.iloc[train][c].max())] for c in RAW[1:]})
    (models/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([report]).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'validation_search.csv',index=False)
    np.savez(out/'splits.npz',train=train,validation=val,test=test)
    pred=df.iloc[test][RAW].copy(); pred.insert(0,'row_index',test); pred['y_true']=y[test]; pred['score']=ts; pred['prediction']=(ts>=threshold).astype(int); pred.to_csv(out/'test_predictions.csv',index=False)
    for name, xx, yy, xlabel,ylabel,baseline in [('precision_recall',*precision_recall_curve(y[test],ts)[:2][::-1],'Recall','Precision',report['prevalence']),('roc',*roc_curve(y[test],ts)[:2],'False positive rate','True positive rate',None)]:
        fig,ax=plt.subplots(figsize=(7,5)); ax.plot(xx,yy,label='XGBoost'); ax.set(xlabel=xlabel,ylabel=ylabel,title=name.replace('_',' ').title()); ax.grid(alpha=.2)
        if baseline is not None: ax.axhline(baseline,ls='--',label='Prevalence')
        else: ax.plot([0,1],[0,1],'--',label='Random')
        ax.legend(); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(); ConfusionMatrixDisplay.from_predictions(y[test],ts>=threshold,ax=ax,cmap='Blues'); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=180); plt.close(fig)
    imp=pd.Series(best.feature_importances_,index=X.columns).sort_values(); imp.to_csv(out/'feature_importance.csv',header=['importance']); fig,ax=plt.subplots(figsize=(8,6)); imp.plot.barh(ax=ax); ax.set_title('XGBoost feature importance'); fig.tight_layout(); fig.savefig(out/'feature_importance.png',dpi=180); plt.close(fig)
    print(json.dumps(report,indent=2)); print('Modelo y resultados guardados.')
if __name__=='__main__': main()
