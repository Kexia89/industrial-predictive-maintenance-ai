"""Transformación compartida entre entrenamiento e inferencia."""
import numpy as np
import pandas as pd

RAW = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
TARGET = 'Machine failure'

def transform(df):
    missing = set(RAW) - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    x = df[RAW].copy()
    x['Type'] = x['Type'].astype(str).str.strip().str.upper()
    if not x['Type'].isin(['L','M','H']).all():
        raise ValueError('Type debe ser L, M o H.')
    for c in RAW[1:]:
        x[c] = pd.to_numeric(x[c], errors='raise')
    if not np.isfinite(x[RAW[1:]].to_numpy()).all() or (x[RAW[1:]] < 0).any().any():
        raise ValueError('Las lecturas deben ser finitas y no negativas.')
    air, process, rpm, torque, wear = [x[c] for c in RAW[1:]]
    x['temperature_delta'] = process-air
    x['angular_speed'] = rpm*2*np.pi/60
    x['mechanical_power_w'] = x['angular_speed']*torque
    x['torque_wear'] = torque*wear
    x['temperature_torque'] = (process-air)*torque
    for t in ['L','M','H']:
        x['type_'+t] = (x['Type']==t).astype(int)
    return x.drop(columns='Type').rename(columns=lambda c: c.replace('[','(').replace(']',')')).astype(float)
