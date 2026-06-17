#Custom loss functions for glucose prediction
#Created: 2025-03
#Autor: Eloy Prieto Panadero

#Dependencias:
import tensorflow as tf

# ------------------------------------------
# Funciones Básicas (Operaciones TensorFlow)
# ------------------------------------------
def sigmoid(x, centro, pendiente):
    """Función sigmoide para Clinical_Penalty."""
    return 1.0 / (1.0 + tf.exp(-pendiente * (x - centro)))

def MAE(y_true, y_pred):
    """Mean Absolute Error (Error Absoluto Medio)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return tf.reduce_mean(tf.abs(y_pred - y_true))

MAD=MAE # DUDA

def MARD(y_true, y_pred): 
    """Mean Absolute Relative Difference (Diferencia Absoluta Relativa Media)."""
    #Tambien se llama MAPE
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    epsilon = tf.constant(1e-10, dtype=tf.float64)
    relative_diff = tf.abs(y_pred - y_true) / tf.maximum(tf.abs(y_true), epsilon)
    return tf.reduce_mean(relative_diff)

def MSE(y_true, y_pred):
    """Mean Squared Error (Error Cuadrático Medio)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return (tf.square(y_pred - y_true))

def RMSE(y_true, y_pred):
    """Root Mean Squared Error (Raíz del Error Cuadrático Medio)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# ------------------------------------------
# Métricas/Funciones de Pérdida Clínicas
# ------------------------------------------
def clinical_penalty(y_true, y_pred):
    """Función de penalización clínica para glucosa en sangre.
    A Glucose-Specific Metric to Assess Predictors and Identify Models
    Simone Del Favero, Andrea Facchinetti, and Claudio Cobelli, Fellow, IEEE
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6135492
    https://github.com/MiceLab/MiceLab-Grammatical-Evolution-and-Type-1-Diabetes/blob/master/src/ge/EvaluateDataTable.java
    Dominio [0, 500] e Imagen [1, 2.5] de minimización
    """
    output = tf.cast(y_pred, tf.float64)
    target = tf.cast(y_true, tf.float64)
    tetaH = 155.0
    betaH = 100.0
    tetaL = 85.0
    betaL = 30.0
    lambdaH = 20.0
    lambdaL = 10.0
    alphaL = 1.5
    alphaH = 1.0

    Aux1zona1 = sigmoid(target, tetaL - (betaL / 2.0), -0.3)
    Aux2zona1 = sigmoid(output, target + (lambdaL / 2.0), 0.6)
    zona1 = alphaL * Aux1zona1 * Aux2zona1

    Aux1zona2 = sigmoid(target, tetaH + (betaH / 2.0), 0.1)
    Aux2zona2 = sigmoid(output, target - (lambdaH / 2.0), -0.4)
    zona2 = alphaH * Aux1zona2 * Aux2zona2
    
    zona3 = 1.0

    return zona1 + zona2 + zona3

def gMAD(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    """
    Penalized Mean Absolute Deviation for glucose prediction (gMAD)

    Args:
        y_true: Ground truth glucose values g(t)
        y_pred: Predicted glucose values ĝ(t)
    """
    return clinical_penalty(y_true, y_pred) * tf.abs(y_true - y_pred)

def gMSE(y_true, y_pred):
    """Glucose Mean Squared Error (Error Cuadrático Medio Glucoseado)."""
    return clinical_penalty(y_true, y_pred)*MSE(y_true, y_pred)

# def gRMSE_italiana(y_true, y_pred): # DUDA: no se usa
#     """Glucose Root Mean Squared Error (Raíz del Error Cuadrático Medio Glucoseado)."""
#     y_true = tf.cast(y_true, tf.float64)
#     y_pred = tf.cast(y_pred, tf.float64)
#     pen = clinical_penalty(y_true, y_pred)
#     return tf.sqrt(tf.reduce_mean(pen * tf.square(y_true - y_pred)))



def c_sig(y_true, y_pred, p=0.1):
    """Función de penalización clínica para glucosa en sangre.
    Basada en el Clark Error Grid (CEG).
    """
    output = tf.cast(y_pred, 'float64') #EJE Y
    target = tf.cast(y_true, 'float64') #EJE X
    
    #p=0.1 #Yo lo elijo, siempre que p>0.0
    #p= 0.1, 0.6
    # de 1 a 5 lineal el maximo o de 1 a 25 cuadratico
    
    
    ##ZONA A ################################
    A=1.0
    
    #Criterio: D1 = D superior a diagonal
    ##ZONA D1 ################################
    #D11 (recta horizontal entre A y D1)
    pD11=p #Yo lo elijo, siempre que p>0.0
    #mD11=0.0 #0.0 por definición
    cD11=70.0 #70.0 por definición
    D11=sigmoid(output, cD11, pD11) #D11=sigmoide(output - mD11*target, pD11, cD11)
    #D12 (mini diagonal entre A y D1)
    pD12=p #Yo lo elijo, siempre que p>0.0
    mD12=1.2 #(500-70)/(500/1.2-175/3)=6/5=1.2 (+20% error between output and target)
    cD12=0.0 #70.0-mD12*175.0/3.0=0.0 
    D12=sigmoid(output - mD12 * target, cD12, pD12)
    #D13 (recta vertical entre D1 y B1 o C1)
    pD13=-p #Yo lo elijo, siempre que p<0.0
    #mD13=1.0 
    cD13=70.0 #70.0 por definición
    D13=sigmoid(target, cD13, pD13)
    
    D1=D11*D12*D13*2.0
    
    ##ZONA E1 #############################
    #E11 (recta entre D1 y E1)
    # Esta zona no es necesaria definirla
    # se crea sola al sumar D1 y C1, que pasan por debajo de E1
    pE11=p #Yo lo elijo, siempre que p>0.0
    cE11=180.0 #180.0 por definición
    E11=sigmoid(output, cE11, pE11)
    #E12 (recta entre E1 y C1)
    #pE12=-p #Yo lo elijo, siempre que p<0.0
    #cE12=70.0 #70.0 por definición
    #E12=sigmoide(target, cE12, pE12)
    
    E1=0.0
    #E1=E11*E12
    
    ##ZONA C1 #############################
    #C11 (recta entre C1 y D1)
    C11=E11
    #C12 (diagonal entre C1 y B1)
    pC12=p #Yo lo elijo, siempre que p>0.0
    #mC12=1.0 #(500-180)/(390-70)=1.0
    cC12=110.0 #180-mC12*70
    C12=sigmoid(output - target, cC12, pC12)
    
    C1=C11*C12

    ##ZONA B1 #############################
    #B11 (horizontal entre A y B1)
    B11=D11 
    #B12 (diagonal entre A y B1)
    B12=D12 #(+20% error between output and target)
    
    B1=B11*B12
    
    ##ZONA B2 #############################
    #B21 (recta entre A y B2)
    pB21=p #Yo lo elijo, siempre que p>0.0
    #mB21=1.0 #1.0 por definición
    cB21=70.0 #70.0 por definición
    B21=sigmoid(target, cB21, pB21)
    #B22 (diagonal entre A y B2)
    pB22=-p #Yo lo elijo, siempre que p<0.0
    mB22=0.8 #(400-56)/(500-70)=4/5=0.8 (-20% error between output and target)
    cB22=0.0  #56-mB22*70=0.0
    B22=sigmoid(output - mB22 * target, cB22, pB22)
    
    B2=B21*B22
    
    ##ZONA C2 #############################
    #C21 (diagonal entre B2 y C2)
    pC21=-p #Yo lo elijo, siempre que p<0.0
    mC21=1.4 #70/(180-130)=7/5=1.4
    cC21=-182.0 #(0-70)/(180-130)*130.0
    C21=sigmoid(output - mC21 * target, cC21, pC21)
    #C22 (recta entre E2 y D2)
    pC22=-p #Yo lo elijo, siempre que p<0.0
    cC22=70.0 #70.0 por definición
    C22=sigmoid(output, cC22, pC22)
    
    C2=C21*C22
    
    ##ZONA D2 #############################
    #D21 (recta vertical entre B2 y D2)
    pD21=p #Yo lo elijo, siempre que p<0.0
    cD21=240.0 #240.0 por definición
    D21=sigmoid(target, cD21, pD21)
    #D22 (recta horizontal entre B2 y D2)
    pD22=-p #Yo lo elijo, siempre que p<0.0
    cD22=180.0 #180.0 por definición
    D22=sigmoid(output, cD22, pD22)
    
    D2=D21*D22*2.0
    
    ##ZONA E2 #############################
    #E21 (recta vertical entre C2 y E2)
    pE21=p #Yo lo elijo, siempre que p>0.0
    cE21=180.0 #180.0 por definición
    E21=sigmoid(target, cE21, pE21)
    #E22 (recta horizontal entre B2 o D2 y E2)
    E22=C22
    
    E2= E21 * E22 * 2.0 * sigmoid(target, cD21, -pD21)
    #E2=E21*E22*(1.0+2.0*sigmoide(target, cD21, -pD21))
    
    return A + B1 + B2 + C1 + C2 + D1 + D2 + E1 + E2

def c_sig_p01(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1)


def c_sig_p06(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6)


def c_sig_MSE_p01(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1) * MSE(y_true, y_pred)


def c_sig_MSE_p06(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6) * MSE(y_true, y_pred)


#import pandas as pd
#import numpy as np
# Cargar el archivo CSV
# SEG_df = pd.read_csv('Riskpairdata.csv') # DUDA: para qué sirve este CSV si no se implementó finalmente seg?

# def Surveillance_Error_Grid(y_true, y_pred): # DUDA: no se implementó correctamente?
#     y_true = int(abs(y_true))
#     y_pred = int(abs(y_pred))
#     # Buscar la fila que coincida con REF=y_true y BGM=y_pred
#     match = SEG_df[(SEG_df['REF'] == y_true) & (SEG_df['BGM'] == y_pred)]
#
#     if not match.empty:
#         return match.iloc[0]['abs_risk']+1.0
#     else:
#         # Calcular la distancia euclidiana entre (y_true, y_pred) y cada (REF, BGM)
#         SEG_df['distance'] = np.sqrt((SEG_df['REF'] - y_true)**2 + (SEG_df['BGM'] - y_pred)**2)
#
#         # Encontrar la fila con la menor distancia
#         closest_row = SEG_df.loc[SEG_df['distance'].idxmin()]
#
#         return closest_row['abs_risk']+1.0
#
#         #return None
#

# Convertir REF y BGM a tensores de TensorFlow una sola vez
'''
ref_tensor = tf.constant(SEG_df['REF'].values, dtype=tf.float32)
bgm_tensor = tf.constant(SEG_df['BGM'].values, dtype=tf.float32)
abs_risk_tensor = tf.constant(SEG_df['abs_risk'].values, dtype=tf.float32)

def Surveillance_Error_Grid222(y_true, y_pred):
    # Convertir entradas a enteros absolutos (como en tu versión original)
    y_true = tf.cast(tf.abs(y_true), tf.int32)
    y_pred = tf.cast(tf.abs(y_pred), tf.int32)

    # Buscar coincidencia exacta (usando TensorFlow)
    exact_match_mask = tf.math.logical_and(
        tf.equal(ref_tensor, tf.cast(y_true, tf.float32)),
        tf.equal(bgm_tensor, tf.cast(y_pred, tf.float32))
    )
    
    exact_match_indices = tf.where(exact_match_mask)
    
    if tf.size(exact_match_indices) > 0:
        return abs_risk_tensor[exact_match_indices[0, 0]].numpy()
    
    # Si no hay coincidencia exacta, calcular distancias euclidianas
    y_true_float = tf.cast(y_true, tf.float32)
    y_pred_float = tf.cast(y_pred, tf.float32)
    
    distances = tf.sqrt(
        tf.square(ref_tensor - y_true_float) + 
        tf.square(bgm_tensor - y_pred_float)
    )
    
    # Encontrar el índice de la distancia mínima
    closest_idx = tf.argmin(distances)
    
    return abs_risk_tensor[closest_idx].numpy()
'''
#Surveillance_Error_Grid=np.vectorize(Surveillance_Error_Grid)

# def Surveillance_Error_Grid_MSE(y_true, y_pred):
#     """Surveillance Error Grid Mean Squared Error (Error Cuadrático Medio de la Grilla de Vigilancia)."""
#     return Surveillance_Error_Grid(y_true, y_pred)*MSE(y_true, y_pred)

def c_sig_p01(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1)


def c_sig_p06(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6)


def c_sig_p01_MSE(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1) * MSE(y_true, y_pred)


def c_sig_p06_MSE(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6) * MSE(y_true, y_pred)

#FIN

# ----------------------- Custom loss function -----------------------

loss_functions_dic = {"CP": clinical_penalty,
                      "cSigP01": c_sig_p01,  # Former: EloyPenaltyp01,  # c_sig with p = 0.1
                      "cSigP06": c_sig_p06,  # Former: EloyPenaltyp06, # c_sig with p = 0.6
                      "RMSE": RMSE,
                      "gMSE": gMSE,
                      "gMAD": gMAD,
                      "cSigP01MSE": c_sig_p01_MSE,  #  Former: EloyPenaltyMSEp01, # c_sig_MSE with p = 0.1
                      "cSigP06MSE": c_sig_p06_MSE,  #  Former: EloyPenaltyMSEp06, # c_sig_MSE with p = 0.6
                      }

loss_function_name_list = ['RMSE', 'cSigP01', 'cSigP06', 'CP', 'gMSE', 'gMAD', 'cSigP01MSE', 'cSigP06MSE'] # ALL

# ----------------------- End Custom loss function -----------------------

# ------------------------------------------
# Clases para Keras (usar en build_model -> compile -> metrics)
# ------------------------------------------

class ClinicalPenaltyMetric(tf.keras.losses.Loss):
#class ClinicalPenaltyMetric(tf.keras.metrics.Metric):
    """Clase para Clinical_Penalty como función de pérdida en Keras."""
    def __init__(self, name='Clinical_Penalty', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return clinical_penalty(y_true, y_pred)

class c_sig_p01_Metric(tf.keras.losses.Loss):
#class ClinicalPenaltyMetric(tf.keras.metrics.Metric):
    """Clase para Clinical_Penalty como función de pérdida en Keras."""
    def __init__(self, name='c_sig_p01', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return c_sig_p01(y_true, y_pred)
    
class c_sig_p06_Metric(tf.keras.losses.Loss):
#class ClinicalPenaltyMetric(tf.keras.metrics.Metric):
    """Clase para Clinical_Penalty como función de pérdida en Keras."""
    def __init__(self, name='c_sig_p06', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return c_sig_p06(y_true, y_pred)
    
