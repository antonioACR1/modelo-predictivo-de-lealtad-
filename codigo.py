#PARTE 1: LEER, ANALIZAR, LIMPIAR LOS DATOS Y PREPARAR EL FORMATO PARA EL MODELADO

import pandas as pd

#al intentar leer los datos me sale el siguiente error:
#UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfc in position 5: invalid start byte
    
#hago lo siguiente para leer el archivo sin que me salga ese error

import chardet
with open('blacktrust.csv','rb') as f:
    result=chardet.detect(f.read())

df=pd.read_csv('blacktrust.csv',encoding=result['encoding'],header=0) 
df.head()

#Out[210]: 
#   Id  Nivel_de_Satisfaccion  Ultima_Evaluacion FechaDeInicioLicenciatura  \
#0   1                   0.38               0.53                07/09/2003   
#1   2                   0.80               0.86                01/10/2007   
#2   3                   0.11               0.88                01/11/2006   
#3   4                   0.72               0.87                25/07/2008   
#4   5                   0.37               0.52                27/01/2005   
#   Numero_de_Proyectos  Promedio_HorasDeTrabajoAlMes   Escolaridad  \
#0                    2                         157.0  Licenciatura   
#1                    5                         262.0  Licenciatura   
#2                    7                         272.0  Licenciatura   
#3                    5                         223.0  Licenciatura   
#4                    2                         159.0  Licenciatura   
#   antigüedad_Empleo  Accidentes_de_Trabajo  AscensosUltimos3ańos  \
#0                  3                      0                     0   
#1                  6                      0                     0   
#2                  4                      0                     0   
#3                  5                      0                     0   
#4                  3                      0                     0   
#  Area_Trabajo Salario  Obj  
#0       ventas    bajo    1  
#1       ventas   medio    1  
#2       ventas   medio    1  
#3       ventas    bajo    1  
#4       ventas    bajo    1  

#ver tipo de variables
df.dtypes

#ver cuantas filas y columnas hay
df.shape
#Out[221]: (14999, 13)

#ver cuántos NAN's tiene cada variable

df.isnull().sum()
#Out[220]: 
#Id                                 0
#Nivel_de_Satisfaccion              0
#Ultima_Evaluacion                  0
#FechaDeInicioLicenciatura          0
#Numero_de_Proyectos                0
#Promedio_HorasDeTrabajoAlMes    1577
#Escolaridad                        0
#antigüedad_Empleo                  0
#Accidentes_de_Trabajo              0
#AscensosUltimos3ańos               0
#Area_Trabajo                       0
#Salario                            0
#Obj                                0


#quitar filas con NAN's
df=df.dropna(axis=0)

#cuantos valores distintos hay en 'Escolaridad'?

df['Escolaridad'].unique()
#Out[225]: array(['Licenciatura'], dtype=object)

#elimino las variables 'Id','FechaDeInicioLicenciatura' y 'Escolaridad' porque no las considero relevantes para la clasificación
#La escolaridad no la considero relevante porque todas las observaciones tienen las misma escolaridad

df=df.drop(labels=['Id','FechaDeInicioLicenciatura','Escolaridad'],axis=1)

#las variables 'Area_Trabajo' y 'Salario' son categóricas, por tanto las convertiré a númerico con LabelEncoder

df[['Area_Trabajo','Salario']].head()

#Out[231]: 
#  Area_Trabajo Salario
#0       ventas    bajo
#1       ventas   medio
#2       ventas   medio
#3       ventas    bajo
#4       ventas    bajo

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['Area_Trabajo']=le.fit_transform(df['Area_Trabajo'])
df['Salario']=le.fit_transform(df['Salario'])

df[['Area_Trabajo','Salario']].head()

#Out[233]: 
#   Area_Trabajo  Salario
#0             9        1
#1             9        2
#2             9        2
#3             9        1
#4             9        1


#ahora utilizaré train_test_split para separar my dataset en dos, una para el entrenamiento y otra para la evaluacion
#el tamaño del dataset para evaluación sera un 30% del dataset total

from sklearn.cross_validation import train_test_split

#copio la variable 'Obj' y después la quito de df
y=pd.Series(df['Obj'].copy())
df=df.drop(labels=['Obj'],axis=1)

#ahora aplico la separación
df_train,df_test,y_train,y_test=train_test_split(df,y,test_size=0.3)

#PARTE 2: MODELADO Y EVALUACIÓN

#ahora aplico los siguientes algoritmos (con los paámetros default en cada uno) para ver cuál da mejor predicción:
#RandomForestClassifier(),XGBClassifier(), GradientBoostingClassifier(), LogisticRegression(), MLPClassifier(), DecisionTreeClassifier(), GaussianNB(), SVC(), ExtraTreesClassifier() y KNeighborsClassifier()     

#para medir la exactitud de cada algoritmo, usare el dataset de evaluacion (df_test,y_test) junto con accuracy_score

from sklearn.metrics import accuracy_score                    
                                                                           
#para cada uno de los algoritmos antes mencionados llamaré al algoritmo, aplicaré el atributo .fit() para entrenar, luego el atributo .predict() para predecir y finalmente accuracy_score() para medir la exactitud

#guardaré el nombre del algoritmo y su accuracy_score en dos listas, 'algoritmo' y 'score'                                                                                             
                                                                                                 
algoritmo=[]
score=[]                                                                                                  
                                                                                                  
#random forest                                                                                                  
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(df_train,y_train)
predictions=rf.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]: 0.98832878073007202

#guardar resultados
algoritmo.append('random forest')
score.append(accuracy_score(predictions,y_test))

#XGBClassifier
import xgboost    
xgb=xgboost.XGBClassifier()
xgb.fit(df_train,y_train)
predictions=xgb.predict(df_test)
accuracy_score(predictions,y_test)
#Out[269]: 0.98792599950335235

#guardar resultados    
algoritmo.append('xgboost')
score.append(accuracy_score(predictions,y_test))

#gradient boosting                                                                                                 
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(df_train,y_train)
predictions=gb.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]: 0.97467097094611377

#guardar resultados
algoritmo.append('gradient boosting')
score.append(accuracy_score(predictions,y_test))

#logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(df_train,y_train)
predictions=lr.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]: 0.74372982369009188

#guardar resultados
algoritmo.append('logistic regression')
score.append(accuracy_score(predictions,y_test))

#neural network
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.fit(df_train,y_train)
predictions=mlp.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]: 0.90116712192699278

#guardar resultados
algoritmo.append('neural network')
score.append(accuracy_score(predictions,y_test))


#decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(df_train,y_train)
predictions=dt.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]: 0.97516761857462131

#guardar resultados
algoritmo.append('decision tree')
score.append(accuracy_score(predictions,y_test))

#naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(df_train,y_train)
predictions=nb.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]:  0.76434070027315615

#guardar resultados
algoritmo.append('naive bayes')
score.append(accuracy_score(predictions,y_test))

#support vector machine
from sklearn.svm import SVC
svc=SVC()
svc.fit(df_train,y_train)
predictions=svc.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]:  0.94512043704991311

#guardar resultados
algoritmo.append('support vector machine')
score.append(accuracy_score(predictions,y_test))


#extra tree
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier()
et.fit(df_train,y_train)
predictions=et.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]:  0.98733548547305683

#guardar resultados
algoritmo.append('extra tres')
score.append(accuracy_score(predictions,y_test))


#k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(df_train,y_train)
predictions=knn.predict(df_test)
accuracy_score(predictions,y_test)
#Out[264]:  0.92028805562453442

#guardar resultados
algoritmo.append('k nearest neighbors')
score.append(accuracy_score(predictions,y_test))

#reviso los dos scores mas altos
pd.DataFrame({'algoritmo':algoritmo,'score':score})
#Out[322]: 
#                algoritmo     score
#0           random forest  0.988080
#1                 xgboost  0.987926
#2       gradient boosting  0.974671
#3     logistic regression  0.743730
#4          neural network  0.898932
#5           decision tree  0.975664
#6             naive bayes  0.764341
#7  support vector machine  0.945120
#8              extra tres  0.987087
#9     k nearest neighbors  0.920288

#los mas altos fueron random forest y xgboost y por tanto trabajaré un poco más con estos

#aplicare un loop en random forest sobre el parámetro 'n_estimators' para intentar mejorar la prediccion

#creo dos listas 'valores_exactitud' para guardar el accuracy_score e 'indices' para el parámtero 'n_estimators'
valores_exactitud=[]
indices=[]

#aplico un loop para random forest sobre el parámetro 'n_estimators' de 10 en 10
for i in range(0,20):
    rf=RandomForestClassifier(n_estimators=(100 + 10*i))
    rf.fit(df_train,y_train)
    predictions=rf.predict(df_test)
    valores_exactitud.append(accuracy_score(predictions,y_test))
    indices.append((100+10*i))
#pongo las listas como dataframe para encontrar los scores máximos y su respectivo parámetro 'n_estimators'  
mejores_parametros=pd.DataFrame({'n_estimators':indices,'accuracy_score':valores_exactitud})    
#encontrar los scores máximos
mejores_parametros[mejores_parametros['accuracy_score']==max(mejores_parametros['accuracy_score'])]

#Out[334]: 
#    accuracy_score  n_estimators
#0         0.989819           100
#9         0.989819           190
#13        0.989819           230
#14        0.989819           240
#17        0.989819           270

#se obtiene una mejor prediccion cuando tomamos por ejemplo n_estimators = 100
rf=RandomForestClassifier(n_estimators=100)

#ahora aplico otro loop para xgboost sobre los parámtros n_estimators (de 10 en 10) y max_depth (de 2 a 20)

valores_exactitud=[]
indices=[]
for i in range(0,20):
    for j in range(5,20):
        xgb=xgboost.XGBClassifier(n_estimators=(100+10*i),max_depth=j)
        xgb.fit(df_train,y_train)
        predictions=xgb.predict(df_test)
        valores_exactitud.append(accuracy_score(predictions,y_test))
        indices.append([(100+10*i),j])

#pongo las dos listas en un dataframe para encontrar los scores máximos
mejores_parametros=pd.DataFrame({'[n_estimators, max_depth]':indices,'accuracy_score':valores_exactitud})        
#encontrar los scores máximos
mejores_parametros[mejores_parametros['accuracy_score']==max(mejores_parametros['accuracy_score'])]
        
#Out[337]: 
#    [n_estimators, max_depth]  accuracy_score
#85                  [150, 15]        0.989819
#130                 [180, 15]        0.989819
#140                 [190, 10]        0.989819
#145                 [190, 15]        0.989819
#155                 [200, 10]        0.989819
#157                 [200, 12]        0.989819
#160                 [200, 15]        0.989819
#170                 [210, 10]        0.989819
#199                  [230, 9]        0.989819
#200                 [230, 10]        0.989819
#214                  [240, 9]        0.989819
#215                 [240, 10]        0.989819
#229                  [250, 9]        0.989819
#244                  [260, 9]        0.989819
#245                 [260, 10]        0.989819
#259                  [270, 9]        0.989819
#260                 [270, 10]        0.989819
#274                  [280, 9]        0.989819
#289                  [290, 9]        0.989819
        
#se obtiene una mejor predicción cuando tomamos por ejemplo n_estimators=150 y max_depth=15

xgb=xgboost.XGBClassifier(n_estimators=150,max_depth=15)        

#finalmente hago un ensamble de xgboost y random forest con ayuda de VotingClassifier
#luego aplicto .fit(), .predict() y mido el accuracy_score

from sklearn.ensemble import VotingClassifier
ensemble=VotingClassifier([('xgboost',xgb),('random forest',rf)])
ensemble.fit(df_train,y_train)
predictions=ensemble.predict(df_test)
accuracy_score(predictions,y_test)
#Out[343]: 0.99031537124410229

#Se logró una mejor predicción (99%) al combinar random forest (98%) con xgboost (98%)


#PARTE 3: Conclusiones

#analizaré las variables más importances según los algortimos random forest y xgboost
#para ese fin utilizaré el atributo .feature_importances_

#entreno nuevamente los dos algoritmos 
rf.fit(df_train,y_train)
xgb.fit(df_train,y_train)

#llamo matplotlib para visualizar las variables más importantes para random forest
import matplotlib
import matplotlib.pyplot as plt

#saco la improtancia de cada variable 
importances=rf.feature_importances_
#pongo los nombres de cada variable (abreviado)
variables=["satisf","eval","proy","horasT","antig","accid","ascen","area","salar"]

#para visualizar las variables más importantes para random forest
plt.figure()
plt.title("variables mas importantes")
plt.xticks(range(df_train.shape[1]),variables)
plt.bar(range(df_train.shape[1]),importances,color="b",align="center")

#el gráfico se llama 'variables importantes random forest.png' en los archivos adjuntos

#según random forest, las tres variables más importantes son, de mayor a menor: el nivel de satisfacción, la antigüedad y el número de proyectos

pd.DataFrame(df_train.columns,rf.feature_importances_)

#Out[351]: 
#                                     0
#0.353376         Nivel_de_Satisfaccion
#0.121005             Ultima_Evaluacion
#0.172566           Numero_de_Proyectos
#0.128862  Promedio_HorasDeTrabajoAlMes
#0.198130             antigüedad_Empleo
#0.004838         Accidentes_de_Trabajo
#0.000928          AscensosUltimos3ańos
#0.011901                  Area_Trabajo
#0.008393                       Salario

#repito lo mismo para xgboost  

#para eso saco la improtancia de cada variable 
importances=xgb.feature_importances_
#luego pongo los nombres de cada variable (abreviado)
variables=["satisf","eval","proy","horasT","antig","accid","ascen","area","salar"]

#y para visualizar las variables más importantes para random forest:
plt.figure()
plt.title("variables mas importantes")
plt.xticks(range(df_train.shape[1]),variables)
plt.bar(range(df_train.shape[1]),importances,color="b",align="center")

#el gráfico se llama 'variables importantes xgboost.png' en los archivos adjuntos

#según xgboost, las tres variables más importantes son, de mayor a menor: promedio de horas de trabajo, nivel de satisfacción y la última evaluación

pd.DataFrame(df_train.columns,xgb.feature_importances_)
#Out[353]: 
#                                     0
#0.238836         Nivel_de_Satisfaccion
#0.207937             Ultima_Evaluacion
#0.081799           Numero_de_Proyectos
#0.265185  Promedio_HorasDeTrabajoAlMes
#0.068783             antigüedad_Empleo
#0.009524         Accidentes_de_Trabajo
#0.000423          AscensosUltimos3ańos
#0.088148                  Area_Trabajo
#0.039365                       Salario

#aunque las tres variables más importantes no son las mismas para cada algoritmo, sí podemos concluir que el nivel de satisfacción es fundamental para la clasificación

#por último, realizaré un gráfico en 3D para visualizar las tres variables más importantes según xgboost con respecto a los empleados que se van y los que se quedan
#para eso, utilizaré las variables 'Nivel_de_Satisfacción','Promedio_HorasDeTrabajoAlMes', 'Ultima_Evaluacion' y 'Obj' de todo el dataset df

#llamo Axes3D y pylab para los gráficos en 3D
from mpl_toolkits.mplot3d import Axes3D
import pylab
fig=pylab.figure()
ax=Axes3D(fig)
ax.set_zlabel("horas de trabajo")
ax.set_ylabel("última evauación")
ax.set_xlabel("satisfaccion")
#las observaciones donde 'Obj' es 0 estarán representadas por puntos rojos mientras que las que tienen 'Obj' igual a 0 serán puntos azules
ax.scatter(df['Nivel_de_Satisfaccion'],df['Ultima_Evaluacion'],df['Promedio_HorasDeTrabajoAlMes'],s=.8,c=['red' if l==0 else 'blue' for l in y])
plt.show() 

#el gráfico se llama 'variables importantes 3D xgboost.png' en los archivos adjuntos

#según el gráfico, los puntos rojos son empleados que siguen colaborando mientras que los puntos azules son los que dejan la empresa
#podemos apreciar que los empleados que siguen colaborando tuvieron en su mayoría un nivel de satisfacción entre 0.4 y 0.9
#y también podemos apreciar que por lo menos 2/3 partes de los empleados que se fueron tenían un promedio alto de horas de trabajo al mes
#así mismo, alrededor de 2/3 partes de los empleados que se fueron tenían un nivel de satisfacción menor a 0.4.
#finalmente, alrededor de 1/3 parte de los empleados que se fueron tenían una evaluación muy baja (menor a 0.6 según el gráfico).

#ahora repetiré un procedimiento similar para dos de las tres variables más importantes según random forest, a saber 'Promedio_HorasDeTrabajoAlMes' y 'Nivel_de_Satisfaccion'
#también agregaré la variable 'Promedio_HorasDeTrabajoAlMes' de xgboost. 
                                                                                                  
                                                                                                  
fig=pylab.figure()
ax=Axes3D(fig)
ax.set_xlabel("horas de trabajo")
ax.set_zlabel("numero proyectos")
ax.set_ylabel("satisfaccion")

#las observaciones donde 'Obj' es 0 estarán representadas por puntos rojos mientras que las que tienen 'Obj' igual a 0 serán puntos azules
ax.scatter(df['Promedio_HorasDeTrabajoAlMes'],df['Nivel_de_Satisfaccion'],df['Numero_de_Proyectos'],s=.8,c=['red' if l==0 else 'blue' for l in y])
plt.show() 

#el gráfico se llma 'variables importantes 3D random forest.png' en los archivos adjuntos
#del gráfico se puede ver que algunos de los empleados que dejaron la empresa tenían un número alto de horas de trabajo y de proyectos 

#ahora hago un zoom 
fig=pylab.figure()
ax=Axes3D(fig)
ax.set_xlabel("horas de trabajo")
ax.set_xlim([0,300])
ax.set_zlabel("numero proyectos")
ax.set_ylabel("satisfaccion")
ax.set_zlim([0,7])
#las observaciones donde 'Obj' es 0 estarán representadas por puntos rojos mientras que las que tienen 'Obj' igual a 0 serán puntos azules
ax.scatter(df['Promedio_HorasDeTrabajoAlMes'],df['Nivel_de_Satisfaccion'],df['Numero_de_Proyectos'],s=.8,c=['red' if l==0 else 'blue' for l in y])
plt.show() 

#el zoom se llama 'variables importantes 3D random forest zoom.png' en los archivos adjuntos

#a partir de este zoom se puede volver a verificar que varios de los que dejaron la empresa tenían un número de proyectos y horas de trabajo alto.
#sin embargo, es de notar que hay muchos empleados que siguen laborando en la empresan también tenían un número de proyectos y horas de trabajo similar.


#PARTE 4: Guardar modelo con pickle y abrir nuevamente

#poner nombre al modelo con terminación pkl
nombre = 'pkl_ensemble.pkl'
#abrirlo y guardarlo
pkl_ensemble = open(nombre, 'wb')
pickle.dump(ensemble, pkl_ensemble)
# cerrarlo
pkl_ensemble.close()

#cuando haya un batch de nuevas observaciones listas para entrenamiento, abrimos nuevamente el modelo

ensemble_actualizado = open(nombre, 'rb')
ensemble_actualizado = pickle.load(ensemble_actualizado)

#y entrenar con nuevas observaciones (me aseguro que estén en el mismo formato con el que entrené mi modelo original)

#por ejemplo, que los nuevos datos se llamen df_actualizado, y_actualizado y entonces el entrenamiento sería:
ensemble_actualizado.fit(df_actualizado,y_actualizado)

#FIN










