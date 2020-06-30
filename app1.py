##Carregando os pacotes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as SKL
import sklearn.ensemble as skl
##Carregando os dados
df=pd.read_csv('wine_dataset.csv')
## Print dos primeiros dados!
#print(df.head())

##Estamos transformando a varivael style para uma variavel boleana
df['style']=df['style'].replace('red',0)
df['style']=df['style'].replace('white',1)
#print(df.head())
sns.heatmap(df.corr(),annot=True,fmt="0.2f")
plt.show()

##Separação em variaveis preditoras e variavel alvo
y=df['style']
x=df.drop('style',axis=1)
#print(x)

## criando conjunto treino e conjunto teste
x_treino,x_teste,y_treino,y_teste=SKL.train_test_split(x,y,test_size=0.3)# 30% vai virar teste

##Criação do modelo
modelo=skl.ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)
## imprimindo resultados:
resultados=modelo.score(x_teste,y_teste)
print("Acuracia:",resultados)

## Previsão com base no modelo
x_previsao=x_teste[400:403]
Y_testefinal=y_teste[400:403]
previsao=modelo.predict(x_previsao)
print("Previsão:",previsao)
print("Objetivo:",Y_testefinal)
