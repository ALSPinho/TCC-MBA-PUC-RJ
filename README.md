
# Análise RNN-LSTM da serie temporal do Bitcoin aliada à análise de sentimentos de mídias sociais com VADER (Valence Aware Dictionary and sEntiment Reasoner)

#### Aluno:		[Alexandre Lima Santiago de Pinho](https://github.com/ALSPinho)
#### Orientador:	[Leonardo Fonte Mendoza]()
#### Co-orientador:	[Felipe Borges](https://github.com/FelipeBorgesC)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

[1-TCC-Tweets.ipynb]() - Tweet database analysis and wordcloud

[2-TCC-Vader-Price.ipynb] - Tweet sentiment analysis (VADER)

[3-TCC-Stock-Vader.ipynb] - Best LSTM configuration analysis for crypto, with incorporation of tweet sentiment analysis

[TCC-Resultados.docx] - Graphical results of all analyzed LSTM configurations

---

### Resumo

Este trabalho apresenta um modelo de previsão do Bitcoin através de redes neurais recorrentes (RNN) do tipo Long Short-Term Memory (LSTM).

O método proposto neste trabalho consiste em aplicar redes neurais LSTM na série histórica do Bitcoin para entender seu comportamento e identificar sua tendência. 

Na sequência também é avaliado o ganho no resultado de predição com a inclusão da análise de sentimento das redes sociais, aplicando o algoritmo de VADER (Valence Aware Dictionary and sEntiment Reasoner) a fim de apurar eficácia na convergência da determinação do preço do ativo.

Buscou-se minimizar o erro de validação através de métricas de regressão com otimização de hiper-parâmetros, além de várias configurações da RNN com diferentes funções de ativação. Os resultados mostraram boa performance sobre os dados de teste.


### Abstract

This work presents a Bitcoin prediction model through Long Short-Term Memory (LSTM) recurrent neural networks (RNN).

The method proposed in this work consists of applying LSTM neural networks in the Bitcoin historical series to understand its behavior and identify its trend.

After that, the gain in the prediction result is also evaluated with the inclusion of the sentiment analysis of social networks, applying the VADER (Valence Aware Dictionary and sEntiment Reasoner) algorithm in order to determine the effectiveness in the convergence of the determination of the asset price.

The goals were to minimize the validation error through regression metrics with hyper-parameter optimization. The results showed good performance on the test data.


### 1. Introdução

Sabe-se que para operar no mercado de capitais tem-se dois princípio: a análise técnica e a fundamentalista. A primeira se restringe à avaliação dos gráficos dos ativos; a segunda, avalia balanço patrimonial, fluxo de caixa, resultados de empresa e conjuntura econômica.

Para ativos digitais, tem-se como tendência de mercado a utilização quase que exclusiva da análise técnica dos criptoativos, uma vez que a análise fundamentalista de projetos que estão no início de seu desenvolvimento não são tão simples de serem realizadas.

A motivação deste trabalho tem como objetivo prático aplicar os conceitos de redes neurais para entender o comportamento dos diversos criptoativos e conseguir prever sua posição no tempo futuro, de modo a ser um aliado na tomada de decisão de compra e venda, sem inicialmente desconsiderar a conjuntura econômica, mas deixando a análise técnica convencional em segundo plano.


### 2. Modelagem

Foi definido um algoritmo para análise de sentimentos do tweeter e outro para a série histórica do Bitcoin, onde diversos estudos de otimização e melhoria de resultados foram realizados. A última etapa foi aplicar no melhor modelo RNN-LSTM o conjunto da série histórica do Bitcoin com análise de sentimentos do tweet para entender o ganho de informação na análise em conjunto.







[TCC-Resultados.docx] - Graphical results of all analyzed LSTM configurations


#### Análise de sentimentos por VADER

##### i) Preparação e análise dos dados

[1-TCC-Tweets.ipynb]() - Tweet database analysis and wordcloud

Foi utilizado a base de dados 'Bitcoin_tweets.csv' [(https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)] com aproximadamente 2.000.000 registros coletados no período de 2021/02/10 a 2021/12/29. 

Após inspeção da base de dados, com o intuito de facilitar sua manipulação e tratamento, criou-se um data-frame de trabalho de 200 registros. Concluída esta etapa, o pipeline de pre-tratamento do texto foi aplicado em toda a base de dados obtida no Kaggle. O pipeline de pre-tratamento do texto tem as funções definidas abaixo, sendo possível montar sua nuvem de pontos para melhor visualização dos resultados:
	I)    remoção acentuação;
	II)   converter www.* e https* para URL;
	III)  remoção números;
	IV)   converter @username para AT_USER;
	V)    remoção caracteres especiais;
	VI)   conversão caracteres para letras minúsculas;
	VII)  tokenizaçao;
	VIII) remoção stopwords;
	IX)   stemming;

Na sequencia foi realizado a classificação do sentimento do texto por Vader (Valence Aware Dictionary and sEntiment Reasoner), sendo obtida a distribuição de sentimentos abaixo:
	I)   positive    989549 registros;
	II)  neutral     742911 registros; e
	III) negative    267074 registros.

Exclusivamente por definição da etapa de trabalho para desenvolvimento deste projeto, foi definido que este pipeline de tratamento de dados iria ter como arquivo csv de saída: tweet_formated.

[2-TCC-Vader-Price.ipynb] - Tweet sentiment analysis (VADER)

Este arquivo, por sua vez, foi manipulado para concatenar todas os sentimentos de forma diária, entretanto, ao iniciar esta manipulação verificou-se a existência de 11 (onze) registros com caracteres "['ETH', 'BTC', 'Bitcoin']" na coluna da data, impossibilitando inclusive o tratamento desta coluna para o formato datetime. Sanada esta inconsistência, fez-se breve estudo com base na soma e na média dos registros com o respectivo agrupamento destes sentimentos para expressar um único valor. O arquivo csv de saída apresenta a média diária dos sentimentos: 'tweet_sentiment'


#### Serie histórica do Bitcoin


##### i) Preparação e análise dos dados

[3-TCC-Stock-Vader.ipynb] - Best LSTM configuration analysis for crypto, with incorporation of tweet sentiment analysis

A série histórica do Bitcoin, oftida a partir do Yfinance, juntamente com o arquivo 'tweet_sentiment' foram manipulados dando origem à um novo banco de dados denominado 'treated_df' onde tem-se registro do valor de fechamento do Bitcoin com o respectivo sentimento para o período de de 2021/02/10 a 2021/12/29. Entretanto, nesta etapa do projeto, somente a série histórica do Bitcoin foi analisada.

A verificação e exclusão de valores nulos foi realizada, bem como o tratamento básico dos dados conforme descrito abaixo para entrada de dados na rede LSTM:
	i)   definição janela de tempo;
	ii)  divisão base em treino e teste;
	iii) montagem matriz de entrada de dados para LSTM;
	iv)  Normalização dos dados com MinMaxScaler; e
	v)   Reshaping da Matriz.


##### ii) Implementação da rede neural

A implementação da rede LSTM foi realizada utilizando a biblioteca Keras, que é um framework prático e intuitivo para construir e treinar modelos de redes neurais. Algumas análises iniciais foram realizadas e constatou-se que os melhores resultados estavam com a configuração de 2 camadas de neurônios.

A rede foi então configurada com 2 camadas totalmente conectadas. Foram adicionadas camadas de eliminação (dropout) após cada camada LSTM oculta, para evitar o problema de overfitting devido à rede densa, entretanto, constatado que aplicação de dropout não trazia ganhos a rede para este projeto.

Após definir as camadas da rede, foram especificadas as configurações de aprendizagem. Foi definido o otimizador Relu com taxa de aprendizado padrão e a raiz erro médio quadrático (RMSE, do inglês Root Mean Squared Error) foi definido como função de perda.

Foram definidos valores iniciais de hiper-parâmetros para dar início ao processo de treinamento e otimização da rede. Os hiper-parâmetros iniciais foram definidos de forma empírica através de tentativa e erro, conforme demostrado no item seguinte. 


##### iii) Treinamento e resultados

Foram realizadas análises de forma a otimizar o resultado, considerando as variações abaixo:
	i)   Camadas variando entre 64, 32, 16, 8 e 5 neurônios;
	ii)  Funções de ativação Relu e LeakyRelu;
	iii) Variação épocas entre 200 e 100
	iv)  Variação batch-size ente 16 e 8
	v)   Estudo com EarlyStop com patience de 100 e 50, combinados com validation_split de 0.1 e 0.0001

A melhor configuração da RNN-LSTM foi com 2 camadas LSTM, respectivamente com 16 e 5 neurônios, ambas com função de ativação Relu, batch-size de 16 e 200 épocas. A partir desta configuração foram realizadas variações das camadas LSTM para melhoria dos resultados, considerando:
	i.	    Primeira Camada bidirecional
	ii.	    Primeira Camada bidirecional e camada BatchNormalizations após primeira LSTM
	iii.    Primeira Camada bidirecional e 2 camada BatchNormalizations após cada LSTM
	iv.	    Primeira Camada bidirecional, 2 camada BatchNormalizations após cada LSTM e Flatten antes da Dense
	v.	    2 Camadas bidirecional, 2 camada BatchNormalizations após cada LSTM e Flatten antes da Dense
	vi.	    LSTM, Segunda Camadas bidirecional, 2 camada BatchNormalizations após cada LSTM e Dense
	vii.	LSTM, Segunda Camadas bidirecional, BatchNormalizations e Dense
	viii.	2 LSTM, BatchNormalizations e Dense
	ix.	    2 LSTM, BatchNormalizations, Flatten e Dense
	x.	    2 LSTM, Flatten e Dense

A melhor configuração da RNN-LSTM continuou sendo com 2 camadas LSTM com 16 e 5 neurônios, ambas com função de ativação Relu, batch-size de 16 e 200 épocas.


#### Analise do Bitcoin com análise de sentimento do tweet

Nesta etapa do projeto, a preparação, análise dos dados e implementação da RNN seguiu o pipeline definido na análise isolada do Bitcoin, sendo que nesta fase foi inserida na matriz de entrada da LSTM a valor do sentimento do tweet.

Para a etapa de treinamento foi realizado somente para a configuração que apresentou a melhor configuração de pesos da LSTM, ou seja, 2 camadas LSTM com 16 e 5 neurônios, ambas com função de ativação Relu, batch-size de 16 e 200 épocas.

Como resultado, manteve-se aderência dos gráficos train-test e predição, visualmente sem alteração relevante. O valor RMSE apresentou-se ligeiramente mais alto, mas sem alteração significativa.



### 3. Resultados

Os resultados gráficos, comentados, das diversas análises realizadas estão no arquivo TCC-Resultados.docx.

A predição do valor do Bitcoin com análise de sentimentos, com base na métrica de regressão RMSE, não apresentou ganho de informação ou performance quando comparada a predição isolada da serie de preços do criptoativo. Este resultado atípico deve-se a base de dados de sentimentos não adequada. O tweet é uma rede mundial onde diversos usuários, com e sem entendimento técnico da variação do ativo, aliados a propagandas e mensagens de robôs de trade, interferem numa análise de sentimentos consciente e aderente acerca da real situação da moeda digital. 


### 4. Conclusões

Este trabalho propôs uma abordagem com diversas configurações de RNN-LSTM para previsão de preços do Bitcoin. Os modelos propostos passaram por ajustes de hiper-parâmetros e apresentaram um desempenho satisfatório nos dados de teste. A inclusão de análise de sentimento de rede social não agregou valor na performance da RNN para predição do criptoativo.


Acredita-se que a utilização de análise de sentimentos de uma rede social com público leigo não traz ganho para a rede. A própria rede do ativo possui um termómetro de medo e ganancia dos seus usuários. Assim, entende-se que a analise de preço do ativo em conjunto com os indicadores on-chain (stock-to-flow model, cursas de crescimento logarítmico, relative unrealized Profit/Loss e Puell Multiple), que caracterizam a movimentação do ativo dentro da rede, são indicadores que trarão ganhos de informação a predição do preço do Biticon.

A utilização de RNN-LSTM com mecanismos de atenção também é uma outra proposta para desenvolvimento de trabalhos futuros.



### 5. Referência

Bitcoin_tweets.csv - [(https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)]


---

Matrícula: 201.190.249

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
