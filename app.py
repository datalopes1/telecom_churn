# --------------- CONFIGURAÇÃO INICIAL ---------------
# Importação de bibliotecas
import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuração do ambiente
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configurações do Streamlit
st.set_page_config(
    page_title="Telecom Churn",
    page_icon = "📶",
    layout = "wide"
)

st.title("📶 Case Telco Telecom")

# --------------- FUNÇÕES ---------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    
    return df

def plot_bar(data, x, y, color, title, barmode, xlabel, ylabel):
    fig = px.bar(
        data,
        x = x,
        y = y,
        color = color,
        title = title,
        barmode = barmode,
        labels = {x: xlabel, y: ylabel},
        color_discrete_sequence=['#0f4c5c', '#9a031e']
    )

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = xlabel,
        yaxis_title = ylabel,
        showlegend = False
    )

    return fig

def plot_hist(data, x, color, title, xlabel, ylabel):
    fig = px.histogram(
        data,
        x = x,
        color = color,
        barmode = 'overlay',
        title = title,
        labels = {x: xlabel, 'count': ylabel},
        color_discrete_sequence=['#0f4c5c', '#9a031e']
    )

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = xlabel,
        yaxis_title = ylabel,
        bargap = 0.05
    )

    return fig

# --------------- DADOS ---------------

df = load_data()
model = joblib.load("models/classifier.pkl")

# --------------- TABS ---------------
tab_report, tab_pred, tab_analytics = st.tabs(["📝 Relatório","🤖 Preditor", "📊 Dashboard"])

# ------------- RELATÓRIO DE ANÁLISE -------------
with tab_report:
    st.title("Relatório de Análise")
    st.markdown(
        '''
        ## Sumário Executivo

        **Propósito**: Identificação de padrões e insights sobre os casos de churn no terceiro trimestre de 2024.

        **Insights-chave**:

        - Clientes com pouco tempo de contrato tem maior probabilidade de churn.
        - Contratos de renovação mensal tem maior taxa de churn.
        - Clientes que efetuam pagamentos automáticos possuem menor probabilidade de churn.

        ---

        ## 1. Introdução

        **Problema**: A Telco Telecom precisa de um entendimento sobre os casos de Churn para interpretar seu estado atual na empresa, e traçar ações para sua diminuição.

        **Origem dos Dados**: 
        
        - CRM Interno (Tristre 3 - 2024)
        - 7043 consumidores, 21 variáveis 

        **Escopo**: Foram exploradras as relações entre tipo de contrato e sua duração, tipo de serviço, lifetime value, entre outras variáveis com os casos de Churn. 

        ---

        ## 2. Dados 

        ### Estrutura do Dataset
        |Feature|Descrição|
        |---|---|
        |`customerID`|Identificador único dos clientes|
        |`gender`|Gênero|
        |`SeniorCitizen`|É idoso?|
        |`Partner`|Possui parceiro?|
        |`Dependents`|Possui dependentes?|
        |`tenure`|Tempo de relacionamento (em meses)|
        |`PhoneService`|Possui serviço telefonico?|
        |`MultipleLines`|Possui multiplas linhas?(Sim, não, não possui serviço telefonico)|
        |`InternetService`|Provedor de serviços de internet (DSL, Fibra ou não)|
        |`OnlineSecurity`|Possui seguro online?|
        |`OnlineBackup`|Possui backup online?|
        |`DeviceProtection`|Possui proteção do dispositivo?|
        |`TechSupport`|Tem suporte técnico?|
        |`StreamingTV`|Possui streaming de TV?|
        |`StreamingMovies`|Possui streaming de Filmes?|
        |`Contract`|Tipo de contrato(mês-a-mês, anual ou bi-anual)|
        |`PaperlessBilling`|Recebe boletos?|
        |`PaymentMethod`|Método de pagamento|
        |`MonthlyCharges`|Taxa de serviço|
        |`TotalCharges`|Total pago pelo cliente|
        |`Churn`|Alvo|

        ### Qualidade dos Dados
        - Valores nulos: `TotalCharges` tem um total de 11 valores nulos.
        - Outliers: Não foram encontrados valores extremos.

        ### Limpeza e manipulação
        - Para a análise foi necessária o ajuste da coluna `SeniorCitizen` de binário numérico (0, 1) para em forma de texto ('Yes' e 'No').
        - A coluna `TotalCharges` tinha problemas com seu preenchimento que foi corrigido.

        ---
        '''
    )
    st.markdown(
        '''
        ## 3. Análise e insights
        ### 3.1. Retenção de clientes
        '''
    )

    churn = df['Churn'].value_counts(normalize = True).reset_index()
    churn['proportion'] = (churn['proportion'] * 100).round(2)

    st.plotly_chart(
        plot_bar(
            churn, 
            x = 'Churn', 
            y = 'proportion', 
            color = 'Churn', 
            title = 'Distribuição da Retenção de Clientes', 
            barmode = 'relative',
            xlabel = 'Churn', 
            ylabel = 'Proporção'
        ), 
        use_container_width=True
    )
    st.write(
        '''
        A retenção de clientes é um dos grandes desafios no setor de telefonia, a Telco Telecom mantém uma taxa de retenção de 73.46% em seus contratos, com uma média de 32 meses, ou seja, pouco mais de dois anos na duração do relacionamento cliente/empresa.

        O tipo de contrato firmado com maior frequência é de renovação mensal, a forma de pagamento mais comum é o eCheck (electronic check). 

        ### 3.2. Serviços de internet
        '''
    )

    internet = df.groupby(['Churn', 'InternetService']).agg(Contagem = ('InternetService', 'count')).reset_index()

    st.plotly_chart(
        plot_bar(
            internet, 
            x = 'InternetService', 
            y = 'Contagem', 
            color = 'Churn', 
            title = 'Serviço de Internet x Churn', 
            barmode = 'group',
            xlabel = 'Serviço', 
            ylabel = 'Quantidade'
        ), 
        use_container_width=True
    )
    st.markdown(
        '''
        O serviço de fibra ótica é o segundo mais utilizando entre os de internet mas possui uma alta proporção de churn se comparado ao DSL e clientes que não possuem internet contratada.

        ### 3.3. Tipo de Contrato
        '''
    )

    contract = df.groupby(['Churn', 'Contract']).agg(Contagem = ('Contract', 'count')).reset_index()
    contract_plot = plot_bar(
            contract,
            x = 'Contract',
            y = 'Contagem',
            color = 'Churn',
            title = 'Tipo de Contrato x Churn',
            barmode = 'group',
            xlabel = 'Tipo',
            ylabel = 'Quantidade'
            )

    st.plotly_chart(contract_plot, use_container_width=True)
    st.markdown(
        '''
        O contrato de renovação mensal é o mais frequente e o com maior proporção de Churn, os outros tipos (anual, e bi-anual) tem uma taxa proporcionalmente muito baixa e podem ser chave para o aumento da retenção.
        
        ### 3.4. Método de Pagamento
        '''
    )

    pay = df.groupby(['Churn', 'PaymentMethod']).agg(Contagem = ('PaymentMethod', 'count')).reset_index()
    pay_plot = plot_bar(
            pay,
            x = 'PaymentMethod',
            y = 'Contagem',
            color = 'Churn',
            title = 'Método de Pagamento x Churn',
            barmode = 'group',
            xlabel = 'Método',
            ylabel = 'Quantidade'
        )
    st.plotly_chart(pay_plot, use_container_width=True)

    st.markdown(
        '''
        O eCheck é o método mais utilizado e o com maior proporção de Churn, um fator que chama a tenção é a baixíssima quantidade de casos em meios de pagamento automático, o que é outro ponto chave para planejar ações para aumentar a retenção de clientes.

        ### 3.4. Tempo de Relacionamento
        '''
    )
    tenure_plot = plot_hist(
            df,
            x = 'tenure',
            color = 'Churn',
            title = 'Distribuição de Churn por Tempo de Relacionamento',
            xlabel = 'Meses',
            ylabel = 'Quantidade'
        )
    st.plotly_chart(tenure_plot, use_container_width=True)
    
    st.markdown(
        '''
        Quanto mais tempo passamos consumindo um serviço, seja por comodidade ou apego, mais dificilmente deixaremos ele. Mas no início de contrato a atenção aos detalhes é maior, então após testar a hipotése de — clientes mais recentes tem maior probabilidade de se tornarem Churners, o comportamento foi confirmado.

        ### 3.5. Fatura Mensal
        '''
    )
    charges_plot = plot_hist(
            df,
            x = 'MonthlyCharges',
            color = 'Churn',
            title = 'Distribuição de Churn por Valor da Mensalidade',
            xlabel = 'USD',
            ylabel = 'Quantidade'
        )
    st.plotly_chart(charges_plot, use_container_width=True)
    st.markdown(
        '''
        Assim como o comportamento em relação a contratos recentes, decidi também testar a hipotése de contratos com maiores faturas mensais estarem sob maior proabilidade de ser um casos de Churn, clientes dispostos a pagar serviços mais caros também irão exigir melhor qualidade em sua prestação — o que se tornou mais uma hipotése confirmada.

        ---
        
        ## 4. Recomendações
        Com a análise concluída as recomendações para o aumento da retenção de clientes foram as seguintes:

        - Criar ações para fidelização de clientes com ofertas e descontos especiais.
        - Buscar uma melhora no atendimento e oferecer vantagens nos contratos com maiores faturas mensais.
        - Incentivar a efetivação de planos de contrato anual, e de pagamentos por vias automáticas.

        Com isso em mente também sugiro criação de campanhas de marketing e novos planos de serviço na Telco, algumas opções seriam (1) oferecer um plano anual com desconto caso o método de pagamento escolhido seja um dos automáticos, e (2) revisar preços de contratos de consumidores com maior tempo de relacionamento e oferecer vatangens na renovação para planos de duração mais longa. 

        ---

        ## 5. Conclusões
        A retenção média no setor é de 69%[*](https://customergauge.com/blog/average-churn-rate-by-industry), marca superada pela Telco Telecom, o que mostra um bom desempenho no terceiro trimestre mas apesar disso foram detectados vários pontos de melhora que podem aumentar a retenção dos clientes como a atenção aos planos de maior duração, e as formas  de pagamento automáticos. O bom desempenho pode ser melhorado através de ações retenção de novos clientes (contratos com menos de 6 meses), e de transição de clientes que atualmente possuem renovação mensal para planos mais longos.

        '''
    )

# -------- PREDITOR DE CHURN ---------
with tab_pred:
    st.header("🤖 Preditor de Cancelamento de Contratos")
    st.subheader("Insira os Dados e Calcule a Probabilidade")

    gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
    senior = st.selectbox("Idoso", ["Sim", "Não"])
    partner = st.selectbox("Possui parceiro", ["Sim", "Não"])
    dependents = st.selectbox("Dependentes", ["Sim", "Não"])
    tenure = st.number_input("Tempo de Contrato em Meses", min_value = 1, max_value = df['tenure'].max(), value = 1)
    phoneservice = st.selectbox("Serviço Telefônico", ["Sim", "Não"])
    lines = st.selectbox("Multiplas Linhas", ["Sim", "Não", "Não possui linha"])
    internetservice = st.selectbox("Serviço de Internet", ["DSL", "Fibra ótica", "Não"])
    onlinesec = st.selectbox("Segurança Online", ["Sim", "Não", "Não possui internet"])
    onlinebackup = st.selectbox("Backup Online", ["Sim", "Não", "Não possui internet"])
    deviceprotection = st.selectbox("Proteção de Dispositivo", ["Sim", "Não", "Não possui internet"])
    techsupport = st.selectbox("Suporte Técnico", ["Sim", "Não", "Não possui internet"])
    streamingtv = st.selectbox("Streaming de TV", ["Sim", "Não", "Não possui internet"])
    streamingmovies = st.selectbox("Streaming de Filmes", ["Sim", "Não", "Não possui internet"])
    contract = st.selectbox("Tipo de Contrato", list(df["Contract"].unique()))
    paperless = st.selectbox("Fatura sem Papel", ["Sim", "Não"])
    paymethod = st.selectbox("Método de Pagamento", list(df["PaymentMethod"].unique()))
    monthlycharge = st.slider("Mensalidade", 20, 120)

    input_features = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': lines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesec,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': paymethod,
        'MonthlyCharges': monthlycharge,
        'TotalCharges': tenure * monthlycharge
    }

    input_df = pd.DataFrame(input_features, index = [0])
    input_df["gender"] = input_df["gender"].map({"Masculino": "Male", "Feminino": "Female"})
    input_df["SeniorCitizen"] = input_df["SeniorCitizen"].map({"Sim": "Yes", "Não": "No"})
    input_df["Partner"] = input_df["Partner"].map({"Sim": "Yes", "Não": "No"})
    input_df["Dependents"] = input_df["Dependents"].map({"Sim": "Yes", "Não": "No"})
    input_df["PhoneService"] = input_df["PhoneService"].map({"Sim": "Yes", "Não": "No"})
    input_df["MultipleLines"] = input_df["MultipleLines"].map({"Sim": "Yes", "Não": "No", "Não possui linha": "No phone service"})
    input_df["InternetService"] = input_df["InternetService"].map({"Fibra ótica": "Fiber optic", "Não": "No"})
    input_df["OnlineSecurity"] = input_df["OnlineSecurity"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["OnlineBackup"] = input_df["OnlineBackup"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["DeviceProtection"] = input_df["DeviceProtection"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["TechSupport"] = input_df["TechSupport"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["StreamingTV"] = input_df["StreamingTV"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["StreamingMovies"] = input_df["StreamingMovies"].map({'Não': 'No', 'Sim': 'Yes', 'Não possui internet':'No internet service'})
    input_df["PaperlessBilling"] = input_df["PaperlessBilling"].map({"Sim": "Yes", "Não": "No"})

    with st.container():
        if st.button("Resultado"):
            prob = model.predict_proba(input_df)[:,1][0]
            if prob > 0.40:
                st.markdown("## Alto Potencial de Cancelamento")
                st.error(f"Probabilidade de {prob:.2%}")
            else:
                st.markdown("## Baixo Potencial de Cancelamento")
                st.success(f"Probabilidade de {prob:.2%} ")

# ------------- DASHBOARD ANALÍTICO -------------
with tab_analytics:
    st.subheader("📊 Dashboard Análitico")
    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label = "Total de Clientes", value = df.shape[0])
    with col2:
        st.metric(label = "Taxa de Churn", value = f"{(df['Churn'].map({'No': 0, 'Yes': 1}).mean() * 100):.2f} %")
    with col3:
        st.metric(label = "Lifetime Value Médio", value = f"$ {df['TotalCharges'].mean():.2f}")
    with col4:
        st.metric(label = "Tempo Médio de Contrato", value = f"{df['tenure'].mean():.0f} meses")

    # Gráficos
    col1, col2 = st.columns(2)

    pm = df[df['Churn'] == 'Yes'].groupby(['Churn', 'PaymentMethod']).agg(Quantidade = ('PaymentMethod', 'count')).reset_index()
    ct = df[df['Churn'] == 'Yes'].groupby(['Churn', 'Contract']).agg(Quantidade = ('Contract', 'count')).reset_index()

    with col1:
        st.plotly_chart(
            plot_bar(
                pm,
                x = 'PaymentMethod',
                y = 'Quantidade',
                color = None,
                title = 'Churn por Método de Pagamento',
                barmode = 'relative',
                xlabel='Método',
                ylabel='Quantidade'
        ),
        use_container_width=True
        )
        st.plotly_chart(
            plot_hist(
            df,
            x = 'tenure',
            color = 'Churn',
            title = 'Churn por Tempo de Relacionamento',
            xlabel = 'Meses',
            ylabel = 'Quantidade'
        ),
        use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_bar(
                ct,
                x = 'Contract',
                y = 'Quantidade',
                color = None,
                title = 'Churn por Tipo de Contrato',
                barmode = 'relative',
                xlabel='Tipo',
                ylabel='Quantidade'
        ),
        use_container_width=True
        )
        st.plotly_chart(
            plot_hist(
            df,
            x = 'MonthlyCharges',
            color = 'Churn',
            title = 'Churn por Valor da Mensalidade',
            xlabel = 'USD',
            ylabel = 'Quantidade'
        ),
        use_container_width=True
        )