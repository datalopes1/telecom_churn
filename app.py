import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scr.utils import load_data
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configurações do Streamlit
st.set_page_config(
    page_title="Telecom Churn",
    page_icon = "📶",
    layout = "wide"
)

st.title("📶 Case Telco Telecom")

# Dados
df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df['TotalCharges'] = df["TotalCharges"].replace(' ', np.nan)
df['TotalCharges'] = df["TotalCharges"].astype(float)

# Modelo de Classificação
model = joblib.load("models/classifier.pkl")


tab_report, tab_home, tab_analytics = st.tabs(["📝 Relatório","🤖 Preditor", "📊 Dashboard"])

with tab_report:
    st.title("Relatório de Análise")
    st.markdown("## Sumário Executivo")
    st.write(
        '''
        **Propósito**: Identificação de padrões e insights sobre os casos de churn no terceiro trimestre de 2024.

        **Insights-chave**:

        - Clientes com pouco tempo de contrato tem maior probabilidade de churn.
        - Contratos de renovação mensal tem maior taxa de churn.
        - Clientes que efetuam pagamentos automáticos possuem menor probabilidade de churn.

        ---
        '''
    )
    st.markdown("## 1. Introdução")
    st.write(
        '''
        **Problema**: A Telco Telecom precisa de um entendimento sobre os casos de Churn para interpretar seu estado atual na empresa, e traçar ações para sua diminuição.

        **Origem dos Dados**: 
        
        - CRM Interno (Tristre 3 - 2024)
        - 7043 consumidores, 21 variáveis 

        **Escopo**: Foram exploradras as relações entre tipo de contrato, tempo de relacionametno e outras variáveis com casos de Churn. 

        ---
        '''
    )
    st.markdown("## 2. Dados ")
    st.markdown(
        '''
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
    st.markdown("## 3. Análise e insights")
    st.markdown(
        '''
        ### 3.1. Taxa de Churn
        A Telco Telecom possui uma taxa de Churn de 26.53%, enquanto a média no setor é de 31.00%
        '''
    )
    img_path = "notebooks/plots/plot_2.png"
    st.image(img_path)

    st.markdown(
        '''
        ### 3.2. Churn x Tipo de Contrato
        Clientes com contrato de maior duração tem uma menor proporção de Churn quando comparados com as renovações mês-a-mês. 
        '''
    )
    img_path2 = "notebooks/plots/plot_4.png"
    st.image(img_path2)

    st.markdown(
        '''
        ### 3.3. Churn x Método de Pagamento
        Aqueles que optam por meios de pagamento automáticos possuem uma menor chance de Churn. 
        '''
    )
    img_path3 = "notebooks/plots/plot_5.png"
    st.image(img_path3)

    st.markdown(
        '''
        ### 3.4. Churn x Tempo de Relacionamento
        Clientes com maior tempo de relacionamento com a operadora tem uma menor probabilidade de deixar seus serviços.
        '''
    )
    img_path4 = "notebooks/plots/plot_6.png"
    st.image(img_path4)

    st.markdown(
        '''
        ### 3.5. Churn x Valor da Mensalidade
        Clientes quem pagam maiores mensalidades tem uma maior probabilidade de Churn.
        '''
    )
    img_path5 = "notebooks/plots/plot_7.png"
    st.image(img_path5)
    st.markdown("---")

    st.markdown(
        '''
        ## 4. Conclusões e recomendações
        - Clientes com menor tempo de relacionamento tem maior probabilidade de churn.
        - Aqueles que pagam maiores mensalidades também são os com maior probabilidade de churn.
        - Planos de renovação mensal são os com maior probabilidade de churn.
        - Pagamentos automáticos são os com menor probabilidade de churn.

        ### Recomendações
        - Criar ações para fidelização de clientes, como descontos e ofertas especiais
        - Melhorar o atendimento e oferecer vantagens exclusivas para clientes dos planos mais caros
        - Incentivar a efetivação de planos de renovação anual e com pagamentos automáticos

        É possível criar um plano de ação em forma de campanhas de marketing e novos planos de serviço para diminuir a taxa de Churn na Telco, algumas opções seriam (1) oferecer um plano anual com desconto caso o meio de pagamento escolhido seja Bank transnfer ou Credit card, (2) revisar preço e dar descontos para clientes com mais tempo de relacionamento que possuem planos mais caros para renovação anual. 
        '''
    )

with tab_home:
    st.subheader("🤖 Faça Predições")
    st.markdown("### Insira os Dados")

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
            probability = float(prob) * 100
            if prob > 0.40:
                st.markdown("### Este cliente é um possível caso de Churn.")
                st.write(f"A probabilidade de cancelamento é de {probability:.2f}%.")
            else:
                st.markdown("### Este cliente não é um possível caso de Churn.")
                st.write(f"A probabilidade de cancelamento é de {probability:.2f}%.")

with tab_analytics:
    st.subheader("📊 Dashboard Análitico")
# KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label = "Total de Clientes", value = df.shape[0])
    with col2:
        st.metric(label = "Taxa de Churn", value = f"{(df['Churn'].mean() * 100):.2f} %")
    with col3:
        st.metric(label = "Lifetime Value Médio", value = f"$ {df['TotalCharges'].mean():.2f}")
    with col4:
        st.metric(label = "Tempo Médio de Contrato", value = f"{df['tenure'].mean():.0f} meses")

    # Gráficos
    col1, col2 = st.columns(2)

    ## Churn por Método de Pagamento
    pm = pd.DataFrame(df[df['Churn'] == 1]['PaymentMethod'].value_counts().reset_index())
    pm.columns = ['PaymentMethod', 'count']

    fig = px.bar(pm, 
                x = 'PaymentMethod', 
                y = 'count', 
                title = 'Churn por Método de Pagamento')

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = 'Método',
        yaxis_title = 'Quantidade',
        showlegend = False
    )

    ## Churn por Tipo de Contrato
    ct = pd.DataFrame(df[df['Churn'] == 1]['Contract'].value_counts().reset_index())
    fig2 = px.bar(ct, x = 'Contract', y = 'count', title = "Churn por Tipo de Contrato")

    fig2.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = 'Tipo',
        yaxis_title = 'Quantidade',
        showlegend = False
    )

    ## Churn x Tempo de Relacionamento
    fig3 = px.histogram(
        df, 
        x = 'tenure', 
        color = 'Churn', 
        barmode = 'overlay', 
        title = "Distribuição de Churn por Tempo de Relacionamento",
        labels = {'tenure': 'Tempo de Relacionamento (Meses)', 'count': 'Proporção (%)'}
    )

    fig3.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = 'Meses',
        yaxis_title = 'Contagem',
        bargap = 0.1
    )

    ## TotalCharges x Tenure

    fig4 = px.histogram(df, 
                    x = 'MonthlyCharges',
                    color = 'Churn',
                    barmode = 'overlay',
                    title = "Distribuição de Churn por Valor da Mensalidade")

    fig4.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = '$',
        yaxis_title = 'Contagem',
        bargap = 0.1
    )

    with col1:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)