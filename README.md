# 📶 Telecom Churn - Predição e Análise de Cancelamento
Um projeto de análise e previsão de churn em telecomunicações usando Streamlit, Machine Learning e Visualização de Dados.

## 📜 Sumário
1. [📌 Sobre o Projeto](#-sobre-o-projeto)
2. [⚙️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
3. [🚀 Como Executar](#-como-executar)
4. [📊 Estrutura do Projeto](#-estrutura-do-projeto)
5. [🗒️ Licença](#️-licença)
6. [📞 Contato](#-contato)


## 📌 Sobre o Projeto
Este projeto tem como objetivo identificar padrões e prever a **probabilidade de cancelamento** de clientes de uma empresa de telecomunicações.

Com um relatório e dashboard, é possível explorar os principais fatores que influenciam o churn e realizar previsões com um modelo de Machine Learning utilizando o preditor.

## ⚙️ Tecnologias Utilizadas
Este projeto foi desenvolvido utilizando:
- 🐍 **Python 3.12+** 
- 📊 **Streamlit (Interface)**
- 🔢 **Pandas & NumPy (Manipulação de Dados)**
- 🤖 **Scikit-learn, CatBoost (Machine Learning)**
    - 🔭 **Optuna, Feature Engine, Category Encoders (Otimização e Feature Engineering)**
- 📈 **Plotly (Visualização de Dados)**
- 💾 **Joblib (Manipulação do Modelo)**

## 🚀 Como Executar
Acesse a aplicação web no [Streamlit Cloud](https://telcotelecom-churn.streamlit.app/). 
#### Pré-requisitos
- Python 3.12+
- Git

#### Execução 
1️⃣ **Clone o repositório**
```bash
git clone https://github.com/datalopes1/telecom_churn.git
cd telecom_churn
```

2️⃣ **Crie e ative um ambiente virutal (recomendado)**
```bash
python -m venv .venv
source .venv/bin/activate  # Mac e Linux
.venv\Scripts\activate  # Windows
```
3️⃣ **Instale as dependências**
```bash
pip install -r requirements.txt
```

4️⃣ **Execute o projeto**
```bash
streamlit run app.py
```

## 📊 Estrutura do Projeto
```plaintext
telecom-churn/
│-- data/                       # Dados do projeto
|   ├── raw/                    # Dados brutos
|   ├── processed/              # Dados tratados
|-- models/                     # Modelos treinados
|-- notebooks
|   ├── plots/                  # Arquivos .png gerados na EDA
|   ├── eda.ipynb               # Notebook de Análise Exploratória de Dados
|   ├── modeling.ipynb          # Notebook de Construção do modelo de ML
|-- scr/                        # Scripts 
|   ├── __init__.py
|   ├── data_preprocessing.py   # Script de funções de pré-processamento
|   ├── evaluate_model.py       # Script de avaliação do modelo
|   ├── predict.py              # Script para gerar predições
|   ├── train_model.py          # Script de treinamento do modelo
|   ├── utils.py                # Script com funções auxiliares
|-- .gitignore                  # Arquivos ignorados pelo Git
|-- app.py                      # Aplicação do Streamlit
|-- LICENSE.md                  # Licença
|-- poetry.lock                 # Configuração do Poetry e dependências do projeto
|-- pyproject.toml              # Versões exatas das dependências instaladas
|-- README.md                   # Documentação do projeto
|-- requirements.txt            # Lista de dependências
```

## 🗒️ Licença
Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## 📞 Contato
- LinkedIn: https://www.linkedin.com/in/andreluizls1
- Portfolio: https://sites.google.com/view/datalopes1
- E-mail: datalopes1@proton.me
