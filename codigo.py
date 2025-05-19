import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixar stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Função para remover stopwords
def preprocess_text(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Carregar dados
df = pd.read_excel('./poemas.xlsx')

# Remover linhas com valores nulos na coluna 'Content'
df.dropna(subset=['Content'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Aplicar a remoção de stopwords
df['Content'] = df['Content'].apply(preprocess_text)

# Vetorização dos textos usando TF-IDF
tfvec = TfidfVectorizer(max_features=10000)
x = tfvec.fit_transform(df['Content'])

def find_similar(poem_text):
    # Preprocessar o poema de entrada
    poem_text = preprocess_text(poem_text)
    # Vetorizar o poema de entrada
    poem_vector = tfvec.transform([poem_text])
    # Calcular a similaridade com todos os poemas
    simi = [(i, cosine_similarity(poem_vector, x[i])[0][0]) for i in range(x.shape[0])]
    # Ordenar pela similaridade em ordem decrescente
    simi.sort(key=lambda x: x[1], reverse=True)
    # Extrair os índices e similaridades
    indices = [s[0] for s in simi[:10]]
    similaridades = [f"{s[1] * 100:.2f}%" for s in simi[:10]]  # Formatar como porcentagem
    # Criar DataFrame de retorno
    df_ret = df.iloc[indices, [0,1]].copy()
    df_ret["similarity"] = similaridades
    
    return df_ret

def find_similar_by_title(poem_title):
    resultado = df[df.Title.str.lower() == poem_title.lower()]
    
    while resultado.empty:
        print(f"Poema '{poem_title}' não encontrado.")
        poem_title = input("Por favor, insira um novo título de poema para buscar: ")
        resultado = df[df.Title.str.lower() == poem_title.lower()]
    
    poem_text = resultado.iloc[0]["Content"]
    print(f"\n📖 Comparando com o poema: {resultado.iloc[0]['Title']} - {resultado.iloc[0]['Author']}\n")
    
    return find_similar(poem_text)

# Exemplo de uso
poema_busca = input("Digite o título do poema para buscar similaridade: ")
resultado_similaridade = find_similar_by_title(poema_busca)
print(resultado_similaridade)
