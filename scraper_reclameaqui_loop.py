from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import os

# ---------- CONFIGURAÇÃO DO SELENIUM ----------
options = Options()
options.add_argument('--start-maximized')
#options.add_argument('--headless')  # opcional
service = Service(r'chromedriver-win64\chromedriver-win64\chromedriver.exe')  # ajuste o caminho
driver = webdriver.Chrome(service=service, options=options)

# ---------- FUNÇÕES AUXILIARES ----------
def get_by_class(cl):
    try:
        return driver.find_element(By.CLASS_NAME, cl).text
    except:
        return None

def get_all_by_class(cl):
    try:
        return [e.text for e in driver.find_elements(By.CLASS_NAME, cl)]
    except:
        return []

# ---------- LER LINKS DO EXCEL ----------
df_links = pd.read_excel("links.xlsx")
# 3ª coluna -> índice 2
lista_urls = df_links["LINK"].dropna().tolist()

# ---------- LISTA PARA ARMAZENAR TODOS OS DADOS ----------
dados = []

# ---------- LOOP PRINCIPAL ----------
for url_principal in lista_urls:
    print(f"Acessando URL principal: {url_principal}")
    driver.get(url_principal)
    time.sleep(8)

    # PEGAR TODOS OS SUBLINKS
    links_reclamacoes = []
    elementos = driver.find_elements(By.ID, "site_bp_lista_ler_reclamacao")
    for elem in elementos:
        try:
            link = elem.get_attribute("href")
            if link and link not in links_reclamacoes:
                links_reclamacoes.append(link)
        except:
            continue

    print(f"Encontrados {len(links_reclamacoes)} sublinks nesta página.")

    # LOOP NAS RECLAMAÇÕES
    for link in links_reclamacoes:
        driver.get(link)
        time.sleep(8)

        titulo = get_by_class("sc-lzlu7c-3")
        try:
            marca = driver.find_element(By.CSS_SELECTOR, ".sc-lzlu7c-5 a").text
        except:
            marca = None
        local_data = get_all_by_class("sc-lzlu7c-6")
        localizacao = local_data[0] if len(local_data) > 0 else None
        data = local_data[1] if len(local_data) > 1 else None
        id_reclamacao = get_by_class("sc-lzlu7c-12")
        try:
            topicos_container = driver.find_element(By.CSS_SELECTOR, "ul.sc-1dmxdqs-0")
            topicos = [e.text for e in topicos_container.find_elements(By.CSS_SELECTOR, "li[data-testid^='listitem-'] a")]
        except:
            topicos = []
        texto = get_by_class("sc-lzlu7c-17")
        status = get_by_class("sc-1a60wwz-1")

        dados.append({
            "TITULO": titulo,
            "MARCA": marca,
            "LOCALIZACAO": localizacao,
            "DATA": data,
            "ID": id_reclamacao,
            "TOPICOS": ", ".join(topicos),
            "TEXTO": texto,
            "STATUS": status,
            "LINK": link
        })

# ---------- FECHAR DRIVER ----------
driver.quit()

# ---------- SALVAR RESULTADOS ----------
arquivo_excel = "dados.xlsx"
df_novo = pd.DataFrame(dados)

if os.path.exists(arquivo_excel):
    df_existente = pd.read_excel(arquivo_excel)
    df_novo = df_novo.reindex(columns=df_existente.columns)
    df_final = pd.concat([df_existente, df_novo], ignore_index=True, sort=False)
else:
    df_final = df_novo

df_final.to_excel(arquivo_excel, index=False)
print("Coleta concluída e dados salvos em dados.xlsx")
