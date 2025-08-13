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
# options.add_argument('--headless')  # opcional
service = Service(r'chromedriver-win64/chromedriver-win64/chromedriver.exe')  # ajuste o caminho
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

# ---------- LINK ESPECÍFICO ----------
link = "https://www.reclameaqui.com.br/peugeot-do-brasil/recall-realizado-e-com-pendencia-no-senatran-ctd_2gVngFbn4jwQ6mXp/"
driver.get(link)
time.sleep(8)

# ---------- COLETAR OS DADOS ----------
titulo = get_by_class("sc-lzlu7c-3")

# Captura somente o nome da marca
try:
    marca = driver.find_element(By.CSS_SELECTOR, ".sc-lzlu7c-5 a").text
except:
    marca = None

local_data = get_all_by_class("sc-lzlu7c-6")
localizacao = local_data[0] if len(local_data) > 0 else None
data = local_data[1] if len(local_data) > 1 else None
id_reclamacao = get_by_class("sc-lzlu7c-12")

# Captura apenas os tópicos corretos
try:
    topicos_container = driver.find_element(By.CSS_SELECTOR, "ul.sc-1dmxdqs-0")
    topicos = [e.text for e in topicos_container.find_elements(By.CSS_SELECTOR, "li[data-testid^='listitem-'] a")]
except:
    topicos = []


texto = get_by_class("sc-lzlu7c-17")
status = get_by_class("sc-1a60wwz-1")



novo_dado = [{
    "TITULO": titulo,
    "MARCA": marca,
    "LOCALIZACAO": localizacao,
    "DATA": data,
    "ID": id_reclamacao,
    "TOPICOS": topicos,
    "TEXTO": texto,
    "STATUS": status,
    "LINK": link
}]

# ---------- FECHAR DRIVER ----------
driver.quit()

# ---------- SALVAR/ACRESCENTAR RESULTADOS ----------
arquivo_excel = "peugeot.xlsx"
df_novo = pd.DataFrame(novo_dado)

if os.path.exists(arquivo_excel):
    df_existente = pd.read_excel(arquivo_excel)
    # Garante que as colunas fiquem na mesma ordem
    df_novo = df_novo.reindex(columns=df_existente.columns)
    df_final = pd.concat([df_existente, df_novo], ignore_index=True, sort=False)
else:
    df_final = df_novo

df_final.to_excel(arquivo_excel, index=False)
print(f"Coleta concluída. Dados salvos/atualizados em {arquivo_excel}")
