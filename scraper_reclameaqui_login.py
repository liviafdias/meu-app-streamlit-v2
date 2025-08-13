from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import json
import time
import os
import random

# ================== CONFIGURAÇÃO BROWSER ==================
options = Options()
options.add_argument("--start-maximized")
# options.add_argument("--headless=new")  # Evite headless em sites com Cloudflare/CAPTCHA
# Opcional: user-agent realista
# options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

service = Service(r'chromedriver-win64\chromedriver-win64\chromedriver.exe')  # ajuste caminho
driver = webdriver.Chrome(service=service, options=options)
WAIT = WebDriverWait(driver, 25)

HOME_URL = "https://www.reclameaqui.com.br/"
LOGIN_URL = "https://www.reclameaqui.com.br/login/consumidor/"
COOKIES_PATH = "cookies.json"

# ================== HELPERS ==================
def human_sleep(a=0.9, b=1.8):
    time.sleep(random.uniform(a, b))

def long_human_sleep(a=3.0, b=6.0):
    time.sleep(random.uniform(a, b))

def save_cookies():
    try:
        with open(COOKIES_PATH, "w", encoding="utf-8") as f:
            json.dump(driver.get_cookies(), f, ensure_ascii=False, indent=2)
        print("💾 Cookies salvos.")
    except Exception as e:
        print(f"⚠️ Erro ao salvar cookies: {e}")

def load_cookies():
    if not os.path.exists(COOKIES_PATH):
        return False
    try:
        driver.get(HOME_URL)
        human_sleep()
        with open(COOKIES_PATH, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        for c in cookies:
            if "sameSite" in c and c["sameSite"] is None:
                c["sameSite"] = "Lax"
            try:
                driver.add_cookie(c)
            except Exception:
                pass
        print("🍪 Cookies carregados.")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao carregar cookies: {e}")
        return False

def detect_challenge():
    src = driver.page_source.lower()
    signals = [
        "cf-challenge", "cloudflare", "captcha", "hcaptcha", "g-recaptcha",
        "attention required", "are you human", "temporarily blocked"
    ]
    return any(s in src for s in signals)

def pause_for_challenge():
    print("⚠️ Desafio/CAPTCHA detectado. Resolva na janela do navegador.")
    try:
        input("Depois de resolver e a página carregar, pressione ENTER aqui para continuar...")
    except EOFError:
        print("⏳ Sem input interativo; aguardando 90s para você resolver manualmente...")
        time.sleep(90)
    # após resolver, consolidar sessão
    save_cookies()

def wait_ready(css_hint="body", timeout=25):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_hint))
        )
    except Exception:
        pass

def navigate_with_retries(url, css_hint="body", max_retries=3, backoff_base=2.0):
    """
    Abre URL, detecta Cloudflare/CAPTCHA e pede intervenção manual se necessário.
    Faz algumas tentativas com backoff. Retorna True se carregou, False se desistiu.
    """
    for attempt in range(1, max_retries + 1):
        driver.get(url)
        human_sleep()
        wait_ready(css_hint)

        if detect_challenge():
            pause_for_challenge()          # você resolve e salvamos cookies
            wait_ready(css_hint, timeout=40)
            if not detect_challenge():
                print(f"✅ Desafio resolvido na tentativa {attempt} para: {url}")
                return True
        else:
            # sem desafio — OK
            return True

        # ainda tem desafio? aguarda e tenta de novo
        sleep_s = backoff_base ** attempt + random.uniform(0.5, 1.5)
        print(f"⏳ Ainda bloqueado. Aguardando {sleep_s:.1f}s e tentando novamente...")
        time.sleep(sleep_s)

    print(f"❌ Não foi possível carregar (após {max_retries} tentativas): {url}")
    return False

def get_text_by_class(cl):
    try:
        el = WAIT.until(EC.presence_of_element_located((By.CLASS_NAME, cl)))
        return el.text.strip()
    except Exception:
        return None

def get_all_texts_by_class(cl):
    try:
        els = WAIT.until(EC.presence_of_all_elements_located((By.CLASS_NAME, cl)))
        return [e.text.strip() for e in els if e.text.strip()]
    except Exception:
        return []

def collect_sublinks():
    sublinks = set()

    # seletor original por classe
    for a in driver.find_elements(By.CLASS_NAME, "sc-1pe7b5t-0"):
        try:
            href = a.get_attribute("href")
            if href:
                sublinks.add(href)
        except Exception:
            pass

    # fallback: qualquer link de reclamação
    for a in driver.find_elements(By.CSS_SELECTOR, 'a[href*="/reclamacao/"]'):
        try:
            href = a.get_attribute("href")
            if href:
                sublinks.add(href)
        except Exception:
            pass

    return list(sublinks)

def ensure_session():
    # Tenta recuperar sessão por cookies
    loaded = load_cookies()
    if loaded and navigate_with_retries(HOME_URL):
        print("✅ Sessão recuperada via cookies.")
        return

    # Login manual
    print("🔐 Iniciando login manual...")
    if not navigate_with_retries(LOGIN_URL, css_hint='input[type="email"]'):
        raise RuntimeError("Não consegui abrir a página de login (bloqueado).")

    print("👉 Faça login manualmente (e resolva o CAPTCHA) no navegador.")
    try:
        input("Quando terminar o login e a conta estiver logada, pressione ENTER aqui para continuar...")
    except EOFError:
        print("⏳ Sem input; aguardando 120s para você concluir o login...")
        time.sleep(120)

    save_cookies()
    navigate_with_retries(HOME_URL)
    print("✅ Sessão autenticada.")

# ================== FLUXO PRINCIPAL ==================
def main():
    try:
        ensure_session()

        # Lê lista de páginas principais
        df_links = pd.read_excel("links.xlsx")
        links_principais = df_links["LINK"].dropna().tolist()

        dados = []
        retry_queue = []  # sublinks que falharem, reprocessamos no fim

        for url_principal in links_principais:
            print(f"\n📌 Coletando sublinks de: {url_principal}")

            if not navigate_with_retries(url_principal, css_hint="body"):
                print("❌ Pulando página principal (bloqueio persistente).")
                continue

            # simula comportamento humano: scroll em etapas
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, document.body.scrollHeight/3);")
                human_sleep()

            sublinks = collect_sublinks()
            print(f"   🔗 {len(sublinks)} sublinks encontrados.")

            for link in sublinks:
                # Respeita um intervalo entre navegações
                long_human_sleep(1.5, 3.5)

                ok = navigate_with_retries(link, css_hint="body", max_retries=3)
                if not ok:
                    retry_queue.append((link, url_principal))
                    continue

                # Espera elementos mais específicos, se possível (opcional)
                # try: WAIT.until(EC.any_of(
                #         EC.presence_of_element_located((By.CLASS_NAME, "sc-lzlu7c-3")),
                #         EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
                #     ))
                # except Exception: pass

                titulo = get_text_by_class("sc-lzlu7c-3")
                marca = get_text_by_class("sc-lzlu7c-5")
                local_data = get_all_texts_by_class("sc-lzlu7c-6")
                localizacao = local_data[0] if len(local_data) > 0 else None
                data = local_data[1] if len(local_data) > 1 else None
                id_reclamacao = get_text_by_class("sc-lzlu7c-12")
                topicos = get_all_texts_by_class("sc-1s8uljb-0")
                texto = get_text_by_class("sc-lzlu7c-17")
                status = get_text_by_class("sc-1a60wwz-1")

                dados.append({
                    "Título": titulo,
                    "Marca": marca,
                    "Localização": localizacao,
                    "Data": data,
                    "ID": id_reclamacao,
                    "Tópicos": ", ".join(topicos),
                    "Texto": texto,
                    "Status": status,
                    "Link Reclamação": link,
                    "Link Página": url_principal
                })

        # Reprocessa sublinks problemáticos (uma rodada extra)
        if retry_queue:
            print(f"\n🔁 Retentando {len(retry_queue)} sublinks problemáticos...")
            second_round = []
            for link, url_principal in retry_queue:
                long_human_sleep(2.0, 4.0)
                ok = navigate_with_retries(link, css_hint="body", max_retries=2, backoff_base=3.0)
                if not ok:
                    second_round.append((link, url_principal))
                    continue

                titulo = get_text_by_class("sc-lzlu7c-3")
                marca = get_text_by_class("sc-lzlu7c-5")
                local_data = get_all_texts_by_class("sc-lzlu7c-6")
                localizacao = local_data[0] if len(local_data) > 0 else None
                data = local_data[1] if len(local_data) > 1 else None
                id_reclamacao = get_text_by_class("sc-lzlu7c-12")
                topicos = get_all_texts_by_class("sc-1s8uljb-0")
                texto = get_text_by_class("sc-lzlu7c-17")
                status = get_text_by_class("sc-1a60wwz-1")

                dados.append({
                    "Título": titulo,
                    "Marca": marca,
                    "Localização": localizacao,
                    "Data": data,
                    "ID": id_reclamacao,
                    "Tópicos": ", ".join(topicos),
                    "Texto": texto,
                    "Status": status,
                    "Link Reclamação": link,
                    "Link Página": url_principal
                })

            if second_round:
                print(f"⚠️ {len(second_round)} sublinks permaneceram bloqueados e foram pulados.")

        # SALVA EXCEL
        arquivo_excel = "dados.xlsx"
        df_novo = pd.DataFrame(dados)

        if os.path.exists(arquivo_excel):
            df_existente = pd.read_excel(arquivo_excel)
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_excel(arquivo_excel, index=False)
        print("\n✅ Coleta concluída e dados salvos em dados.xlsx")

    finally:
        driver.quit()

# ================== EXECUÇÃO ==================
if __name__ == "__main__":
    main()
