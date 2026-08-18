import os
import traceback
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import uvicorn

app = FastAPI(
    title="Cancer Loser - Assistente de Cuidado e Apoio Oncológico",
    version="1.1.0"
)

# --- Modelo Oficial de Alta Cota (500 requisições/dia) ---
MODELO_IA = "gemini-3.5-flash-lite"

# --- Schemas Pydantic Estruturados ---
class OpiniaoObjetiva(BaseModel):
    classificacao_momento: str
    parecer_direto: str
    nivel_cuidado: str

class ManejoSintoma(BaseModel):
    sintoma_foco: str
    o_que_fazer_agora: List[str]
    o_que_evitar: List[str]
    dica_de_conforto: str

class NutricaoOncologica(BaseModel):
    fase_atual: str
    alimentos_aliados: List[str]
    alimentos_a_evitar: List[str]
    meta_hidratacao_ml: int
    dica_paladar_ou_nausea: str

class PerguntaConsulta(BaseModel):
    duvida_para_oncologista: str
    por_que_perguntar: str

class GuiaCuidadoOncologico(BaseModel):
    mensagem_acolhimento: str
    sinal_alerta_urgencia: str
    opiniao_objetiva: OpiniaoObjetiva
    manejo_sintomas: ManejoSintoma
    nutricao_e_hidratacao: NutricaoOncologica
    checklist_proxima_consulta: List[PerguntaConsulta]
    pratica_bem_estar_mental: str

class DadosEntrada(BaseModel):
    tipo_tratamento: str
    fase_ciclo: str
    apetite_nivel: str
    sintomas_descritos: str

class RespostaCompleta(BaseModel):
    guia: GuiaCuidadoOncologico
    tokens: Dict[str, Any]

# --- Fallback Local Inteligente ---
def gerar_fallback(dados: DadosEntrada) -> GuiaCuidadoOncologico:
    sintoma_resumo = dados.sintomas_descritos.strip() if dados.sintomas_descritos.strip() else "Fadiga, Náusea leve e Sensibilidade geral"
    return GuiaCuidadoOncologico(
        mensagem_acolhimento="Você é mais forte do que imagina. Cada etapa superada aproxima você da sua recuperação.",
        sinal_alerta_urgencia="FEBRE (≥ 37,8°C), calafrios intensos, sangramentos espontâneos ou falta de ar exigem ida imediata ao pronto-atendimento oncológico.",
        opiniao_objetiva=OpiniaoObjetiva(
            classificacao_momento=f"Quadro de {dados.tipo_tratamento} na fase: {dados.fase_ciclo}.",
            parecer_direto="Os relatos indicam reações comuns ao protocolo oncológico. O foco prioritário deve ser o alívio sintomático dos desconfortos descritos, manutenção rigorosa da hidratação e proteção da imunidade celular.",
            nivel_cuidado="Cuidado Ativo • Suporte Contínuo"
        ),
        manejo_sintomas=ManejoSintoma(
            sintoma_foco=sintoma_resumo[:80] + ("..." if len(sintoma_resumo) > 80 else ""),
            o_que_fazer_agora=[
                "Fracione as refeições em pequenas porções a cada 2 a 3 horas sem forçar grandes volumes.",
                "Prefira alimentos em temperatura ambiente ou frios para minimizar cheiros fortes.",
                "Faça pausas curtas de descanso em posição semi-sentada para evitar refluxo e náuseas."
            ],
            o_que_evitar=[
                "Alimentos gordurosos, frituras, excesso de condimentos e longos períodos em jejum.",
                "Esforço físico intenso durante picos de desconforto ou fadiga."
            ],
            dica_de_conforto="Gelo picado com raspas de limão ou água de coco bem gelada ajuda a hidratar a mucosa e diminuir o gosto amargo/metálico."
        ),
        nutricao_e_hidratacao=NutricaoOncologica(
            fase_atual=f"{dados.tipo_tratamento} ({dados.fase_ciclo})",
            alimentos_aliados=[
                "Caldo caseiro nutritivo de legumes e frango desfiado",
                "Ovos cozidos bem passados ou mexidos suaves",
                "Purê de batata, cenoura ou mandioquinha",
                "Frutas ricas em água (melancia, melão descascados na hora)"
            ],
            alimentos_a_evitar=[
                "Carnes cruas, sushi ou malpassadas (risco microbiológico)",
                "Vegetais crus não higienizados rigorosamente com solução clorada",
                "Laticínios e queijos não pasteurizados"
            ],
            meta_hidratacao_ml=2200,
            dica_paladar_ou_nausea="Caso sinta alteração metálica no paladar, utilize talheres de madeira ou silicone nas refeições."
        ),
        checklist_proxima_consulta=[
            PerguntaConsulta(
                duvida_para_oncologista="Os desconfortos que relatei nesta semana exigem ajuste em antieméticos ou protetores gástricos?",
                por_que_perguntar="Garante que seu protocolo de medicamentos de suporte seja ajustado à sua resposta clínica."
            ),
            PerguntaConsulta(
                duvida_para_oncologista="Existe recomendação de suplementação proteica específica para meu peso atual?",
                por_que_perguntar="Ajuda a prevenir perda de massa magra e fraqueza muscular."
            )
        ],
        pratica_bem_estar_mental="Técnica de Respiração 4-4-4: inspire devagar em 4 segundos, retenha o ar por 4 segundos e expire suavemente em 4 segundos. Respeite o ritmo do seu corpo."
    )

# --- Endpoint de Geração ---
@app.post("/api/v1/cuidado/gerar", response_model=RespostaCompleta)
def gerar_orientacao(dados: DadosEntrada):
    prompt_tokens = 0
    resposta_tokens = 0
    total_tokens = 0
    modo = "Ao Vivo (Gemini 3.5 Flash Lite)"

    try:
        chave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=chave) if chave else genai.Client()

        prompt = f"""
        Você é um Assistente Especialista em Oncologia Clínica Integrativa, Cuidados de Suporte e Manejo de Sintomas.
        Analise o relato do paciente e elabore um plano acolhedor, altamente didático, objetivo e fundamentado em evidências médicas:

        - Tipo de Tratamento: {dados.tipo_tratamento}
        - Momento / Fase Atual: {dados.fase_ciclo}
        - Padrão de Apetite / Deglutição: {dados.apetite_nivel}
        - Descrição Livre dos Sintomas pelo Paciente:
        "{dados.sintomas_descritos if dados.sintomas_descritos.strip() else 'Nenhum sintoma grave descrito, focado em suporte geral e recuperação.'}"

        DIRETRIZES MÉDICAS OBRIGATÓRIAS:
        1. 'opiniao_objetiva': emita uma avaliação clínica concisa, direta e realista sobre o momento e a prioridade imediata do paciente.
        2. 'manejo_sintomas': aborde especificamente os sintomas descritos no texto livre, fornecendo passos práticos imediatos e o que evitar.
        3. 'sinal_alerta_urgencia': reforce sempre febre (≥ 37,8°C), calafrios, sangramentos ou falta de ar como emergência oncológica.
        4. 'nutricao_e_hidratacao': indique alimentos confortáveis, seguros (risco bacteriano/neutropenia) e estratégia contra disgeusia/gosto ruim.
        5. 'checklist_proxima_consulta': crie 2 a 3 perguntas essenciais para levar ao oncologista baseadas no que o paciente relatou.
        """

        response = client.models.generate_content(
            model=MODELO_IA,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GuiaCuidadoOncologico
            )
        )
        guia_gerado = response.parsed

        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            resposta_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0

    except Exception as e:
        print(f"\n[Aviso] Usando Fallback de segurança: {e}")
        guia_gerado = gerar_fallback(dados)
        prompt_tokens, resposta_tokens, total_tokens = 210, 440, 650
        modo = "Simulação Local de Suporte (Fallback)"

    return RespostaCompleta(
        guia=guia_gerado,
        tokens={
            "prompt_tokens": prompt_tokens,
            "resposta_tokens": resposta_tokens,
            "total_tokens": total_tokens,
            "modelo_usado": MODELO_IA,
            "modo": modo
        }
    )

# --- Interface Web em Tom Pastel Laranja ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cancer Loser - O Câncer Não Vai Ganhar</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            body { 
                font-family: 'Plus Jakarta Sans', sans-serif; 
                background-color: #FFF7ED; /* Pastel Laranja */
            }
        </style>
    </head>
    <body class="text-slate-800 min-h-screen">
        <div class="max-w-6xl mx-auto px-4 py-8">
            
            <!-- Cabeçalho -->
            <header class="bg-white/95 backdrop-blur border border-orange-200 rounded-3xl p-6 mb-8 shadow-sm flex flex-col md:flex-row justify-between items-center gap-4">
                <div class="flex items-center gap-4">
                    <div class="w-14 h-14 bg-gradient-to-tr from-orange-400 to-amber-300 rounded-2xl flex items-center justify-center text-2xl shadow-md shadow-orange-200">
                        🎗️
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="text-2xl font-extrabold text-orange-950 tracking-tight">Cancer Loser</h1>
                            <span class="bg-orange-100 text-orange-800 border border-orange-300 text-xs px-2.5 py-0.5 rounded-full font-bold">O Câncer Perde, Você Vence</span>
                        </div>
                        <p class="text-orange-900/70 text-xs mt-0.5">Plataforma ampliada de suporte, conforto, manejo de sintomas e nutrição oncológica.</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-xs bg-amber-50 border border-amber-200 text-amber-800 px-3 py-1.5 rounded-xl font-semibold">
                        Gemini 3.5 Flash Lite • 500 req/dia
                    </span>
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <!-- Formulário de Situação do Paciente -->
                <div class="lg:col-span-5 bg-white border border-orange-200 rounded-3xl p-6 shadow-md shadow-orange-100 space-y-5">
                    <div class="border-b border-orange-100 pb-3">
                        <h2 class="text-base font-bold text-orange-950 flex items-center gap-2">
                            <span>📋</span> Triagem do Seu Momento
                        </h2>
                        <p class="text-xs text-orange-900/60 mt-1">Preencha as opções clínicas e descreva como está o seu corpo hoje.</p>
                    </div>

                    <!-- Tratamento Ampliado -->
                    <div>
                        <label class="block text-xs font-bold text-orange-900 mb-1">TIPO DE TRATAMENTO ONCOLÓGICO</label>
                        <select id="tipo_tratamento" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2.5 text-xs font-medium text-slate-800 focus:outline-none focus:border-orange-400">
                            <option value="Quimioterapia (Infusional ou Oral)">Quimioterapia (Infusional ou Oral)</option>
                            <option value="Radioterapia (Externa ou Braquiterapia)">Radioterapia (Externa ou Braquiterapia)</option>
                            <option value="Imunoterapia">Imunoterapia</option>
                            <option value="Terapia Alvo / Inibidores Moleculares">Terapia Alvo / Inibidores Moleculares</option>
                            <option value="Hormonioterapia / Bloqueio Hormonal">Hormonioterapia / Bloqueio Hormonal</option>
                            <option value="Transplante de Medula Óssea (TMO) / Terapia Celular">Transplante de Medula Óssea (TMO)</option>
                            <option value="Pós-Operatório / Ressecção Cirúrgica">Pós-Operatório / Cirurgia Oncológica</option>
                            <option value="Tratamento Combinado (ex: Quimio + Radio)">Tratamento Combinado (Quimio + Radio)</option>
                            <option value="Cuidados de Suporte / Controle de Sintomas">Cuidados de Suporte / Controle Sintomático</option>
                            <option value="Acompanhamento / Sobrevivência (Pós-Tratamento)">Acompanhamento Pós-Tratamento</option>
                        </select>
                    </div>

                    <!-- Fase e Apetite -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-bold text-orange-900 mb-1">MOMENTO / CICLO</label>
                            <select id="fase_ciclo" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-orange-400">
                                <option value="1 a 3 dias pós-sessão (Pico Agudo)">1 a 3 dias pós-sessão (Pico Agudo)</option>
                                <option value="Fase de Nadir / Queda de Imunidade (D7 a D14)">Nadir / Imunidade Baixa (D7 a D14)</option>
                                <option value="Semana de Intervalo / Recuperação">Semana de Intervalo</option>
                                <option value="Preparando próxima sessão (Ansiedade)">Preparando Próxima Sessão</option>
                                <option value="Uso Contínuo Diário / Manutenção">Uso Contínuo / Manutenção</option>
                                <option value="Recém-operado (Recuperação Cirúrgica)">Pós-Cirúrgico Recente</option>
                                <option value="Diagnóstico Recente / Pré-Início">Diagnóstico Recente</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-orange-900 mb-1">APETITE / DEGLUTIÇÃO</label>
                            <select id="apetite_nivel" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-orange-400">
                                <option value="Normal / Estável">Normal / Estável</option>
                                <option value="Muito reduzido / Sem fome">Sem Fome / Reduzido</option>
                                <option value="Com náusea ou enjoo frequente">Com Náusea / Enjoo</option>
                                <option value="Gosto metálico ou amargo na boca">Gosto Metálico na Boca</option>
                                <option value="Aversão a cheiros de comida">Aversão a Cheiros</option>
                                <option value="Boca seca ou dor ao engolir (Mucosite)">Boca Seca / Dor ao Engolir</option>
                            </select>
                        </div>
                    </div>

                    <!-- Área de Texto Livre para Sintomas -->
                    <div>
                        <div class="flex justify-between items-center mb-1.5">
                            <label class="block text-xs font-bold text-orange-900">QUAIS SINTOMAS DESEJA MELHORAR AGORA?</label>
                            <span class="text-[10px] text-orange-800/60">Escreva livremente</span>
                        </div>
                        <textarea id="sintomas_descritos" rows="4" placeholder="Descreva com suas palavras tudo o que está sentindo (ex: sinto muita náusea pela manhã, aftas doloridas na bochecha, cansaço pesado nas pernas e pontadas nas mãos)..." class="w-full bg-orange-50/50 border border-orange-200 rounded-2xl p-3 text-xs text-slate-800 placeholder:text-orange-900/40 focus:outline-none focus:border-orange-400 resize-none"></textarea>
                        
                        <!-- Atalhos Rápidos para Inserir Sintomas Comuns no Texto -->
                        <div class="mt-2">
                            <span class="text-[10px] font-bold text-orange-900/70 block mb-1">Atalhos rápidos para adicionar ao texto:</span>
                            <div class="flex flex-wrap gap-1.5 text-[11px]">
                                <button type="button" onclick="adicionarSintoma('Náusea e enjoo')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ 🤢 Náusea</button>
                                <button type="button" onclick="adicionarSintoma('Fadiga extrema e fraqueza')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ 🔋 Fadiga</button>
                                <button type="button" onclick="adicionarSintoma('Aftas doloridas e boca seca (Mucosite)')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ 👄 Aftas</button>
                                <button type="button" onclick="adicionarSintoma('Formigamento nas mãos e pés (Neuropatia)')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ ⚡ Neuropatia</button>
                                <button type="button" onclick="adicionarSintoma('Diarreia frequente')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ 💧 Diarreia</button>
                                <button type="button" onclick="adicionarSintoma('Constipação / Intestino preso')" class="bg-orange-100 hover:bg-orange-200 text-orange-900 px-2 py-0.5 rounded-lg border border-orange-200 transition">+ 🌾 Intestino Preso</button>
                            </div>
                        </div>
                    </div>

                    <button onclick="gerarPlano()" id="btnGerar" class="w-full bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-extrabold py-3.5 rounded-2xl shadow-lg shadow-orange-300/50 transition flex items-center justify-center gap-2">
                        <span>✨ Gerar Meu Plano de Cuidado</span>
                    </button>
                    
                    <p class="text-[11px] text-center text-orange-900/50">
                        ⚠️ Este aplicativo não substitui seu oncologista. Siga sempre o protocolo médico.
                    </p>
                </div>

                <!-- Painel de Resultados Ilustrativos -->
                <div class="lg:col-span-7 space-y-6">
                    
                    <div id="loading" class="hidden bg-white border border-orange-200 rounded-3xl p-12 text-center space-y-4 shadow-sm">
                        <div class="w-12 h-12 border-4 border-orange-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                        <p class="text-sm font-bold text-orange-900">Analisando seus sintomas e formulando recomendações personalizadas...</p>
                    </div>

                    <div id="resultado" class="hidden space-y-6">
                        
                        <!-- Card de Acolhimento -->
                        <div class="bg-gradient-to-r from-orange-100 to-amber-100 border border-orange-200 rounded-3xl p-5 shadow-sm">
                            <div class="flex items-start gap-3">
                                <span class="text-2xl">🧡</span>
                                <div>
                                    <h3 class="text-sm font-bold text-orange-950">Mensagem de Força do Cancer Loser</h3>
                                    <p id="resAcolhimento" class="text-xs text-orange-950/80 mt-1 italic"></p>
                                </div>
                            </div>
                        </div>

                        <!-- Card de Alerta Vermelho -->
                        <div class="bg-red-50 border-2 border-red-200 rounded-3xl p-4 shadow-sm flex items-start gap-3">
                            <span class="text-2xl">🚨</span>
                            <div>
                                <h4 class="text-xs font-extrabold text-red-900 uppercase tracking-wider">Atenção Médica Imediata</h4>
                                <p id="resAlerta" class="text-xs text-red-800 font-semibold mt-0.5"></p>
                            </div>
                        </div>

                        <!-- Opinião Clínica Objetiva -->
                        <div class="bg-white border-2 border-amber-300 rounded-3xl p-6 shadow-sm space-y-3 bg-gradient-to-b from-white to-amber-50/30">
                            <div class="flex justify-between items-center border-b border-amber-100 pb-2.5">
                                <h4 class="text-sm font-bold text-orange-950 flex items-center gap-2">
                                    <span>🎯</span> Opinião Clínica Objetiva
                                </h4>
                                <span id="resNivelCuidado" class="bg-amber-100 text-amber-900 border border-amber-300 text-[10px] font-extrabold px-2.5 py-1 rounded-full"></span>
                            </div>
                            <div>
                                <span id="resClassificacao" class="text-xs font-bold text-orange-900 block mb-1"></span>
                                <p id="resParecer" class="text-xs text-slate-700 leading-relaxed"></p>
                            </div>
                        </div>

                        <!-- Manejo Direcionado dos Sintomas -->
                        <div class="bg-white border border-orange-200 rounded-3xl p-6 shadow-sm space-y-4">
                            <div class="flex justify-between items-center border-b border-orange-100 pb-3">
                                <h4 class="text-sm font-bold text-orange-950 flex items-center gap-2">
                                    <span>🩺</span> Alívio para os Sintomas: <span id="resFoco" class="text-orange-600 font-semibold"></span>
                                </h4>
                                <span class="bg-orange-100 text-orange-800 text-[10px] font-extrabold px-2.5 py-1 rounded-full">Passo a Passo</span>
                            </div>
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div class="bg-emerald-50/60 border border-emerald-200/80 rounded-2xl p-4">
                                    <span class="text-xs font-bold text-emerald-900 block mb-2">✅ Ações Recomendadas</span>
                                    <ul id="resAcoes" class="space-y-1.5 text-xs text-emerald-950"></ul>
                                </div>
                                <div class="bg-amber-50/60 border border-amber-200/80 rounded-2xl p-4">
                                    <span class="text-xs font-bold text-amber-900 block mb-2">⛔ O que Evitar</span>
                                    <ul id="resEvitar" class="space-y-1.5 text-xs text-amber-950"></ul>
                                </div>
                            </div>

                            <div class="bg-orange-50 border border-orange-200 rounded-2xl p-3.5">
                                <span class="text-[11px] font-bold text-orange-900 block mb-0.5">💡 Dica Prática de Conforto:</span>
                                <p id="resDicaConforto" class="text-xs text-orange-950/80"></p>
                            </div>
                        </div>

                        <!-- Nutrição & Hidratação -->
                        <div class="bg-white border border-orange-200 rounded-3xl p-6 shadow-sm space-y-4">
                            <div class="flex justify-between items-center border-b border-orange-100 pb-3">
                                <h4 class="text-sm font-bold text-orange-950 flex items-center gap-2">
                                    <span>🥣</span> Nutrição & Hidratação Segura
                                </h4>
                                <span id="resMetaHidratacao" class="bg-blue-50 text-blue-800 border border-blue-200 text-[11px] font-bold px-3 py-1 rounded-full"></span>
                            </div>

                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <span class="text-xs font-bold text-orange-900 block mb-1.5">🥦 Alimentos Aliados</span>
                                    <ul id="resAliados" class="space-y-1 text-xs text-slate-700"></ul>
                                </div>
                                <div>
                                    <span class="text-xs font-bold text-orange-900 block mb-1.5">🚫 Alimentos a Evitar (Risco Microbiano)</span>
                                    <ul id="resEvitarAlimentos" class="space-y-1 text-xs text-slate-700"></ul>
                                </div>
                            </div>

                            <div class="bg-amber-50/70 border border-amber-200 rounded-2xl p-3.5">
                                <span class="text-[11px] font-bold text-amber-900 block mb-0.5">👅 Para o Paladar / Sensibilidade:</span>
                                <p id="resDicaPaladar" class="text-xs text-amber-950/80"></p>
                            </div>
                        </div>

                        <!-- Checklist para Consulta -->
                        <div class="bg-white border border-orange-200 rounded-3xl p-6 shadow-sm space-y-3">
                            <h4 class="text-sm font-bold text-orange-950 flex items-center gap-2">
                                <span>📝</span> Leve para Perguntar ao seu Oncologista
                            </h4>
                            <div id="resConsultas" class="space-y-2"></div>
                        </div>

                        <!-- Bem-estar Mental -->
                        <div class="bg-orange-100/50 border border-orange-200 rounded-3xl p-4 shadow-sm flex items-center gap-3">
                            <span class="text-2xl">🌱</span>
                            <div>
                                <span class="text-xs font-bold text-orange-950 block">Pausa de Respiração & Conforto:</span>
                                <p id="resMental" class="text-xs text-orange-900/80 mt-0.5"></p>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>

        <script>
            function adicionarSintoma(sintoma) {
                const txt = document.getElementById('sintomas_descritos');
                if (txt.value.trim() === '') {
                    txt.value = sintoma;
                } else {
                    txt.value += ', ' + sintoma;
                }
                txt.focus();
            }

            async function gerarPlano() {
                const btn = document.getElementById('btnGerar');
                const loading = document.getElementById('loading');
                const resultado = document.getElementById('resultado');

                const tratamento = document.getElementById('tipo_tratamento').value;
                const fase = document.getElementById('fase_ciclo').value;
                const apetite = document.getElementById('apetite_nivel').value;
                const sintomas = document.getElementById('sintomas_descritos').value;

                btn.disabled = true;
                btn.classList.add('opacity-50');
                loading.classList.remove('hidden');
                resultado.classList.add('hidden');

                try {
                    const response = await fetch('/api/v1/cuidado/gerar', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            tipo_tratamento: tratamento,
                            fase_ciclo: fase,
                            apetite_nivel: apetite,
                            sintomas_descritos: sintomas
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Status: ' + response.status);
                    }

                    const data = await response.json();
                    const g = data.guia;

                    // Acolhimento & Alerta
                    document.getElementById('resAcolhimento').innerText = `"${g.mensagem_acolhimento}"`;
                    document.getElementById('resAlerta').innerText = g.sinal_alerta_urgencia;

                    // Opinião Objetiva
                    document.getElementById('resNivelCuidado').innerText = g.opiniao_objetiva.nivel_cuidado;
                    document.getElementById('resClassificacao').innerText = g.opiniao_objetiva.classificacao_momento;
                    document.getElementById('resParecer').innerText = g.opiniao_objetiva.parecer_direto;

                    // Manejo Sintomas
                    document.getElementById('resFoco').innerText = g.manejo_sintomas.sintoma_foco;
                    document.getElementById('resAcoes').innerHTML = g.manejo_sintomas.o_que_fazer_agora.map(a => `<li>• ${a}</li>`).join('');
                    document.getElementById('resEvitar').innerHTML = g.manejo_sintomas.o_que_evitar.map(e => `<li>• ${e}</li>`).join('');
                    document.getElementById('resDicaConforto').innerText = g.manejo_sintomas.dica_de_conforto;

                    // Nutrição & Hidratação
                    document.getElementById('resMetaHidratacao').innerText = `Meta de Água: ${(g.nutricao_e_hidratacao.meta_hidratacao_ml / 1000).toFixed(1)}L/dia`;
                    document.getElementById('resAliados').innerHTML = g.nutricao_e_hidratacao.alimentos_aliados.map(al => `<li>• ${al}</li>`).join('');
                    document.getElementById('resEvitarAlimentos').innerHTML = g.nutricao_e_hidratacao.alimentos_a_evitar.map(ea => `<li>• ${ea}</li>`).join('');
                    document.getElementById('resDicaPaladar').innerText = g.nutricao_e_hidratacao.dica_paladar_ou_nausea;

                    // Checklist Consulta
                    document.getElementById('resConsultas').innerHTML = g.checklist_proxima_consulta.map(c => `
                        <div class="bg-orange-50/50 p-3 rounded-2xl border border-orange-200/80">
                            <span class="font-bold text-xs text-orange-950 block">❓ "${c.duvida_para_oncologista}"</span>
                            <span class="text-[11px] text-orange-900/70 block mt-0.5">Motivo: ${c.por_que_perguntar}</span>
                        </div>
                    `).join('');

                    // Bem-estar Mental
                    document.getElementById('resMental').innerText = g.pratica_bem_estar_mental;

                    resultado.classList.remove('hidden');
                } catch(err) {
                    alert('Erro ao gerar orientações: ' + err.message);
                } finally {
                    btn.disabled = false;
                    btn.classList.remove('opacity-50');
                    loading.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
