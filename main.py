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
    title="Cancer Loser - Apoio Oncológico",
    version="1.0.0"
)

# --- Modelo Padrão de Alta Cota (500 req/dia) ---
MODELO_PADRAO = "gemini-3.5-flash-lite"

# --- Schemas Pydantic Estruturados ---
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
    manejo_sintomas: ManejoSintoma
    nutricao_e_hidratacao: NutricaoOncologica
    checklist_proxima_consulta: List[PerguntaConsulta]
    pratica_bem_estar_mental: str

class DadosEntrada(BaseModel):
    tipo_tratamento: str
    fase_ciclo: str
    sintomas_atuais: List[str]
    apetite_nivel: str

class RespostaCompleta(BaseModel):
    guia: GuiaCuidadoOncologico
    status_info: Dict[str, Any]

# --- Fallback Local Inteligente (Garante resposta mesmo sem internet/cota) ---
def gerar_fallback(dados: DadosEntrada) -> GuiaCuidadoOncologico:
    sintomas_str = ", ".join(dados.sintomas_atuais) if dados.sintomas_atuais else "Fadiga e Náusea leve"
    return GuiaCuidadoOncologico(
        mensagem_acolhimento="Você é mais forte do que imagina. Cada etapa cumprida é uma vitória contra o câncer.",
        sinal_alerta_urgencia="FEBRE (≥ 37,8°C), calafrios intensos, sangramentos espontâneos ou falta de ar exigem ida imediata ao pronto-atendimento oncológico.",
        manejo_sintomas=ManejoSintoma(
            sintoma_foco=sintomas_str,
            o_que_fazer_agora=[
                "Fracione as refeições em pequenas porções a cada 2 a 3 horas.",
                "Prefira alimentos em temperatura ambiente ou frios para diminuir odores fortes.",
                "Mantenha repouso em posição semi-inclinada após as refeições."
            ],
            o_que_evitar=[
                "Frituras, alimentos muito condimentados e longos períodos em jejum.",
                "Bebidas gaseificadas ou excessivamente açucaradas."
            ],
            dica_de_conforto="Cubos de gelo com raspas de limão ou água de coco aliviam o enjoo e a sensação de boca seca."
        ),
        nutricao_e_hidratacao=NutricaoOncologica(
            fase_atual=f"{dados.tipo_tratamento} ({dados.fase_ciclo})",
            alimentos_aliados=[
                "Caldo caseiro de legumes",
                "Ovos cozidos bem passados",
                "Purê de batata ou mandioquinha",
                "Frutas com alto teor de água (melancia, melão)"
            ],
            alimentos_a_evitar=[
                "Carnes cruas ou malpassadas (risco microbiológico)",
                "Vegetais crus não sanitizados rigorosamente",
                "Laticínios não pasteurizados"
            ],
            meta_hidratacao_ml=2200,
            dica_paladar_ou_nausea="Use talheres de plástico ou madeira caso sinta gosto metálico durante a alimentação."
        ),
        checklist_proxima_consulta=[
            PerguntaConsulta(
                duvida_para_oncologista="A intensidade dos sintomas que senti nesta semana está dentro do esperado para o meu protocolo?",
                por_que_perguntar="Ajuda o médico a avaliar necessidade de ajuste de dosagem ou antieméticos."
            ),
            PerguntaConsulta(
                duvida_para_oncologista="Posso utilizar algum suplemento hiperproteico para evitar perda muscular?",
                por_que_perguntar="Mantém a reserva energética e combate a sarcopenia."
            )
        ],
        pratica_bem_estar_mental="Exercício 4-4-4: inspire em 4 segundos, segure por 4 segundos e expire suavemente em 4 segundos. Respeite os limites do seu corpo hoje."
    )

# --- Endpoint da API ---
@app.post("/api/v1/cuidado/gerar", response_model=RespostaCompleta)
def gerar_orientacao(dados: DadosEntrada):
    modo = "Gemini 3.5 Flash Lite (Tempo Real)"
    
    try:
        chave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=chave) if chave else genai.Client()

        prompt = f"""
        Você é um Especialista em Cuidados e Conforto Oncológico.
        Gere um plano estruturado, acolhedor e altamente didático para um paciente oncológico:

        - Tratamento: {dados.tipo_tratamento}
        - Momento: {dados.fase_ciclo}
        - Sintomas relatados: {', '.join(dados.sintomas_atuais) if dados.sintomas_atuais else 'Bem-estar geral'}
        - Apetite: {dados.apetite_nivel}

        DIRETRIZES MÉDICAS:
        1. Alerte com ênfase sobre febre (≥ 37,8°C) como urgência médica.
        2. Dê orientações práticas de nutrição segura (evitar alimentos crus e risco bacteriano).
        3. Forneça dicas para alterações de paladar e náusea.
        4. Monte perguntas relevantes para a consulta com o oncologista.
        """

        response = client.models.generate_content(
            model=MODELO_PADRAO,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GuiaCuidadoOncologico
            )
        )
        guia_gerado = response.parsed

    except Exception as e:
        print(f"\n[Aviso] Falha na chamada da IA ({e}), usando Fallback seguro.")
        guia_gerado = gerar_fallback(dados)
        modo = "Guia Local de Segurança (Fallback Ativo)"

    return RespostaCompleta(
        guia=guia_gerado,
        status_info={"modelo": MODELO_PADRAO, "modo": modo}
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
                        <p class="text-orange-900/70 text-xs mt-0.5">Guia de suporte, conforto e nutrição oncológica com IA (Gemini 3.5 Flash Lite - 500 req/dia).</p>
                    </div>
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <!-- Formulário -->
                <div class="lg:col-span-5 bg-white border border-orange-200 rounded-3xl p-6 shadow-md shadow-orange-100 space-y-5">
                    <div class="border-b border-orange-100 pb-3">
                        <h2 class="text-base font-bold text-orange-950 flex items-center gap-2">
                            <span>📋</span> Como você está se sentindo hoje?
                        </h2>
                        <p class="text-xs text-orange-900/60 mt-1">Preencha os campos para receber as condutas de alívio personalizadas.</p>
                    </div>

                    <div>
                        <label class="block text-xs font-bold text-orange-900 mb-1">QUAL TRATAMENTO ESTÁ FAZENDO?</label>
                        <select id="tipo_tratamento" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2.5 text-sm text-slate-800 focus:outline-none focus:border-orange-400">
                            <option value="Quimioterapia">Quimioterapia</option>
                            <option value="Radioterapia">Radioterapia</option>
                            <option value="Imunoterapia / Terapia Alvo">Imunoterapia / Terapia Alvo</option>
                            <option value="Pós-Cirúrgico / Recuperação">Pós-Cirúrgico / Recuperação</option>
                            <option value="Acompanhamento Preventivo">Acompanhamento Preventivo</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-bold text-orange-900 mb-1">FASE / MOMENTO</label>
                            <select id="fase_ciclo" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-orange-400">
                                <option value="1 a 3 dias pós-sessão">1 a 3 dias pós-sessão</option>
                                <option value="Semana de intervalo">Semana de intervalo</option>
                                <option value="Preparando próxima sessão">Preparando sessão</option>
                                <option value="Manutenção">Manutenção</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-orange-900 mb-1">NÍVEL DO APETITE</label>
                            <select id="apetite_nivel" class="w-full bg-orange-50/50 border border-orange-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-orange-400">
                                <option value="Normal">Normal</option>
                                <option value="Reduzido / Pouco apetite">Reduzido</option>
                                <option value="Com enjoo ou náusea">Com enjoo</option>
                                <option value="Gosto metálico na boca">Gosto metálico</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-bold text-orange-900 mb-2">SINTOMAS QUE DESEJA ALIVIAR</label>
                        <div class="grid grid-cols-2 gap-2 text-xs text-slate-700">
                            <label class="flex items-center gap-2 bg-orange-50/60 p-2.5 rounded-xl border border-orange-200/80 cursor-pointer hover:bg-orange-100/60 transition">
                                <input type="checkbox" value="Náusea ou Enjoo" class="sintoma" checked> 🤢 Náusea
                            </label>
                            <label class="flex items-center gap-2 bg-orange-50/60 p-2.5 rounded-xl border border-orange-200/80 cursor-pointer hover:bg-orange-100/60 transition">
                                <input type="checkbox" value="Cansaço ou Fadiga" class="sintoma" checked> 🔋 Fadiga
                            </label>
                            <label class="flex items-center gap-2 bg-orange-50/60 p-2.5 rounded-xl border border-orange-200/80 cursor-pointer hover:bg-orange-100/60 transition">
                                <input type="checkbox" value="Boca seca ou Aftas" class="sintoma"> 👄 Aftas / Secura
                            </label>
                            <label class="flex items-center gap-2 bg-orange-50/60 p-2.5 rounded-xl border border-orange-200/80 cursor-pointer hover:bg-orange-100/60 transition">
                                <input type="checkbox" value="Sensibilidade Intestinal" class="sintoma"> 🌾 Intestino
                            </label>
                        </div>
                    </div>

                    <button onclick="gerarPlano()" id="btnGerar" class="w-full bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-extrabold py-3.5 rounded-2xl shadow-lg shadow-orange-300/50 transition flex items-center justify-center gap-2">
                        <span>✨ Gerar Meu Plano de Conforto</span>
                    </button>
                    
                    <p class="text-[11px] text-center text-orange-900/50">
                        ⚠️ Este aplicativo não substitui seu oncologista. Siga sempre o protocolo médico.
                    </p>
                </div>

                <!-- Painel de Resultados -->
                <div class="lg:col-span-7 space-y-6">
                    
                    <div id="loading" class="hidden bg-white border border-orange-200 rounded-3xl p-12 text-center space-y-4 shadow-sm">
                        <div class="w-12 h-12 border-4 border-orange-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                        <p class="text-sm font-bold text-orange-900">Gerando recomendações com Gemini 3.5 Flash Lite...</p>
                    </div>

                    <div id="resultado" class="hidden space-y-6">
                        
                        <!-- Card Acolhimento -->
                        <div class="bg-gradient-to-r from-orange-100 to-amber-100 border border-orange-200 rounded-3xl p-5 shadow-sm">
                            <div class="flex items-start gap-3">
                                <span class="text-2xl">🧡</span>
                                <div>
                                    <h3 class="text-sm font-bold text-orange-950">Mensagem de Força do Cancer Loser</h3>
                                    <p id="resAcolhimento" class="text-xs text-orange-950/80 mt-1 italic"></p>
                                </div>
                            </div>
                        </div>

                        <!-- Alerta Vermelho -->
                        <div class="bg-red-50 border-2 border-red-200 rounded-3xl p-4 shadow-sm flex items-start gap-3">
                            <span class="text-2xl">🚨</span>
                            <div>
                                <h4 class="text-xs font-extrabold text-red-900 uppercase tracking-wider">Atenção Médica Imediata</h4>
                                <p id="resAlerta" class="text-xs text-red-800 font-semibold mt-0.5"></p>
                            </div>
                        </div>

                        <!-- Manejo Sintomas -->
                        <div class="bg-white border border-orange-200 rounded-3xl p-6 shadow-sm space-y-4">
                            <div class="flex justify-between items-center border-b border-orange-100 pb-3">
                                <h4 class="text-sm font-bold text-orange-950 flex items-center gap-2">
                                    <span>🎯</span> O que Fazer para Conforto: <span id="resFoco" class="text-orange-600"></span>
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
                                <span class="text-[11px] font-bold text-orange-900 block mb-0.5">💡 Dica Prática de Alívio:</span>
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
                                <span class="text-[11px] font-bold text-amber-900 block mb-0.5">👅 Para o Paladar / Gosto Metálico:</span>
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
            async function gerarPlano() {
                const btn = document.getElementById('btnGerar');
                const loading = document.getElementById('loading');
                const resultado = document.getElementById('resultado');

                const tratamento = document.getElementById('tipo_tratamento').value;
                const fase = document.getElementById('fase_ciclo').value;
                const apetite = document.getElementById('apetite_nivel').value;
                const sintomas = Array.from(document.querySelectorAll('.sintoma:checked')).map(c => c.value);

                btn.disabled = true;
                btn.classList.add('opacity-50');
                loading.classList.remove('hidden');
                resultado.classList.add('hidden');

                try {
                    // Chamada com rota relativa (funciona local e no Render)
                    const response = await fetch('/api/v1/cuidado/gerar', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            tipo_tratamento: tratamento,
                            fase_ciclo: fase,
                            sintomas_atuais: sintomas,
                            apetite_nivel: apetite
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Servidor retornou status ' + response.status);
                    }

                    const data = await response.json();
                    const g = data.guia;

                    document.getElementById('resAcolhimento').innerText = `"${g.mensagem_acolhimento}"`;
                    document.getElementById('resAlerta').innerText = g.sinal_alerta_urgencia;
                    document.getElementById('resFoco').innerText = g.manejo_sintomas.sintoma_foco;

                    document.getElementById('resAcoes').innerHTML = g.manejo_sintomas.o_que_fazer_agora.map(a => `<li>• ${a}</li>`).join('');
                    document.getElementById('resEvitar').innerHTML = g.manejo_sintomas.o_que_evitar.map(e => `<li>• ${e}</li>`).join('');
                    document.getElementById('resDicaConforto').innerText = g.manejo_sintomas.dica_de_conforto;

                    document.getElementById('resMetaHidratacao').innerText = `Meta de Água: ${(g.nutricao_e_hidratacao.meta_hidratacao_ml / 1000).toFixed(1)}L/dia`;
                    document.getElementById('resAliados').innerHTML = g.nutricao_e_hidratacao.alimentos_aliados.map(al => `<li>• ${al}</li>`).join('');
                    document.getElementById('resEvitarAlimentos').innerHTML = g.nutricao_e_hidratacao.alimentos_a_evitar.map(ea => `<li>• ${ea}</li>`).join('');
                    document.getElementById('resDicaPaladar').innerText = g.nutricao_e_hidratacao.dica_paladar_ou_nausea;

                    document.getElementById('resConsultas').innerHTML = g.checklist_proxima_consulta.map(c => `
                        <div class="bg-orange-50/50 p-3 rounded-2xl border border-orange-200/80">
                            <span class="font-bold text-xs text-orange-950 block">❓ "${c.duvida_para_oncologista}"</span>
                            <span class="text-[11px] text-orange-900/70 block mt-0.5">Motivo: ${c.por_que_perguntar}</span>
                        </div>
                    `).join('');

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
