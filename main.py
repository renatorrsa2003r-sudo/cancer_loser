import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

app = FastAPI(title="Cancer Loser", version="1.0.0")

class TriagemInput(BaseModel):
    tratamento_atual: str = Field(..., description="Quimioterapia, Radioterapia, Imunoterapia, Hormonal ou Pós-Cirúrgico")
    momento_ciclo: str = Field(..., description="D0, D1-D4, D7-D14, ou Recuperação")
    nivel_dor: int = Field(..., description="Escala EVA 0 a 10")
    apetite: str = Field(..., description="Normal, Ausente, Aversão a cheiros, Disgeusia")
    sintomas: List[str] = Field(default=[], description="Lista de sintomas ativos")
    opiniao_usuario: Optional[str] = Field(default="", description="Opinião, relato ou dúvida específica do paciente")
    peso_kg: float = Field(default=70.0, description="Peso em kg para cálculo hídrico")

class CondutaFarmacologica(BaseModel):
    passo: str
    descricao: str

class AlertaVermelho(BaseModel):
    necessita_atendimento_imediato: bool
    motivo: str
    instrucoes_urgencia: List[str]

class PlanoNutricional(BaseModel):
    meta_hidrica_ml: int
    alimentos_recomendados: List[str]
    alimentos_proibidos_seguranca: List[str]
    dicas_gosto_metalico: List[str]

class OncoLog(BaseModel):
    perguntas_para_medico: List[str]

class PsicoOncologia(BaseModel):
    mensagem_dia: str
    exercicio_respiracao: str

class CancerLoserResponse(BaseModel):
    alerta_vermelho: AlertaVermelho
    analise_ia_opiniao: str
    condutas_imediatas: List[CondutaFarmacologica]
    plano_nutricional: PlanoNutricional
    onco_log: OncoLog
    psico_oncologia: PsicoOncologia

def get_fallback_response(data: TriagemInput) -> CancerLoserResponse:
    febre = "Febre" in data.sintomas or data.nivel_dor > 8
    return CancerLoserResponse(
        alerta_vermelho=AlertaVermelho(
            necessita_atendimento_imediato=febre,
            motivo="Temperatura ou dor elevada detectada na triagem." if febre else "Nenhum sinal crítico imediato detectado, mas monitore.",
            instrucoes_urgencia=["Procure o pronto-socorro oncológico se a temperatura atingir 37.8°C ou mais.", "Mantenha-se hidratado."]
        ),
        analise_ia_opiniao=f"Análise baseada no seu relato ('{data.opiniao_usuario}'): É fundamental manter a comunicação com sua equipe médica e relatar qualquer mudança brusca no seu bem-estar.",
        condutas_imediatas=[
            CondutaFarmacologica(passo="1. Higiene Oral", descricao="Use solução de água com bicarbonato sem álcool 4x ao dia para prevenir mucosite."),
            CondutaFarmacologica(passo="2. Conforto Térmico", descricao="Evite extremos de temperatura nas mãos e pés para mitigar a neuropatia.")
        ],
        plano_nutricional=PlanoNutricional(
            meta_hidrica_ml=int(data.peso_kg * 35),
            alimentos_recomendados=["Caldos coados", "Frango cozido desfiado", "Arroz branco", "Ovos bem cozidos"],
            alimentos_proibidos_seguranca=["Carnes mal passadas", "Sushi", "Queijos não pasteurizados", "Frutas e verduras cruas sem higienização com cloro"],
            dicas_gosto_metalico=["Substitua talheres de metal por plástico ou madeira", "Adicione limão ou hortelã aos líquidos"]
        ),
        onco_log=OncoLog(
            perguntas_para_medico=[
                "Devo fazer uso de fator de crescimento para os leucócitos neste ciclo?",
                "Como posso ajustar o horário dos medicamentos para náusea?",
                "Quais exames de sangue devo realizar antes da próxima infusão?"
            ]
        ),
        psico_oncologia=PsicoOncologia(
            mensagem_dia="O câncer não vai ganhar esta batalha. Cada dia vencido é um passo rumo à sua cura.",
            exercicio_respiracao="Inspire pelo nariz contando até 4, segure o ar por 7 segundos e expire lentamente pela boca contando até 8. Repita 4 vezes."
        )
    )

@app.get("/api/v1/status")
async def status():
    return {"status": "online", "sistema": "Cancer Loser Ativo"}

@app.post("/api/v1/cuidado/gerar", response_model=CancerLoserResponse)
async def gerar_cuidado(data: TriagemInput):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return get_fallback_response(data)

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Você é um Oncologista Integrativo Especialista e Psico-oncologista humanizado e sênior. 
        Analise os dados do paciente com base nos protocolos MASCC, ASCO e SBOC:
        - Tratamento Atual: {data.tratamento_atual}
        - Momento do Ciclo: {data.momento_ciclo}
        - Nível de Dor (EVA 0-10): {data.nivel_dor}
        - Apetite: {data.apetite}
        - Sintomas Atuais: {', '.join(data.sintomas)}
        - Peso: {data.peso_kg} kg
        - Opinião/Dúvida/Relato Direto do Paciente: "{data.opiniao_usuario}"

        Forneça orientações humanizadas, rigorosas de segurança alimentar neutropênica, manejo farmacológico/sensorial e responda diretamente à opinião/relato do paciente de forma acolhedora.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CancerLoserResponse,
                temperature=0.3,
            ),
        )
        import json
        result_dict = json.loads(response.text)
        return CancerLoserResponse(**result_dict)
    except Exception as e:
        print(f"Erro na IA, usando fallback: {e}")
        return get_fallback_response(data)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="pt-BR" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Loser - O Câncer Perde, Você Vence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        @media print {
            body { background: white !important; }
            .no-print { display: none !important; }
            .print-only { display: block !important; }
            .card-print { border: 1px solid #cbd5e1 !important; box-shadow: none !important; }
        }
    </style>
</head>
<body class="bg-[#FFF7ED] text-slate-800 h-full flex flex-col antialiased">
    <!-- Header -->
    <header class="bg-white border-b border-orange-200 sticky top-0 z-50 shadow-sm no-print">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="bg-gradient-to-r from-orange-400 to-amber-500 text-white p-2.5 rounded-xl shadow-md font-bold text-xl">
                    CL
                </div>
                <div>
                    <h1 class="text-xl font-bold bg-gradient-to-r from-orange-600 to-amber-600 bg-clip-text text-transparent">Cancer Loser</h1>
                    <p class="text-xs text-orange-900/60 font-medium">O Câncer Perde, Você Vence</p>
                </div>
            </div>
            <div class="text-xs bg-orange-100 text-orange-800 font-semibold px-3 py-1.5 rounded-full border border-orange-200">
                Suporte Clínico Integrativo MASCC/ASCO
            </div>
        </div>
    </header>

    <!-- Navigation Tabs -->
    <nav class="bg-white/80 backdrop-blur border-b border-orange-200 no-print">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-2 sm:space-x-8 overflow-x-auto py-2">
            <button onclick="switchTab('triagem')" id="btn-triagem" class="tab-btn px-4 py-2 font-semibold text-sm rounded-lg transition-all bg-orange-500 text-white shadow-sm whitespace-nowrap">1. Triagem & Relato</button>
            <button onclick="switchTab('resultados')" id="btn-resultados" class="tab-btn px-4 py-2 font-semibold text-sm rounded-lg transition-all text-orange-900/70 hover:bg-orange-100/50 whitespace-nowrap">2. Plano de Cuidado & IA</button>
            <button onclick="switchTab('nutricao')" id="btn-nutricao" class="tab-btn px-4 py-2 font-semibold text-sm rounded-lg transition-all text-orange-900/70 hover:bg-orange-100/50 whitespace-nowrap">3. Nutrição & Segurança</button>
            <button onclick="switchTab('oncolog')" id="btn-oncolog" class="tab-btn px-4 py-2 font-semibold text-sm rounded-lg transition-all text-orange-900/70 hover:bg-orange-100/50 whitespace-nowrap">4. Onco-Log & Médico</button>
            <button onclick="switchTab('psico')" id="btn-psico" class="tab-btn px-4 py-2 font-semibold text-sm rounded-lg transition-all text-orange-900/70 hover:bg-orange-100/50 whitespace-nowrap">5. Acolhimento & Respiração</button>
        </div>
    </nav>

    <!-- Main Content Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 overflow-y-auto">
        
        <!-- TAB 1: TRIAGEM -->
        <section id="tab-triagem" class="tab-content space-y-6">
            <div class="bg-white p-6 sm:p-8 rounded-2xl border border-orange-200 shadow-sm">
                <h2 class="text-xl font-bold text-orange-900 mb-2">Painel Multiparamétrico de Triagem Diária</h2>
                <p class="text-sm text-slate-600 mb-6">Preencha seus dados clínicos atuais e compartilhe sua opinião ou sentimentos para personalizarmos seu protocolo com Inteligência Artificial.</p>
                
                <form id="form-triagem" onsubmit="enviarTriagem(event)" class="space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-2">Tratamento Atual:</label>
                            <select id="tratamento_atual" class="w-full bg-[#FFF7ED] border border-orange-200 rounded-xl px-4 py-3 text-sm font-medium focus:ring-2 focus:ring-orange-500 focus:outline-none">
                                <option value="Quimioterapia">Quimioterapia (adjuvante/neoadjuvante)</option>
                                <option value="Radioterapia">Radioterapia</option>
                                <option value="Imunoterapia / Terapia Alvo">Imunoterapia / Terapia Alvo</option>
                                <option value="Terapia Hormonal">Terapia Hormonal</option>
                                <option value="Pós-Cirúrgico">Pós-Cirúrgico</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-2">Momento do Ciclo:</label>
                            <select id="momento_ciclo" class="w-full bg-[#FFF7ED] border border-orange-200 rounded-xl px-4 py-3 text-sm font-medium focus:ring-2 focus:ring-orange-500 focus:outline-none">
                                <option value="Dia da infusão (D0)">Dia da infusão (D0)</option>
                                <option value="Fase de pico de toxicidade (D1 a D4)">Fase de pico de toxicidade (D1 a D4)</option>
                                <option value="Período de nadir/imunossupressão (D7 a D14)">Período de nadir/imunossupressão (D7 a D14)</option>
                                <option value="Período de recuperação">Período de recuperação</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-2">Nível de Dor (EVA 0-10): <span id="val-dor" class="text-orange-600 font-bold">2</span></label>
                            <input type="range" id="nivel_dor" min="0" max="10" value="2" oninput="document.getElementById('val-dor').innerText = this.value" class="w-full accent-orange-500 cursor-pointer">
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-2">Apetite:</label>
                            <select id="apetite" class="w-full bg-[#FFF7ED] border border-orange-200 rounded-xl px-4 py-3 text-sm font-medium focus:ring-2 focus:ring-orange-500 focus:outline-none">
                                <option value="Normal">Normal</option>
                                <option value="Ausente">Ausente</option>
                                <option value="Aversão a cheiros">Aversão a cheiros</option>
                                <option value="Disgeusia / gosto metálico">Disgeusia / gosto metálico</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-2">Peso Atual (kg):</label>
                            <input type="number" id="peso_kg" value="70" step="0.1" class="w-full bg-[#FFF7ED] border border-orange-200 rounded-xl px-4 py-3 text-sm font-medium focus:ring-2 focus:ring-orange-500 focus:outline-none">
                        </div>
                    </div>

                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-2">Sintomas Atuais (Marque todos aplicáveis):</label>
                        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Náusea/Vômitos" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium">Náusea / Vômitos</span>
                            </label>
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Fadiga Oncológica" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium">Fadiga Oncológica</span>
                            </label>
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Mucosite/Aftas" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium">Mucosite / Aftas</span>
                            </label>
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Febre" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium text-red-600 font-bold">Febre ou Calafrios</span>
                            </label>
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Neuropatia Periférica" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium">Neuropatia Periférica</span>
                            </label>
                            <label class="flex items-center space-x-2 bg-[#FFF7ED] p-3 rounded-xl border border-orange-200 cursor-pointer hover:bg-orange-100/50">
                                <input type="checkbox" name="sintoma" value="Ansiedade pré-ciclo" class="accent-orange-500 h-4 w-4">
                                <span class="text-sm font-medium">Ansiedade pré-ciclo</span>
                            </label>
                        </div>
                    </div>

                    <!-- Área de Relato e Opinião Objetivo do Paciente -->
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-2">Sua Opinião, Relato ou Dúvida Específica:</label>
                        <textarea id="opiniao_usuario" rows="3" placeholder="Escreva aqui como você está se sentindo hoje, o que está achando do tratamento ou qualquer dúvida que gostaria que a I.A. e sua equipe médica avaliassem..." class="w-full bg-[#FFF7ED] border border-orange-200 rounded-xl p-4 text-sm font-medium focus:ring-2 focus:ring-orange-500 focus:outline-none"></textarea>
                    </div>

                    <div class="flex justify-end">
                        <button type="submit" id="btn-submit" class="bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-bold px-8 py-3.5 rounded-xl shadow-lg shadow-orange-500/20 transition-all flex items-center space-x-2">
                            <span>Gerar Protocolo Personalizado</span>
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
                        </button>
                    </div>
                </form>
            </div>
        </section>

        <!-- TAB 2: RESULTADOS / PLANO DE CUIDADO -->
        <section id="tab-resultados" class="tab-content hidden space-y-6">
            <!-- Alerta Vermelho -->
            <div id="alerta-box" class="hidden p-6 rounded-2xl border-2 shadow-sm">
                <div class="flex items-start space-x-4">
                    <div class="p-3 bg-red-100 text-red-600 rounded-xl font-bold text-xl">⚠️</div>
                    <div>
                        <h3 id="alerta-titulo" class="text-lg font-bold text-red-900">ALERTA VERMELHO DE EMERGÊNCIA</h3>
                        <p id="alerta-motivo" class="text-sm text-red-700 mt-1"></p>
                        <ul id="alerta-instrucoes" class="mt-3 space-y-1 text-sm font-medium text-red-800 list-disc list-inside"></ul>
                    </div>
                </div>
            </div>

            <!-- Resposta da IA com Base na Opinião -->
            <div class="bg-white p-6 rounded-2xl border border-orange-200 shadow-sm">
                <h3 class="text-lg font-bold text-orange-900 mb-3">Análise Humanizada & Resposta ao Seu Relato</h3>
                <div id="ia-opiniao-texto" class="text-sm text-slate-700 leading-relaxed bg-[#FFF7ED] p-4 rounded-xl border border-orange-200">
                    Aguardando preenchimento da triagem...
                </div>
            </div>

            <!-- Condutas Farmacológicas e Sensoriais -->
            <div class="bg-white p-6 rounded-2xl border border-orange-200 shadow-sm">
                <h3 class="text-lg font-bold text-orange-900 mb-4">Manejo Farmacológico & Sensorial de Toxicidades</h3>
                <div id="condutas-container" class="space-y-4">
                    <p class="text-sm text-slate-500">Nenhum protocolo gerado ainda.</p>
                </div>
            </div>
        </section>

        <!-- TAB 3: NUTRIÇÃO & SEGURANÇA -->
        <section id="tab-nutricao" class="tab-content hidden space-y-6">
            <div class="bg-white p-6 sm:p-8 rounded-2xl border border-orange-200 shadow-sm">
                <h3 class="text-lg font-bold text-orange-900 mb-4">Plano Nutricional Seguro & Neutropenic Safety</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div class="bg-[#FFF7ED] p-4 rounded-xl border border-orange-200 text-center">
                        <span class="text-xs text-orange-800/70 font-bold uppercase tracking-wider">Meta Hídrica Diária</span>
                        <p id="nutri-agua" class="text-2xl font-bold text-orange-600 mt-1">2450 mL</p>
                    </div>
                    <div class="bg-emerald-50 p-4 rounded-xl border border-emerald-200 col-span-2">
                        <span class="text-xs text-emerald-800 font-bold uppercase tracking-wider">Estratégia Anti-Gosto Metálico</span>
                        <ul id="nutri-metalico" class="text-sm text-emerald-900 mt-1 list-disc list-inside space-y-1">
                            <li>Substituir talheres de metal por polímero ou madeira.</li>
                            <li>Utilizar limão, hortelã e temperos naturais.</li>
                        </ul>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="border border-emerald-200 bg-emerald-50/50 p-5 rounded-xl">
                        <h4 class="font-bold text-emerald-900 mb-3 flex items-center space-x-2">
                            <span>✅ Alimentos Recomendados (Alta Densidade)</span>
                        </h4>
                        <ul id="nutri-recomendados" class="space-y-2 text-sm text-emerald-800">
                            <li>Carne bem cozida e desfiada</li>
                            <li>Ovos cozidos firmes</li>
                            <li>Caldos nutritivos</li>
                        </ul>
                    </div>

                    <div class="border border-red-200 bg-red-50/50 p-5 rounded-xl">
                        <h4 class="font-bold text-red-900 mb-3 flex items-center space-x-2">
                            <span>🚫 Alimentos Proibidos (Risco Neutropênico)</span>
                        </h4>
                        <ul id="nutri-proibidos" class="space-y-2 text-sm text-red-800">
                            <li>Carnes mal passadas ou sushi</li>
                            <li>Queijos não pasteurizados</li>
                            <li>Frutas e vegetais crus sem higienização com cloro</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 4: ONCO-LOG & MÉDICO -->
        <section id="tab-oncolog" class="tab-content hidden space-y-6">
            <div class="bg-white p-6 sm:p-8 rounded-2xl border border-orange-200 shadow-sm card-print">
                <div class="flex justify-between items-center mb-6 no-print">
                    <div>
                        <h3 class="text-lg font-bold text-orange-900">Onco-Log & Smart Questions</h3>
                        <p class="text-sm text-slate-600">Perguntas estratégicas geradas para a sua próxima consulta oncológica.</p>
                    </div>
                    <button onclick="window.print()" class="bg-slate-900 hover:bg-slate-800 text-white font-bold px-4 py-2.5 rounded-xl text-sm flex items-center space-x-2 shadow-sm transition-all">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"></path></svg>
                        <span>Imprimir / Salvar PDF</span>
                    </button>
                </div>

                <div class="print-only hidden mb-6">
                    <h1 class="text-xl font-bold text-slate-900">Cancer Loser - Relatório para a Equipe Médica</h1>
                    <p class="text-xs text-slate-500">Protocolo clínico gerado em base integrativa</p>
                </div>

                <div id="oncolog-perguntas" class="space-y-3">
                    <div class="p-4 bg-[#FFF7ED] rounded-xl border border-orange-200 text-sm text-slate-700">
                        1. Como posso ajustar a medicação preventiva para náusea tardia?
                    </div>
                    <div class="p-4 bg-[#FFF7ED] rounded-xl border border-orange-200 text-sm text-slate-700">
                        2. Há necessidade de profilaxia com fator de crescimento para os leucócitos neste ciclo?
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 5: PSICO-ONCOLOGIA -->
        <section id="tab-psico" class="tab-content hidden space-y-6">
            <div class="bg-white p-6 sm:p-8 rounded-2xl border border-orange-200 shadow-sm text-center space-y-6">
                <div class="bg-gradient-to-r from-orange-500 to-amber-500 text-white p-6 rounded-2xl shadow-md">
                    <span class="text-xs uppercase tracking-widest font-bold bg-white/20 px-3 py-1 rounded-full">Mensagem do Dia</span>
                    <h3 id="psico-mensagem" class="text-xl sm:text-2xl font-bold mt-3">"O câncer não vai ganhar esta batalha. Cada dia é uma vitória."</h3>
                </div>

                <div class="bg-[#FFF7ED] p-6 rounded-2xl border border-orange-200 text-left space-y-3">
                    <h4 class="font-bold text-orange-900 flex items-center space-x-2">
                        <span>🧘 Exercício de Respiração 4-7-8 (Alívio de Ansiedade e Náusea)</span>
                    </h4>
                    <p id="psico-exercicio" class="text-sm text-slate-700 leading-relaxed">
                        Inspire profundamente pelo nariz contando até 4. Segure o ar nos pulmões contando até 7. Expire completamente pela boca fazendo som de sopro contando até 8. Repita este ciclo 4 vezes.
                    </p>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-orange-200 py-4 text-center text-xs text-orange-900/50 font-medium no-print">
        Cancer Loser &copy; 2025 &bull; Protocolos Clínicos de Apoio Oncológico MASCC / ASCO / SBOC
    </footer>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('bg-orange-500', 'text-white', 'shadow-sm');
                el.classList.add('text-orange-900/70', 'hover:bg-orange-100/50');
            });

            document.getElementById('tab-' + tabId).classList.remove('hidden');
            const activeBtn = document.getElementById('btn-' + tabId);
            activeBtn.classList.remove('text-orange-900/70', 'hover:bg-orange-100/50');
            activeBtn.classList.add('bg-orange-500', 'text-white', 'shadow-sm');
        }

        async function enviarTriagem(event) {
            event.preventDefault();
            const btn = document.getElementById('btn-submit');
            btn.innerText = "Processando Protocolo...";
            btn.disabled = true;

            const sintomas = Array.from(document.querySelectorAll('input[name="sintoma"]:checked')).map(el => el.value);
            
            const payload = {
                tratamento_atual: document.getElementById('tratamento_atual').value,
                momento_ciclo: document.getElementById('momento_ciclo').value,
                nivel_dor: parseInt(document.getElementById('nivel_dor').value),
                apetite: document.getElementById('apetite').value,
                sintomas: sintomas,
                opiniao_usuario: document.getElementById('opiniao_usuario').value,
                peso_kg: parseFloat(document.getElementById('peso_kg').value) || 70.0
            };

            try {
                const response = await fetch('/api/v1/cuidado/gerar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                renderizarDados(data);
                switchTab('resultados');
            } catch (err) {
                alert("Erro ao comunicar com o servidor. Verifique sua conexão.");
            } finally {
                btn.innerText = "Gerar Protocolo Personalizado";
                btn.disabled = false;
            }
        }

        function renderizarDados(data) {
            // Alerta Vermelho
            const alertaBox = document.getElementById('alerta-box');
            if (data.alerta_vermelho.necessita_atendimento_imediato) {
                alertaBox.classList.remove('hidden');
                alertaBox.className = "p-6 rounded-2xl border-2 border-red-500 bg-red-50 shadow-sm";
                document.getElementById('alerta_titulo').innerText = "ALERTA VERMELHO: ATENDIMENTOm MÉDICO IMEDIATO";
                document.getElementById('alerta_motivo').innerText = data.alerta_vermelho.motivo;
                const ulInst = document.getElementById('alerta_instrucoes');
                ulInst.innerHTML = data.alerta_vermelho.instrucoes_urgencia.map(i => `<li>${i}</li>`).join('');
            } else {
                alertaBox.classList.remove('hidden');
                alertaBox.className = "p-6 rounded-2xl border-2 border-emerald-500 bg-emerald-50 shadow-sm";
                document.getElementById('alerta_titulo').innerText = "Status de Segurança Estável";
                document.getElementById('alerta_motivo').innerText = data.alerta_vermelho.motivo;
                const ulInst = document.getElementById('alerta_instrucoes');
                ulInst.innerHTML = data.alerta_vermelho.instrucoes_urgencia.map(i => `<li>${i}</li>`).join('');
            }

            // Opinião IA
            document.getElementById('ia-opiniao-texto').innerText = data.analise_ia_opiniao;

            // Condutas Imediatas
            const condutasContainer = document.getElementById('condutas-container');
            condutasContainer.innerHTML = data.condutas_imediatas.map(c => `
                <div class="p-4 bg-[#FFF7ED] rounded-xl border border-orange-200">
                    <h4 class="font-bold text-orange-900">${c.passo}</h4>
                    <p class="text-sm text-slate-700 mt-1">${c.descricao}</p>
                </div>
            `).join('');

            // Nutrição
            document.getElementById('nutri-agua').innerText = data.plano_nutricional.meta_hidrica_ml + " mL";
            document.getElementById('nutri-recomendados').innerHTML = data.plano_nutricional.alimentos_recomendados.map(a => `<li>${a}</li>`).join('');
            document.getElementById('nutri-proibidos').innerHTML = data.plano_nutricional.alimentos_proibidos_seguranca.map(a => `<li>${a}</li>`).join('');
            document.getElementById('nutri-metalico').innerHTML = data.plano_nutricional.dicas_gosto_metalico.map(d => `<li>${d}</li>`).join('');

            // Onco-Log
            document.getElementById('oncolog-perguntas').innerHTML = data.onco_log.perguntas_para_medico.map((p, idx) => `
                <div class="p-4 bg-[#FFF7ED] rounded-xl border border-orange-200 text-sm text-slate-700 font-medium">
                    ${idx + 1}. ${p}
                </div>
            `).join('');

            // Psico-Oncologia
            document.getElementById('psico-mensagem').innerText = `"${data.psico_oncologia.mensagem_dia}"`;
            document.getElementById('psico-exercicio').innerText = data.psico_oncologia.exercicio_respiracao;
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)