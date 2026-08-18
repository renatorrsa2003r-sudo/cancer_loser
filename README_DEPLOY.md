# Como Hospedar o cancer_loser na Internet Gratuitamente

### Opção Recomendada: Render.com
1. Crie uma conta gratuita em [Render.com](https://render.com).
2. Suba a pasta deste projeto para um repositório no seu GitHub.
3. No painel da Render, clique em **New +** -> **Web Service**.
4. Selecione seu repositório do GitHub.
5. Em **Environment**, escolha `Python 3`.
6. Em **Start Command**, coloque: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Em **Environment Variables**, adicione sua chave:
   - `GEMINI_API_KEY` = (sua chave do Google AI Studio)
8. Clique em **Deploy Web Service**. O site ficará no ar com HTTPS gratuito!
