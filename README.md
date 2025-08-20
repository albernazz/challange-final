# Telecom X – Parte 2 (Modelagem Preditiva de Churn)

> **Objetivo:** construir, avaliar e interpretar modelos preditivos para estimar a probabilidade de evasão (churn) dos clientes da Telecom X, usando o dataset **tratado** gerado na Parte 1.

> **Observação de consistência:** no enunciado há menções a “DeleconX/TeleconX”. Neste README, padronizo como **Telecom X**.

---

## 1) Contexto

Na Parte 1, realizamos **ETL e EDA**: extraímos o dataset a partir da API/JSON do repositório, tratamos inconsistências (tipos, nulos, colunas aninhadas), transformamos variáveis e geramos insights iniciais sobre o churn.

Nesta **Parte 2**, assumimos o arquivo final **`TelecomX_Tratado.csv`** como ponto de partida e avançamos para:

* Seleção de variáveis relevantes;
* Codificação/normalização;
* Criação de modelos de classificação;
* Avaliação comparativa (métricas e gráficos);
* Interpretação (importâncias e recomendações de negócio);
* Relatório final.

---

## 2) Requisitos e ambiente

* **Python 3.10+**
* Bibliotecas principais:

  * `pandas`, `numpy`
  * `scikit-learn` (modelagem)
  * `matplotlib` (gráficos)
  * `scipy` (opcional)

Instalação rápida (ex.: `pip`):

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 3) Estrutura sugerida do repositório (novo)

> Conforme solicitado, **não** reutilize o repositório da Parte 1.

```
telecomx-churn-ml/
├─ data/
│  └─ TelecomX_Tratado.csv          # dataset tratado (saída da Parte 1)
├─ src/
│  ├─ utils.py                      # funções auxiliares (ex.: gráficos, métricas)
│  ├─ train.py                      # script principal de treino/avaliação
│  └─ inference.py                  # (opcional) previsão em novos dados
├─ notebooks/
│  └─ 01_modelagem_exploratória.ipynb
├─ reports/
│  └─ relatorio_final.md            # relatório final (gerado ao concluir)
├─ README.md                        # este arquivo
└─ requirements.txt                 # dependências
```

---

## 4) Pipeline de modelagem – visão geral

1. **Carregar dados** (`TelecomX_Tratado.csv`).
2. **Definir alvo**: coluna `Churn` (ou equivalente padronizada). Garantir formato binário (ex.: `Yes/No` → `1/0`).
3. **Mapeamento de tipos**:

   * Numéricas: `tenure`, `Charges.Monthly`, `Charges.Total` (ajustar conforme dataset).
   * Categóricas: todas as demais não numéricas.
4. **Pré-processamento**:

   * `OneHotEncoder` para categóricas;
   * `StandardScaler` (ou `None`) para numéricas (alguns modelos se beneficiam de escala).
5. **Split**: treino/validação (ex.: `train_test_split` estratificado, 80/20).
6. **Treino de modelos base**:

   * `LogisticRegression` (baseline forte e interpretável);
   * `RandomForestClassifier` (captura não linearidades);
   * (opcional) `GradientBoostingClassifier`.
7. **Avaliação**:

   * Métricas: `ROC AUC`, `F1`, `precision`, `recall`, `accuracy`;
   * Curva ROC, matriz de confusão;
   * Validação cruzada (k-fold) para robustez.
8. **Seleção do modelo**: comparar e escolher por `ROC AUC` (e `F1` quando classe positiva é minoritária).
9. **Interpretação**:

   * Importância de features (árvores) e `permutation_importance`;
   * Ranking das variáveis mais influentes no churn.
10. **Exportar artefatos** (opcional): pipeline treinado (`joblib`), métricas e gráficos em `reports/`.

---

## 5) Código base (script único – rápido para começar)

> Copie este bloco para `src/train.py` ou execute em notebook. Ajuste o nome da coluna-alvo se necessário.

```python
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# =====================
# 1) Carregar dataset
# =====================
DATA_PATH = Path('data/TelecomX_Tratado.csv')
df = pd.read_csv(DATA_PATH)

# Detectar coluna de churn (case-insensitive)
churn_candidates = [c for c in df.columns if 'churn' in c.lower()]
assert len(churn_candidates) >= 1, 'Coluna de Churn não encontrada.'
TARGET = churn_candidates[0]

# Garantir binarização (ex.: Yes/No → 1/0)
if df[TARGET].dtype == 'object':
    mapping = {"Yes": 1, "No": 0, "Sim": 1, "Nao": 0, "Não": 0}
    df[TARGET] = df[TARGET].map(lambda x: mapping.get(str(x), x))

df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')

# =====================
# 2) Tipos de variáveis
# =====================
# Sugeridas como numéricas – ajuste se necessário
num_cols_sug = [c for c in ['tenure', 'Charges.Monthly', 'Charges.Total'] if c in df.columns]
for c in num_cols_sug:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Demais colunas
feature_cols = [c for c in df.columns if c != TARGET]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

# Remover colunas inteiramente nulas
keep = [c for c in feature_cols if df[c].notna().any()]
num_cols = [c for c in num_cols if c in keep]
cat_cols = [c for c in cat_cols if c in keep]

# =====================
# 3) Split estratificado
# =====================
X = df[num_cols + cat_cols]
y = df[TARGET].fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 4) Pré-processamento
# =====================
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='drop'
)

# =====================
# 5) Modelos
# =====================
models = {
    'logreg': LogisticRegression(max_iter=200, n_jobs=None),
    'rf': RandomForestClassifier(n_estimators=400, random_state=42),
    'gb': GradientBoostingClassifier(random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    pipe = Pipeline(steps=[('prep', preprocess), ('clf', clf)])
    cv_scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=['roc_auc', 'f1', 'precision', 'recall', 'accuracy'],
        n_jobs=-1, return_train_score=False
    )
    results[name] = {k: np.mean(v) for k, v in cv_scores.items()}

# Exibir resultados médios de CV
print('\n===== MÉDIAS (CV=5) =====')
for name, metrics in results.items():
    print(name, {k.replace('test_', ''): round(v, 4) for k, v in metrics.items() if k.startswith('test_')})

# =====================
# 6) Ajuste final no melhor modelo (ex.: Random Forest)
# =====================
best_name = max(results, key=lambda n: results[n]['test_roc_auc'])
print(f"\nMelhor por ROC AUC (CV): {best_name}\n")

best_model = Pipeline(steps=[('prep', preprocess), ('clf', models[best_name])])
best_model.fit(X_train, y_train)

# =====================
# 7) Avaliação hold-out (teste)
# =====================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model[-1], 'predict_proba') else None

print('Hold-out:')
print('ROC AUC :', round(roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan, 4))
print('F1      :', round(f1_score(y_test, y_pred), 4))
print('Precision:', round(precision_score(y_test, y_pred), 4))
print('Recall  :', round(recall_score(y_test, y_pred), 4))
print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))

# Curva ROC
if y_proba is not None:
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f'Curva ROC – {best_name}')
    plt.show()

# Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f'Matriz de Confusão – {best_name}')
plt.show()

# =====================
# 8) Importância de atributos (permutation importance)
# =====================
# Para obter nomes após OneHot, usamos o atributo get_feature_names_out
prep = best_model.named_steps['prep']
X_test_prep = prep.transform(X_test)

try:
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = perm.importances_mean
    # Nomes: num + one-hot
    feature_names = []
    if num_cols:
        feature_names += list(num_cols)
    if cat_cols:
        cat_names = best_model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names += list(cat_names)

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)

    print('\nTop 20 features (permutation importance):')
    print(imp_df)

    imp_df.plot(kind='barh', x='feature', y='importance', legend=False)
    plt.gca().invert_yaxis()
    plt.title('Importância de Atributos (Permutation) – Top 20')
    plt.xlabel('Impacto na métrica')
    plt.ylabel('Atributo')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print('Falha ao calcular importâncias por permutação:', e)
```

---

## 6) Boas práticas e dicas

* **Classe desbalanceada**: se `Churn=1` for minoritário, monitore `Recall` e `F1`. Pode testar `class_weight='balanced'` na *Logistic Regression* ou no *Random Forest*.
* **Threshold**: além do `0.5`, avalie limiares diferentes para otimizar `Recall` (reter clientes) ou `Precision` (evitar falsos positivos em campanhas).
* **Validação**: mantenha `StratifiedKFold` para preservar a proporção da classe positiva.
* **Leakage**: evite usar variáveis pós-fato (ex.: `Charges.Total` pode crescer com o tempo; se estiver muito colinear com `tenure`, valide impacto na generalização).

---

## 7) Relatório final – roteiro (modelo)

Use este roteiro em `reports/relatorio_final.md`:

1. **Introdução**

   * Objetivo da modelagem (prever churn, apoiar retenção, priorização de clientes em risco).
2. **Dados**

   * Fonte: `TelecomX_Tratado.csv` (derivado da Parte 1).
   * Descrição do alvo (`Churn`) e principais variáveis.
3. **Metodologia**

   * Pipeline (pré-processamento, split, modelos testados, CV).
4. **Resultados**

   * Tabela com métricas médias de CV (ROC AUC, F1, precision, recall, accuracy) por modelo.
   * Métricas em *hold-out* para o modelo selecionado.
   * Gráficos: Curva ROC, Matriz de Confusão, Importância de Atributos.
5. **Discussão/Interpretação**

   * Principais variáveis associadas ao churn (ex.: `Contract`, `tenure`, `TechSupport`, `OnlineSecurity`, `Charges.Monthly`).
   * Trade-offs de threshold e impacto no negócio.
6. **Recomendações de negócio** (exemplos)

   * Incentivar **contratos de maior duração** com benefícios progressivos.
   * Ofertar **suporte técnico** e **segurança online** como *bundles* para clientes em risco.
   * **Monitorar clientes novos** (baixa `tenure`) com onboarding ativo.
   * Campanhas focadas em perfis com **alto ticket mensal** e baixa fidelização.
7. **Próximos passos**

   * *Hyperparameter Tuning* (Grid/Random/Bayes).
   * Testar *Gradient Boosting* mais forte (ex.: XGBoost/LightGBM).
   * Implementar **pipeline de inferência** e **monitoramento** (drift de dados/métricas).

---

## 8) Checklist (Trello → cards sugeridos)

* **Material de apoio**: links e instruções de uso do Trello.
* **Desafio – Parte 2 (overview)**: escopo, prazos, critérios de aceite.
* **Criar novo repositório (GitHub)**: estrutura inicial e `requirements.txt`.
* **Preparar dados para ML**: validar `TelecomX_Tratado.csv`, tipos e nulos.
* **Seleção de variáveis**: mapa de dados e justificativas.
* **Definir alvo e métrica primária**: `Churn` e *ROC AUC* (com `F1` de apoio).
* **Construir pipeline de pré-processamento**.
* **Treinar modelos baseline** (LogReg, RF, GB) com CV.
* **Comparar e selecionar modelo**: tabela de métricas + decisão.
* **Avaliação hold-out**: curva ROC e matriz de confusão.
* **Interpretação**: importâncias/permutation + narrativa.
* **Relatório final**: montar `reports/relatorio_final.md`.
* **Apresentação** (opcional): slides com problema, solução e resultados.

---

## 9) Como rodar rapidamente

```bash
# 1) criar venv (opcional)
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate                           # (Windows PowerShell)

# 2) instalar dependências
pip install -r requirements.txt

# 3) preparar dados
# Coloque TelecomX_Tratado.csv em data/

# 4) treinar e avaliar
python src/train.py
```

**Bom trabalho!** Ao finalizar, não esqueça de preencher o relatório com as métricas e os gráficos gerados, e redigir as recomendações de negócio com base nas variáveis mais influentes no churn.
