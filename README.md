# Profile-README

# Chirag Mahaveer Chivate

This github consists of **Perosnal Projects** in my areas of interest and a research-grade foundation in Artificial Intelligence and Machine Learning. To achieve this, I have curated a **Registry of repositories**, mirrored directly from leading research labs to audit their engineering patterns and implementation details.

---

## Personal Projects
Beyond the mirrored repositories, I have developed these personal projects to apply core ML methods for applications in my areas of interest:

* **Reasoning & Frontiers**
    * **[Social Cognition Benchmark](https://github.com/chirag-7/Social-Cognition-Benchmark)**: A comprehensive evaluation framework for measuring progress toward AGI through cognitive and social intelligence tasks.
        * **Scope**: Evaluates LLM performance across 24+ complex reasoning tracks including **Theory of Mind (ToM)**, **Game Theory** (Liar's Dice, Centipede Game), and **Pragmatic Intent Detection**.
        * **Multimodality**: Integrates specialized datasets like **MELD** and **EMOTIC** to assess multimodal emotion and social situation recognition.
    * **[LSST Classification](https://github.com/chirag-7/LSST_Classification)**: An astronomical time-series classification pipeline developed for the LSST (Large Synoptic Survey Telescope) project.
        * **Architecture**: Implements **Temporal Fusion Transformers (TFT)** and **State Space Models (SSM)** to classify light curves from the PLAsTiCC dataset.
        * **Technical Depth**: Features a production-grade pipeline with custom data stratification for 100k+ astronomical objects, optimized using weighted log-loss and Brier score metrics.

* **AI Safety & Governance**
    * **[AI Guardrails](https://github.com/chirag-7/AI_Guardrails)**: A modular **Defense-in-Depth** framework for securing Large Language Model agents against adversarial attacks and operational failures.
        * **Architecture**: Implements a **4-Layer Defense Pipeline** (Input, Dialog, Execution, Output) using **NVIDIA NeMo Guardrails** and **Colang** to enforce strict behavioral policies.
        * **Unified Security**: Features **Input Rails** for jailbreak/injection detection, **Dialog Rails** for topic and scope enforcement, **Execution Rails** for secure RBAC tool usage, and **Output Rails** for hallucination mitigation and fact-checking.

- **Data Science**
  - **[Customer Retention Intelligence System](https://github.com/chirag-7/Customer-Retention-Intelligence-System-From-Churn-Prediction-to-Intervention-Strategy)**: Cost-sensitive customer-churn modelling and intervention optimisation project.
    - **Architecture**: Implements preprocessing, calibrated classification, threshold selection, and campaign-cost decision logic on Orange Telecom customer data.
    - **Technical Depth**: Evaluates an 80/20 stratified holdout using PR-AUC and ROC-AUC, with source targets of **0.906 PR-AUC** and **0.941 ROC-AUC**.
  - **[BERT4Rec Sequential Recommendation System](https://github.com/chirag-7/BERT4Rec-Sequential-Recommendation-System)**: Sequential recommendation project using masked-item Transformer learning.
    - **Architecture**: Learns bidirectional item and positional embeddings from MovieLens-1M histories and ranks unseen items using leave-one-out evaluation.
    - **Technical Depth**: Implements full-corpus ranking, HR@K, NDCG@K, and MRR; source results report **HR@10 0.2901** and **NDCG@10 0.1624**.
  - **[M5 Forecasting Accuracy](https://github.com/chirag-7/kaggle-m5-forecasting-accuracy)**: Hierarchical retail-demand forecasting project based on the M5 Walmart competition.
    - **Architecture**: Engineers calendar, price, lag, and rolling-window features for LightGBM ensembles across the sales hierarchy.
    - **Technical Depth**: Produces 28-day forecasts for **42,840** daily series and evaluates weighted RMSSE, targeting source **WRMSSE 0.53583**.

- **Industrial Analytics & Quality Engineering**
  - **[Anomaly Transformer](https://github.com/chirag-7/Anomaly-Transformer)**: Multivariate industrial time-series anomaly-detection project.
    - **Architecture**: Uses association discrepancy between prior and series attention distributions to identify anomalous telemetry windows.
    - **Technical Depth**: Evaluates point-adjusted event F1 on public industrial datasets, with source F1 **92.33** on SMD, **94.07** on SWaT, and **97.89** on PSM.
  - **[Turbofan Engine RUL Prediction](https://github.com/chirag-7/Turbofan-engine-RUL-prediction)**: Remaining-useful-life forecasting project for NASA C-MAPSS turbofan sensor trajectories.
    - **Architecture**: Implements windowed sensor preprocessing, attention-LSTM sequence modelling, and engine-level RUL prediction.
    - **Technical Depth**: Evaluates maintenance forecasts on NASA FD001, targeting source test **RMSE 14.7562**.
  - **[PatchCore Industrial Inspection](https://github.com/chirag-7/patchcore-inspection)**: Visual anomaly-detection project for industrial quality inspection.
    - **Architecture**: Extracts local image patches, builds nearest-neighbour memory banks, and generates image- and pixel-level anomaly maps.
    - **Technical Depth**: Evaluates MVTec AD defects using image AUROC, pixel AUROC, and PRO; source mean image AUROC is **0.9955** and pixel AUROC **0.9823**.

- **Biostatistics & Computational Biology**
  - **[scPhase](https://github.com/chirag-7/scPhase)**: Patient-aware single-cell RNA-seq phenotype-prediction project.
    - **Architecture**: Uses attention-enhanced cell representations and patient-level cross-validation for disease classification.
    - **Technical Depth**: Evaluates cohort-level AUC, with source results of **0.895** for COVID, **0.951** for NSCLC, and **0.962** for colorectal cancer.
  - **[DeepHit](https://github.com/chirag-7/DeepHit)**: Neural competing-risk survival-analysis project.
    - **Architecture**: Combines discrete-time survival likelihood and ranking losses to model censored time-to-event outcomes.
    - **Technical Depth**: Reproduces the METABRIC clinical-survival benchmark, targeting published time-dependent C-index **0.675** and integrated Brier score **0.186**.
  - **[CLEAN](https://github.com/chirag-7/CLEAN)**: Contrastive protein-function annotation project.
    - **Architecture**: Learns protein-sequence embeddings using contrastive/triplet objectives for enzyme-commission classification.
    - **Technical Depth**: Evaluates Swiss-Prot functional prediction with source precision **0.596**, recall **0.479**, F1 **0.497**, and AUC **0.739**.

- **AI Model Training & Post-Training**
  - **[LXMERT](https://github.com/chirag-7/lxmert)**: Multimodal vision-language pretraining project.
    - **Architecture**: Uses separate visual, language, and cross-modal Transformer encoders to learn aligned image-text representations on COCO/VQA data.
    - **Technical Depth**: Supports downscaled pretraining and VQA fine-tuning; the upstream VQA v2 test-dev reference is **72.42%** accuracy.
  - **[QLoRA Fine-Tuning](https://github.com/chirag-7/qlora-finetune)**: Memory-efficient instruction-tuning project for open-weight LLMs.
    - **Architecture**: Combines 4-bit NF4 quantisation, paged optimisers, and low-rank adapters for single-GPU supervised fine-tuning.
    - **Technical Depth**: Reproduces the Guanaco evaluation workflow, whose source reports **99.3%** of ChatGPT’s GPT-4-judged Vicuna benchmark score.
  - **[GSM8K RLVR](https://github.com/chirag-7/GSM8K-RLVR)**: Reinforcement learning with verifiable rewards for mathematical reasoning.
    - **Architecture**: Optimises a Qwen2.5-Math policy using exact-answer reward parsing instead of a learned reward model.
    - **Technical Depth**: Evaluates GSM8K accuracy, with source Qwen2.5-Math-1.5B performance improving from **70.66%** to **77.33%**.

- **AI Engineering, Agents & Evaluation**
  - **[Atlas](https://github.com/chirag-7/atlas)**: Retrieval-augmented question-answering project for few-shot knowledge-intensive NLP.
    - **Architecture**: Combines dense passage retrieval with a generative reader over a Wikipedia corpus.
    - **Technical Depth**: Evaluates Natural Questions exact match, targeting source **EM 38.4** on 64-shot development and **EM 38.8** on test.
  - **[AgentRx](https://github.com/chirag-7/AgentRx)**: Failure-diagnosis and reliability-analysis project for tool-using AI agents.
    - **Architecture**: Normalises execution traces, synthesises guarded invariants, checks violations, uses an LLM judge, and exports root-cause reports.
    - **Technical Depth**: Evaluates **115** labelled failed trajectories and reports source improvements of **23.6 percentage points** in failure localisation and **22.9 points** in root-cause attribution.
  - **[LangSmith Document-Extraction Evaluations](https://github.com/chirag-7/langsmith-evaluations-doc-extraction)**: Gold-set regression-testing project for structured LLM document extraction.
    - **Architecture**: Logs traces, field-level evaluator results, latency percentiles, and model cost across prompt/model variants.
    - **Technical Depth**: Tracks extraction-quality regressions, with source evaluation scores of approximately **0.73** for GPT-4o and **0.93** for o1.

- **Middle & Back Office Quantitative Risk**
  - **[Market-Risk Copula VaR/ES](https://github.com/chirag-7/market-risk-copula-var-es)**: Market-risk forecasting and regulatory-backtesting project.
    - **Architecture**: Implements historical, parametric, filtered-historical, and copula VaR/ES models with Kupiec, Christoffersen, and Acerbi–Székely tests.
    - **Technical Depth**: Produces **24** 95% VaR exceptions in **370** observations versus **18.5** expected; source ES-test p-values are **0.0536** for historical ES and **0.5616** for GARCH-t ES.
  - **[Credit-Card Default Prediction](https://github.com/chirag-7/Credit-Card-Default-Prediction)**: Retail-credit probability-of-default modelling project.
    - **Architecture**: Automates UCI data retrieval, cleaning, feature preparation, model selection, and holdout evaluation.
    - **Technical Depth**: Targets logistic-regression test ROC-AUC **0.767**, F1 **0.531**, recall **0.574**, and precision **0.494**.
  - **[ML Quant Trading](https://github.com/chirag-7/ml-quant-trading)**: Risk-aware CSI 300 factor-allocation and portfolio-backtesting project.
    - **Architecture**: Builds rolling factor features, delayed-execution signals, transaction-cost/slippage modelling, turnover controls, and attribution reports.
    - **Technical Depth**: Targets source 2021–2024 annualised return **22.20%**, volatility **25.26%**, Sharpe **0.919**, turnover **0.1397**, and final equity **2.1616**.

- **Front Office Quantitative Risk & Derivatives**
  - **[Heston Calibration](https://github.com/chirag-7/Heston_Calibration)**: Equity-index volatility-surface calibration project.
    - **Architecture**: Implements COS option pricing and constrained Levenberg–Marquardt calibration of Heston stochastic-volatility parameters.
    - **Technical Depth**: Calibrates an SPX option chain with a checked-in implied-volatility RMSE of **3.4441** across **79** options.
  - **[Empirical Deep Hedging](https://github.com/chirag-7/Empirical-Deep-Hedging)**: Transaction-cost-aware deep-hedging project for S&P 500 options.
    - **Architecture**: Trains TD3 hedge-ratio policies under GBM/Heston settings and compares them with Black–Scholes delta hedging.
    - **Technical Depth**: Under **1 bp** costs and five-day episodes, source Heston-policy P&L is **−0.0095%** versus **−0.0195%** for Black–Scholes, with reward **4.375** versus **4.204**.
  - **[Monte Carlo Risk Engine](https://github.com/chirag-7/montecarlo-risk-engine)**: Counterparty-exposure and wrong-way-risk simulation project.
    - **Architecture**: Simulates Vasicek rates, CIR++ default intensity, collateralised exposure, EE/PFE/CVA, Monte Carlo errors, and Greeks.
    - **Technical Depth**: Models payer-swap wrong-way risk, with CVA increasing from approximately **1.077** at correlation ρ = **−0.95** to **1.140** at ρ = **0.95**, versus uncorrelated CVA **1.1146**.

- **Mid-Frequency Trading & Quantitative Research**
  - **[Return-Prediction Signal Research](https://github.com/chirag-7/Return-prediction-signal)**: Point-in-time cross-sectional equity-alpha research project.
    - **Architecture**: Implements purged and embargoed validation, Newey–West IC inference, factor attribution, and **10 bp per-side** trading costs.
    - **Technical Depth**: Tests survivorship and multiple-testing robustness; the source 12–1 momentum result is net Sharpe **0.01**, DSR **0.35**, and IC **0.0052**.
  - **[Trading Momentum Transformer](https://github.com/chirag-7/trading-momentum-transformer)**: Continuous-futures momentum and regime-detection project.
    - **Architecture**: Combines LSTM/attention trend estimation, online change-point detection, volatility scaling, and fast-reversion overlays across 50 liquid futures.
    - **Technical Depth**: Targets source 1995–2020 raw out-of-sample Sharpe **2.16** under documented **0–5 bp** transaction-cost sensitivity.
  - **[Factor Optimizer](https://github.com/chirag-7/Factor-Optimizer)**: Factor-based constrained portfolio-construction project.
    - **Architecture**: Combines Fama–French five-factor-plus-momentum modelling, covariance shrinkage, CVaR optimisation, bootstrap inference, and deflated-Sharpe validation.
    - **Technical Depth**: Includes **10 bp** transaction costs and targets source Sharpe **1.45**, DSR **0.934**, annualised return **19.7%**, and drawdown **−14.4%**.

- **High-Frequency Trading & Market Microstructure**
  - **[TLOB-2](https://github.com/chirag-7/TLOB-2)**: Dual-attention Transformer project for limit-order-book price-trend prediction.
    - **Architecture**: Combines temporal and spatial attention to model multi-level order-book dynamics on FI-2010, equity LOBSTER, and Bitcoin data.
    - **Technical Depth**: Evaluates chronological F1 across forecasting horizons and targets the source paper’s average **+3.7 F1-score-point** improvement over prior FI-2010 methods.
  - **[atlas-mm](https://github.com/chirag-7/atlas-mm)**: L2 limit-order-book simulation and market-making project.
    - **Architecture**: Implements price-time-priority matching, Poisson order flow, Avellaneda–Stoikov quoting, PPO policies, inventory limits, and fill accounting.
    - **Technical Depth**: Fixed seed-42 results report A–S P&L **−1.69** with **5.68%** fill rate versus PPO P&L **−19.23** with **28.48%** fill rate.
  - **[Queue-Reactive Optimal Execution](https://github.com/chirag-7/qrm_optimal_execution)**: Queue-reactive execution project using calibrated order-arrival intensities and Double DQN.
    - **Architecture**: Models queue state, order-flow intensities, limit-order actions, and execution completion through a calibrated market simulator.
    - **Technical Depth**: Evaluates **20,000** saved paths; the best 5-state/3-action policy has mean implementation-shortfall score **−0.2591** versus **−0.3647** for TWAP.

- **Quantitative Asset Management & Portfolio Allocation**
  - **[Regime-Switching Portfolio](https://github.com/chirag-7/regime-switching-portfolio)**: Regime-aware multi-asset allocation project.
    - **Architecture**: Uses a three-state Gaussian HMM, regime-conditioned moments, shrinkage covariance, volatility targeting, and constrained SPY/QQQ/TLT/GLD allocation.
    - **Technical Depth**: The 2012–2025 source backtest with **5 bp** per-turnover-unit costs reports Sharpe **0.8986**, annual return **11.51%**, volatility **12.13%**, and drawdown **−27.19%**.
  - **[Regime-Sensitive Black–Litterman Tri-Market Study](https://github.com/chirag-7/regime-sensitive-black-litterman-tri-market-study)**: Cross-market strategic-allocation project for US, China, and India ETF baskets.
    - **Architecture**: Combines Black–Litterman views, Ledoit–Wolf covariance, turnover controls, regime analysis, crisis recovery, and factor attribution.
    - **Technical Depth**: Reports source BL/Markowitz Sharpe of **0.650/0.614** for US, **0.042/0.088** for China, and **0.356/0.440** for India.
  - **[FIN496 Foundation Project](https://github.com/chirag-7/FIN496-Foundation-Project)**: Strategic-plus-tactical multi-asset allocation and attribution project.
    - **Architecture**: Implements HMM regimes, IPS constraints, volatility targeting, walk-forward validation, transaction costs, and portfolio attribution.
    - **Technical Depth**: The canonical five-fold run reports annualised return **8.39%**, volatility **7.25%**, Sharpe **0.881**, Sortino **1.245**, drawdown **−21.92%**, and **zero** hard IPS violations.

* **Financial & Causal ML**
    * **[DRW Crypto Market Prediction](https://github.com/chirag-7/DRW-Crypto-Market-Prediction-Kaggle-)**: **4th Place Solution** (Top 0.3% of 1,448 participants) for predicting high-frequency next-tick returns.
        * **Strategy**: A segmented dual-model ensemble using **ARDRegression** for sparse linear signals and **XGBoost** for non-linear interactions, achieving a ~0.1192 Pearson correlation.
    * **[Rossmann Sales Forecasting](https://github.com/chirag-7/rossmann-sales-forecasting-attention)**: Time-series forecasting using Prophet and LSTM with Attention to predict revenue for 1,115 stores.
    * **[Causal Effect Analysis](https://github.com/chirag-7/Minimum-wage-and-employemnt-casual-effect-regression-analysis)**: Regression analysis investigating the causal relationship between minimum wage and employment.
    * **[RL Trading Agent](https://github.com/chirag-7/RL-Trading-Agent)**: Exploratory implementation of Reinforcement Learning for trading strategy optimization.

* **Deep Learning & Systems**
    * **[MRNet Medical Imaging](https://github.com/chirag-7/MRNet-Deep-Learning)**: Deep Learning applications for medical image analysis.
    * **[POI Recommendation System](https://github.com/chirag-7/next-poi-travel-recommendation)**: A recommendation system for predicting travel itineraries from check-in data.

* **Data Analysis**
    * **[USDA Branded Foods](https://github.com/chirag-7/USDA-Branded-Foods)**: Large-scale data analysis and visualization of food databases.
    * **[FIFA 20 Analysis](https://github.com/chirag-7/FIFA_20_Analysis)**: Comprehensive data analysis project on FIFA 20 player statistics.

* **MLOps & Production AI**
     * **[ML-Ops-RAG-pipeline](https://github.com/chirag-7/ML-Ops-RAG-pipeline)**: A general-purpose, production-oriented RAG playground. This project implements best practices in MLOps, featuring LakeFS for document versioning, Qdrant for vector storage, and an end-to-end observability stack with Prometheus and Grafana. It is designed to be a scalable foundation for any retrieval-augmented application.

* **Biotechnology & AI**
    * **[Protein Tuning wt RL](https://github.com/chirag-7/Protein_Tuning_RL)**: A framework for optimizing Protein Language Models (pLMs) to generate stable and functional biological sequences.
        * **Techniques**: Implements a complete **SFT (Supervised Fine-Tuning) → RL (Reinforcement Learning)** pipeline using **Weighted DPO** and **GRPO** algorithms.
        * **Optimization Strategy**: Features a hybrid training system that dynamically switches between **Full Fine-Tuning** for specialized small models (e.g., ZymCTRL) and **Low-Rank Adaptation (LoRA)** for large foundation models (e.g., BioMistral, Llama-3) to enable efficient research-grade protein design on consumer hardware.

---

## The Project Master Registry
This table organizes the mirrors I maintain across key research axes.

| Career Cluster | Key Mirrored Projects |
| :--- | :--- |
| **1. AI Foundations** | [nanoGPT](https://github.com/chirag-7/nanoGPT), [minbpe](https://github.com/chirag-7/minbpe), [BitNet](https://github.com/chirag-7/BitNet), [TinyLlama](https://github.com/chirag-7/TinyLlama), [Mamba](https://github.com/chirag-7/mamba), [mistral-src](https://github.com/chirag-7/mistral-src), [mixture-of-experts](https://github.com/chirag-7/mixture-of-experts), [rotary-embedding-torch](https://github.com/chirag-7/rotary-embedding-torch), [llama3-from-scratch](https://github.com/chirag-7/llama3-from-scratch), [annotated-transformer](https://github.com/chirag-7/annotated-transformer), [Nano-GPT in C++](https://github.com/chirag-7/Nano-GPT-in-C-), [Encoder-Decoder-Transformer](https://github.com/chirag-7/Encoder-Decoder-Transformer), [Samba](https://github.com/chirag-7/Samba) |
| **2. Agents & Search** | [OpenDevin](https://github.com/chirag-7/OpenDevin), [Storm](https://github.com/chirag-7/storm), [Tree-of-Thought](https://github.com/chirag-7/tree-of-thought-llm), [LangGraph](https://github.com/chirag-7/langgraph), [500-AI-Agents-Projects](https://github.com/chirag-7/500-AI-Agents-Projects), [AutoGPT](https://github.com/chirag-7/AutoGPT), [MemGPT](https://github.com/chirag-7/MemGPT), [ChatArena](https://github.com/chirag-7/chatarena), [Verba](https://github.com/chirag-7/verba), [GPTScript](https://github.com/chirag-7/gptscript), [Semantic-Router](https://github.com/chirag-7/semantic-router), [ContextGem](https://github.com/chirag-7/contextgem) |
| **3. Reasoning** | [prm800k](https://github.com/chirag-7/prm800k), [Alpha-Zero-General](https://github.com/chirag-7/alpha-zero-general), [AlphaGeometry](https://github.com/chirag-7/alphageometry), [MCTS](https://github.com/chirag-7/MCTS), [ReasoningAI](https://github.com/chirag-7/ReasoningAI), [Reasoning-Models](https://github.com/chirag-7/reasoning-models), [Reflexion-Human-Eval](https://github.com/chirag-7/reflexion-human-eval), [TinyZero](https://github.com/chirag-7/TinyZero), [PAL](https://github.com/chirag-7/pal) |
| **4. Interpretability** | [ROME](https://github.com/chirag-7/rome), [SAELens](https://github.com/chirag-7/SAELens), [TransformerLens](https://github.com/chirag-7/TransformerLens), [Circuits](https://github.com/chirag-7/circuits), [Grokking](https://github.com/chirag-7/Grokking), [Mergekit](https://github.com/chirag-7/mergekit), [BertViz](https://github.com/chirag-7/bertviz), [Causal-Tracing](https://github.com/chirag-7/causal-tracing), [Interp-Tutorial](https://github.com/chirag-7/eacl2024_transformer_interpretability_tutorial), [Nano-GPT in C++](https://github.com/chirag-7/Nano-GPT-in-C-) |
| **5. High-Perf Systems** | [vLLM](https://github.com/chirag-7/vllm), [Triton](https://github.com/chirag-7/triton), [DeepSpeed](https://github.com/chirag-7/DeepSpeed), [AutoAWQ](https://github.com/chirag-7/AutoAWQ), [FastChat](https://github.com/chirag-7/FastChat), [LMCache](https://github.com/chirag-7/LMCache), [TransformerEngine](https://github.com/chirag-7/TransformerEngine) |
| **6. Financial ML** | [FinRL](https://github.com/chirag-7/FinRL), [DeepLOB](https://github.com/chirag-7/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books), [PFHedge](https://github.com/chirag-7/pfhedge), [Darts](https://github.com/chirag-7/darts), [EconML](https://github.com/chirag-7/EconML), [QuantStats](https://github.com/chirag-7/quantstats), [Financial-Sentiment](https://github.com/chirag-7/Financial-Sentiment-Analysis), [Alpha_Vantage](https://github.com/chirag-7/alpha_vantage), [Causal_Segmentation](https://github.com/chirag-7/AC_NFsh_causal_segmentation), [CorrMat-Nets](https://github.com/chirag-7/BH-SPD-CorrMat-Nets) |
| **7. Safety & Trust** | [NeMo-Guardrails](https://github.com/chirag-7/NeMo-Guardrails), [Opacus](https://github.com/chirag-7/opacus), [PySyft](https://github.com/chirag-7/PySyft), [AIF360](https://github.com/chirag-7/AIF360), [Rebuff](https://github.com/chirag-7/rebuff), [Flower](https://github.com/chirag-7/flower), [LLM-Attacks](https://github.com/chirag-7/llm-attacks), [LM-Watermarking](https://github.com/chirag-7/lm-watermarking), [DoubleML-for-Py](https://github.com/chirag-7/doubleml-for-py) |
| **8. Training & Align** | [Unsloth](https://github.com/chirag-7/unsloth), [LLaMA-Factory](https://github.com/chirag-7/LLaMA-Factory), [DPO](https://github.com/chirag-7/direct-preference-optimization), [DistillKit](https://github.com/chirag-7/DistillKit), [TRL](https://github.com/chirag-7/trl), [QLoRA](https://github.com/chirag-7/qlora), [Self-Rewarding-LM](https://github.com/chirag-7/self-rewarding-lm-pytorch), [UltraFeedback](https://github.com/chirag-7/UltraFeedback), [LLMs-from-Scratch](https://github.com/chirag-7/LLMs-from-scratch) |
| **9. MLOps** | [BentoML](https://github.com/chirag-7/BentoML), [MLflow-Example](https://github.com/chirag-7/mlflow-example), [Hydra](https://github.com/chirag-7/hydra), [ZenML](https://github.com/chirag-7/zenml), [Evidently](https://github.com/chirag-7/evidently), [Dedupe](https://github.com/chirag-7/dedupe), [DataTrove](https://github.com/chirag-7/datatrove), [Model-Card-Generator](https://github.com/chirag-7/model-card-generator), [End-to-End-ML](https://github.com/chirag-7/End-to-End-ML) |
| **10. Multimodal** | [CLIP](https://github.com/chirag-7/CLIP), [Stable-Diffusion](https://github.com/chirag-7/stable-diffusion-pytorch), [AlphaFold](https://github.com/chirag-7/alphafold), [BioGPT](https://github.com/chirag-7/BioGPT), [AI-for-Science](https://github.com/chirag-7/End-to-End-AI-for-Science), [Multimodal-Tools](https://github.com/chirag-7/multimodal-tools), [Recommenders](https://github.com/chirag-7/recommenders) |
| **11. World Models** | [vjepa2](https://github.com/chirag-7/vjepa2) |
| **12. Nested Learning** | [HOPE-Architecture](https://github.com/chirag-7/HOPE-Architecture), [CMS-Mirror](https://github.com/chirag-7/CMS-Mirror) |
| **13.  Memory** | [Engram](https://github.com/chirag-7/Engram) |
| **14. Symbolic Reasoning** | [AlphaGeometry2](https://github.com/chirag-7/AlphaGeometry2), [DeepSeekMath-V2](https://github.com/chirag-7/DeepSeekMath-V2) |
| **15. Agentic Reliability** | [guardrails-ai](https://github.com/chirag-7/guardrails-ai), [SpecGuard](https://github.com/chirag-7/SpecGuard) |
| **16. Frontier Alignment** | [GRPO-implementation](https://github.com/chirag-7/GRPO-implementation) |

---
📧 **Contact**: [LinkedIn](https://www.linkedin.com/in/chirag-chivate-297209133) | chirag_chivate@yahoo.com
