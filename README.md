# Profile-README

# Chirag Mahaveer Chivate

This github consists of **Perosnal Projects** and a research-grade foundation in Artificial Intelligence and Machine Learning. To achieve this, I have curated a **Registry of repositories**, mirrored directly from leading research labs to audit their engineering patterns and implementation details.

---

## Personal Projects
Beyond the mirrored repositories, I have developed these personal projects to apply core ML methods for applications:

* **Reasoning & Frontiers**
    * **[Social Cognition Benchmark](https://github.com/chirag-7/Social-Cognition-Benchmark)**: A comprehensive evaluation framework for measuring progress toward AGI through cognitive and social intelligence tasks.
        * **Scope**: Evaluates LLM performance across 24+ complex reasoning tracks including **Theory of Mind (ToM)**, **Game Theory** (Liar's Dice, Centipede Game), and **Pragmatic Intent Detection**.
        * **Multimodality**: Integrates specialized datasets like **MELD** and **EMOTIC** to assess multimodal emotion and social situation recognition.
    * **[LSST Classification](https://github.com/chirag-7/LSST_Classification)**: An astronomical time-series classification pipeline developed for the LSST (Large Synoptic Survey Telescope) project.
        * **Architecture**: Implements **Temporal Fusion Transformers (TFT)** and **State Space Models (SSM)** to classify light curves from the PLAsTiCC dataset.
        * **Technical Depth**: Features a production-grade pipeline with custom data stratification for 100k+ astronomical objects, optimized using weighted log-loss and Brier score metrics.

* **MLOps & Production AI**
     * **[ML-Ops-RAG-pipeline](https://github.com/chirag-7/ML-Ops-RAG-pipeline)**: A general-purpose, production-oriented RAG playground. This project implements best practices in MLOps, featuring LakeFS for document versioning, Qdrant for vector storage, and an end-to-end observability stack with Prometheus and Grafana. It is designed to be a scalable foundation for any retrieval-augmented application.

* **Biotechnology & AI**
    * **[Protein Tuning wt RL](https://github.com/chirag-7/Protein_Tuning_RL)**: A framework for optimizing Protein Language Models (pLMs) to generate stable and functional biological sequences.
        * **Techniques**: Implements a complete **SFT (Supervised Fine-Tuning) → RL (Reinforcement Learning)** pipeline using **Weighted DPO** and **GRPO** algorithms.
        * **Optimization Strategy**: Features a hybrid training system that dynamically switches between **Full Fine-Tuning** for specialized small models (e.g., ZymCTRL) and **Low-Rank Adaptation (LoRA)** for large foundation models (e.g., BioMistral, Llama-3) to enable efficient research-grade protein design on consumer hardware.

* **AI Safety & Governance**
    * **[AI Guardrails](https://github.com/chirag-7/AI_Guardrails)**: A modular **Defense-in-Depth** framework for securing Large Language Model agents against adversarial attacks and operational failures.
        * **Architecture**: Implements a **4-Layer Defense Pipeline** (Input, Dialog, Execution, Output) using **NVIDIA NeMo Guardrails** and **Colang** to enforce strict behavioral policies.
        * **Unified Security**: Features **Input Rails** for jailbreak/injection detection, **Dialog Rails** for topic and scope enforcement, **Execution Rails** for secure RBAC tool usage, and **Output Rails** for hallucination mitigation and fact-checking.

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
