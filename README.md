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

* **Data Engineering, Data Science & MLOps Pipelines**
    * **[Smart Traffic Urban Traffic Intelligence](https://github.com/chirag-7/Smart-Traffic-Urban-Traffic-Intelligence)**: An end-to-end real-time intelligent transportation system and demand forecasting infrastructure.
        * **Architecture**: Deploys a streaming data ingestion and synchronization layer utilizing Apache Kafka paired with Spark Structured Streaming and Delta Lake storage.
        * **Technical Depth**: Packages containerized LightGBM models optimized via an ONNX Runtime layer to deliver sub-second telemetry predictions monitored via integrated Prometheus and Grafana dashboards.
    * **[Fraud Detection MLOps Pipeline](https://github.com/chirag-7/Fraud-Detection-MLOps-Pipeline)**: Automated enterprise MLOps ecosystem designed for real-time transactional financial fraud detection.
        * **Architecture**: Implements a resilient orchestration workflow utilizing Confluent Kafka and Apache Spark Structured Streaming managed via automated Apache Airflow DAGs.
        * **Technical Depth**: Integrates an end-to-end MLflow model tracking lifecycle with automated versioning and asset logging over AWS S3-compatible object storage layers.
    * **[Retail Demand Forecasting](https://github.com/chirag-7/Retail-demand-forecasting)**: Production-grade supply chain demand prediction platform built for multi-stage operational safety.
        * **Architecture**: Leverages Data Version Control (DVC) pipelines to programmatically track data mutations, feature extractions, and data preparation runs.
        * **Technical Depth**: Coordinates unified experiment run logging and deployment tasks via an integrated containerized pipeline using MLflow registries and Docker microservices.

* **AI Engineering with Foundation Models & Harness Evaluation**
    * **[KnowledgeGraphQA Langgraph](https://github.com/chirag-7/KnowledgeGraphQA-Langgraph)**: An autonomous text-to-graph Knowledge Mining and complex question-answering architecture.
        * **Architecture**: Links stateful LangGraph agent nodes with a Neo4j graph database backend to traverse and map complex multi-hop entities.
        * **Technical Depth**: Implements automated cyclical reflection loops to dynamically translate natural language queries into deterministic Cypher expressions with vector-fallback indices.
    * **[LLM Medical Finetuning](https://github.com/chirag-7/LLM-Medical-Finetuning)**: Domain-specific fine-tuning engine optimized for specialized medical dialogue translation.
        * **Architecture**: Adapts 8B parameter open-source foundational models on targeted biomedical instruct data pools.
        * **Technical Depth**: Implements memory-efficient 4-bit QLoRA structural adapters using Unsloth and BitsAndBytes to reduce VRAM footprints and maximize Model Flops Utilization (MFU).
    * **[Expert Finder Eval Harness](https://github.com/chirag-7/Expert-Finder-eval-harness)**: A continuous integration testing and alignment evaluation harness engineered for retrieval systems.
        * **Architecture**: Constructs an offline-testable validation benchmark over an OpenAlex textual corpus using Promptfoo and Python assertion runners.
        * **Technical Depth**: Implements rubric-based verification frameworks alongside a self-validating LLM judge audited via Cohen's $\kappa$ agreement metrics and positional bias flip-rate probes.

* **Industrial Analytics & Quality Engineering**
    * **[Predictive Maintenance MLOps](https://github.com/chirag-7/predictive-maintenance-mlops)**: High-frequency IoT tool Fault Detection and Classification (FDC) streaming platform.
        * **Architecture**: Connects industrial factory machine telemetry feeds directly into a streaming predictive pipeline.
        * **Technical Depth**: Serves containerized Gradient Boosting classifiers via FastAPI on AWS ECS clusters to process real-time degradation metrics under sub-100ms latency limits.
    * **[Semiconductor Wafer Defect Classification](https://github.com/chirag-7/Semiconductor-Wafer-Defect-Classification)**: Spatial wafer bin map defect recognition and yield attribution system.
        * **Architecture**: Automates cleanroom defect pattern tracking over the benchmark WM-811K semiconductor wafer geometric spatial dataset.
        * **Technical Depth**: Extracts advanced structural geometric, density, and Radon transform features to drive ensemble random forest models for automated yield inspection.
    * **[SPC](https://github.com/chirag-7/SPC)**: Manufacturing quality analytics platform executing automated Statistical Process Control metrics.
        * **Architecture**: Assesses hardware fabrication line stability by processing raw measurements into continuous historical monitoring arrays.
        * **Technical Depth**: Programs a deterministic quality rule engine following the Western Electric pattern rules to calculate Mean $\bar{X}$ and Range $R$ variances, instantly flagging equipment drift.

* **Biostatistics & Computational Biology**
    * **[scRNA Seq ScanPy](https://github.com/chirag-7/scRNA-Seq-ScanPy)**: High-dimensional transcriptomic sequence processing and single-cell expression analysis pipeline.
        * **Architecture**: Utilizes the Scanpy package to process highly sparse, high-dimensional single-cell RNA sequencing matrices from peripheral blood mononuclear cells (PBMCs).
        * **Technical Depth**: Deploys graph-based Leiden clustering algorithms combined with log-normalization loops and Wilcoxon rank-sum differential tests to isolate cell variants.
    * **[Breast Cancer Survival Analysis](https://github.com/chirag-7/Breast-Cancer-Survival-Analysis)**: Clinical trial biostatistics and prognostic risk factor evaluation platform.
        * **Architecture**: Maps longitudinal patient oncology records using non-parametric Kaplan-Meier survival curves and Log-Rank hypothesis significance scoring.
        * **Technical Depth**: Fits semi-parametric Cox Proportional Hazards and parametric Weibull Accelerated Failure Time (AFT) models to properly evaluate covariates under right-censored time constraints.
    * **[VAE for De Novo Molecular Generation](https://github.com/chirag-7/VAE-for-De-Novo-Molecular-Generation)**: Deep variational generative model for structural computational chemistry and drug discovery.
        * **Architecture**: Implements sequence autoencoders parsing molecular strings tokenized via SMILES and SELFIES syntax rules.
        * **Technical Depth**: Evaluates continuous latent space generation via a $\beta$-TC-VAE architecture in PyTorch, mapping outputs against Quantitative Estimate of Drug-likeness (QED) and Synthetic Accessibility (SA) constraints.

* **Large-Scale Training**
    * **[RLVR](https://github.com/chirag-7/RLVR)**: Reasoning optimization post-training alignment suite using Reinforcement Learning with Verifiable Rewards.
        * **Architecture**: Implements Group Relative Policy Optimization (GRPO) over the GSM8K mathematical reasoning token dataset to optimize adapter policies.
        * **Technical Depth**: Bypasses unstable neural reward models by building dual rule-based deterministic parsers that mathematically verify code structure, exact string formats, and solution correctness.
    * **[Pi Zero Pytorch](https://github.com/chirag-7/pi-zero-pytorch)**: Embodied Vision-Language-Action (VLA) robotics foundational policy architecture.
        * **Architecture**: Implements a Pi0-style physical intelligence network to execute training loops over interleaved multi-modal inputs and action-conditioned physical trajectories.
        * **Technical Depth**: Pairs a multi-modal pre-fusion Transformer core with a Flow Matching continuous diffusion inference kernel to stream smooth, real-time spatial manipulation joint coordinates.
    * **[Flamingo Pytorch](https://github.com/chirag-7/flamingo-pytorch)**: Multimodal interleaved vision-language foundational sequence generation engine.
        * **Architecture**: Builds a Pythonic implementation of the DeepMind Flamingo framework, introducing Gated Cross-Attention blocks to bridge frozen Vision Encoders and Large Language Models.
        * **Technical Depth**: Implements a structural Perceiver Resampler to reduce variable-sized visual spatial grids into a fixed token footprint, avoiding context window scaling explosions.

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
