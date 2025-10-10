This is the training script for the neural ranker at https://huggingface.co/minhhungg/multi-hop-rag-reranker

---

### Executive Summary: The Two-Stage Curriculum

The core philosophy is to create a ranker that is both a **"Reasoning Architect"** and a **"Factual Grounder."** A model trained only on complex questions might fail to identify simple facts, and a model trained only on simple facts will fail at multi-hop reasoning. This two-stage process solves that dilemma.

1.  **Stage 1 (HotpotQA - The Reasoning Architect):** The model first learns the difficult skill of **multi-hop reasoning**. It's trained exclusively on the `hotpot_qa` dataset, where questions require connecting pieces of information from multiple different documents to form an answer. This teaches the model to identify sentences that act as logical "bridges."

2.  **Stage 2 (Natural Questions - The Factual Grounder):** After mastering complex reasoning, the model undergoes **continual finetuning** on the `nq` dataset. These are primarily direct, fact-based questions. This stage teaches the model to also value and rank highly the single, definitive sentence that contains a direct answer, preventing it from *only* looking for complex logical chains.

The result is a single, robust ranker that can correctly score and prioritize evidence for virtually any type of query, from simple "who is..." questions to complex "trace the link between..." scenarios.

---

### Deep Dive into Each Stage

#### Stage 1: Training the Reasoning Architect on HotpotQA

This is the foundational stage where the model learns the *structure* of a good answer.

**The Dataset (`hotpot_qa`):**
HotpotQA questions are comparative and inferential (e.g., "were Scott Derrickson and ed wood of the same nationality?"). To answer this, the model can't just find one sentence. It must:
1.  Find a sentence stating Scott Derrickson is an American director.
2.  Find a *different* sentence (possibly in a *different document*) stating Ed Wood was an American filmmaker.
3.  Synthesize these two facts to conclude "yes."

**The Methodology (The "Feature-Rich" Input):**
This is the most critical innovation. The script doesn't just feed the model `query [SEP] sentence`. It enriches the input with a prefix of special tokens that act as powerful hints:

*   `[HOP:{0 or 1}]`: **This is for multi-hop reasoning.** It tells the ranker the "distance" of a sentence from the core query. A `[HOP:0]` sentence is a primary fact directly related to the query, while a `[HOP:1]` sentence might be a supporting fact that connects two primary facts. This explicitly teaches the model to identify and value "bridge" sentences.
*   `[ROLE:{Role}]`: **This provides rhetorical context.** The script classifies sentences as `Main_Claim`, `Supporting_Evidence`, etc. A sentence marked as `[ROLE:Supporting_Evidence]` is likely more valuable than one marked `[ROLE:Background_Information]`.
*   `[ENT:{entities}]`: **This signals information density.** A sentence containing multiple named entities is often more informative and thus a better candidate for an answer.
*   `[CAUSAL]` & `[REASONING]`: **These are explicit logical signals.** They flag sentences that describe cause-and-effect relationships or contain multiple logical components, which are crucial for building a reasoning chain.

**The Outcome:**
After training on HotpotQA with these features, the model is no longer just matching keywords. It has learned to think like a detective, asking: "Is this sentence a primary clue (`HOP:0`), a supporting piece of evidence (`ROLE:Supporting_Evidence`), and does it connect multiple suspects (`ENT:A,B,C`)?" The high F1 score of **0.98** shows it mastered this skill.

#### Stage 2: Factual Grounding with Natural Questions (NQ)

This stage makes the expert reasoner a well-rounded generalist.

**The Dataset (`nq`):**
The NQ consists of simple, factual queries like: "who got the first nobel prize in Physics". The answer is contained within a single sentence. This dataset is the polar opposite of HotpotQA.

**The Methodology (Continual Finetuning):**
1.  **Loading the Expert:** The script loads the fully trained HotpotQA model. It doesn't start from scratch.
2.  **Lower Learning Rate:** The learning rate (`2e-6`) is significantly lower than in Stage 1 (`1.5e-5`). This is critical. It tells the model: "You already have a great foundation. Now, gently adapt your knowledge to this new type of question without forgetting what you learned."
3.  **Preventing "Catastrophic Forgetting":** This process fine-tunes the model's weights just enough to make it good at spotting direct answers, without erasing its powerful ability to handle multi-hop reasoning.

**The Outcome:**
The final model is now a "dual-threat." When it sees a query, it can recognize patterns for both complex reasoning (thanks to HotpotQA) and direct factual answers (thanks to NQ). The final F1 score of **0.92** on NQ demonstrates its new, versatile capability.

### Connection to the RAG Architectures

This expertly trained Neural Ranker is the engine that powers the **Keystone Apex** architecture.

*   The **"Feature-Rich Input"** is the exact mechanism used by the Apex Ranker.
*   The **"Dynamic Hop Assignment"** that was a key feature of the Apex pipeline is an *inference-time application* of the `[HOP]` feature the model was trained on here.
*   This ranker is what allows version to implicitly handle multi-hop questions without needing an explicit "Logical Weaver" module like the base Keystone model. The intelligence is baked directly into the ranker.


To demonstrate how effective the neural ranker is besides AUC & F1 Score that you can see in the notebook cells. Here are my 2 experimental custom RAG setups, aiming at heavily logical, multi-disciplinary multi-doc scenarios - one with explicit modules to handle multi-hop questions(version KEYSTONE) and one that can implicitly handle multi-hop questions just as good without an explicit module (version APEX)

<details> 
  
<summary>KEYSTONE Ver</summary>

```
#!/usr/bin/env python3
"""
Version KEYSTONE
"""

import os
import re
import json
import hashlib
import time
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
from cachetools import TTLCache
import shelve
import atexit
import pickle
import argparse
from collections import defaultdict


# --- Initial Setup ---
nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_CACHE = shelve.open("embedding_cache.db", writeback=True)
atexit.register(lambda: EMBEDDING_CACHE.close())

# ==============================================================================
# --- DATA STRUCTURES & CONFIGURATION ---
# ==============================================================================
class JudgeEvaluation(NamedTuple):
    is_faithful: bool; faithfulness_score: float; faithfulness_reasoning: str
    relevance_score: float; relevance_reasoning: str

@dataclass
class PipelineConfig:
    sbert_model_name: str = 'intfloat/multilingual-e5-large-instruct'
    ner_model_name: str = 'dslim/bert-base-NER'
    nli_model_name: str = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
    neural_ranker_path: str = 'F:\\neural_ranker_hotpot_nq_final'
    fpt_api_key: str = os.getenv("FPT_API_KEY", "")
    fpt_base_url: str = os.getenv("FPT_BASE_URL", "https://mkp-api.fptcloud.com")
    fpt_model_name: str = os.getenv("FPT_MODEL_NAME", "Qwen2.5-7B-Instruct")
    fpt_judge_model_name: str = os.getenv("FPT_JUDGE_MODEL_NAME", "DeepSeek-V3")
    retrieval_k: int = 25
    final_evidence_count: int = 7
    mmr_lambda: float = 0.5
    # parameter for the Logical Weaver trigger
    multihop_doc_count_trigger: int = 3
    contradiction_threshold: float = 0.9
    use_llm_as_judge: bool = True
    # AUDITOR: New hyperparameter for the grounding check
    min_bridge_grounding_score: float = 0.65 # Threshold for validating a hypothesized bridge concept

@dataclass
class Document: doc_id: str; text: str

@dataclass
class Query:
    query_id: str; text: str
    doc_ids: List[str]
    ground_truth: Optional[str] = None

@dataclass(eq=False)
class Sentence:
    doc_id: str; sent_idx: int; text: str
    embedding: np.ndarray; hash: str
    relevance_score: float = 0.0

# ==============================================================================
# --- VALIDATION COMPONENTS ---
# ==============================================================================
class EvidenceValidator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nli_pipeline = hf_pipeline("text-classification", model=config.nli_model_name, device=0 if DEVICE=="cuda" else -1)
    def check_for_contradictions(self, evidence: List[Sentence]) -> Tuple[float, List[str]]:
        warnings, max_score = [], 0.0
        if len(evidence) < 2: return 0.0, warnings
        texts = [s.text for s in evidence]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                try:
                    result = self.nli_pipeline(f"{texts[i]} [SEP] {texts[j]}")
                    for res in result:
                        if res['label'] == 'CONTRADICTION': max_score = max(max_score, res['score'])
                except Exception: continue
        if max_score > self.config.contradiction_threshold: warnings.append("CONTRADICTORY_EVIDENCE")
        return max_score, warnings

class LLMAsJudgeValidator:
    def __init__(self, config: PipelineConfig, client: OpenAI):
        self.config, self.client = config, client
    def _robust_score_parse(self, score_input: Any) -> float:
        if isinstance(score_input, (int, float)): return float(score_input)
        if isinstance(score_input, str):
            numbers = re.findall(r'\d+', score_input)
            if numbers: return float(numbers[0])
        return 0.0
    def evaluate_answer(self, query: Query, answer: str, evidence_map: Dict[str, Sentence], reasoning_chain: str = "") -> JudgeEvaluation:
        if not answer or not evidence_map: return JudgeEvaluation(False, 0.0, "No answer/evidence.", 0.0, "N/A")
        clean_answer = re.sub(r'\s*\[\d+\]', '', str(answer))
        evidence_texts = "\n".join([f"EVIDENCE_{i+1}: {s.text}" for i, s in enumerate(evidence_map.values())])
        
        # "Synthetic Judge" prompt
        prompt = f"""You are a meticulous and impartial evaluator. Your task is to perform a forensic analysis of a generated answer.

<INSTRUCTIONS>
1.  **Analyze Claims**: Break down the <GENERATED_ANSWER> into individual claims.
2.  **Map Evidence**: For each claim, find ALL relevant evidence IDs from <EVIDENCE> that support it. A claim may be a logical synthesis of MULTIPLE evidence IDs.
3.  **Validate Reasoning**: If a <PIPELINE_REASONING_CHAIN> is provided, your primary goal is to assess if the answer's logic faithfully follows that chain.
4.  **Score Faithfulness (INTEGER 1-5)**: Rate if every claim is fully supported by the evidence. If a reasoning chain is provided, rate if the answer adheres to it. 5 is perfect adherence. 1 means it deviates or is unsupported.
5.  **Score Relevance (INTEGER 1-5)**: Rate how well the answer addresses the <QUESTION>. 5 is a perfect answer. 1 is off-topic.
6.  **Final JSON Output**: Provide your analysis in a single, valid JSON object with INTEGER scores.
</INSTRUCTIONS>

<EVIDENCE>
{evidence_texts}
</EVIDENCE>

<QUESTION>
{query.text}
</QUESTION>

<GENERATED_ANSWER>
{clean_answer}
</GENERATED_ANSWER>"""

        if reasoning_chain:
            prompt += f"""
<PIPELINE_REASONING_CHAIN>
{reasoning_chain}
</PIPELINE_REASONING_CHAIN>"""

        prompt += f"""
<OUTPUT_FORMAT>
{{"chain_of_thought": [], "faithfulness_score": 5, "faithfulness_reasoning": "...", "relevance_score": 5, "relevance_reasoning": "..."}}
</OUTPUT_FORMAT>

Begin your forensic analysis now:"""
        
        response = RobustErrorHandler.safe_llm_call(self.client, "LLM-as-a-Judge", "{}", model=self.config.fpt_judge_model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
        
        try:
            data = json.loads(response)
            fs_raw = self._robust_score_parse(data.get("faithfulness_score", 0))
            rs_raw = self._robust_score_parse(data.get("relevance_score", 0))
            fs, rs = fs_raw / 5.0, rs_raw / 5.0
            return JudgeEvaluation(fs>0.8, fs, data.get("faithfulness_reasoning","N/A"), rs, data.get("relevance_reasoning","N/A"))
        except Exception as e:
            logger.error(f"Judge failed to parse response: {e}"); return JudgeEvaluation(False, 0.0, "Parse error.", 0.0, "N/A")

# ==============================================================================
# --- CORE COMPONENTS ---
# ==============================================================================
class RobustErrorHandler:
    @staticmethod
    def safe_execute(op, func, fallback, *a, **kw):
        try: return func(*a, **kw)
        except Exception as e: logger.warning(f"{op} failed: {e}... Using fallback."); return fallback
    @staticmethod
    def safe_llm_call(client, op, fallback, **params):
        try: return client.chat.completions.create(**params).choices[0].message.content.strip()
        except Exception as e: logger.error(f"LLM call {op} failed: {e}"); return fallback

class DataManager:
    def load_documents(self, fp: str) -> Dict[str, Document]:
        docs = {};
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data = json.loads(line); docs[data['doc_id']] = Document(**data)
        return docs
    def load_queries(self, fp: str) -> Dict[str, Query]:
        queries = {};
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data = json.loads(line); queries[data['query_id']] = Query(**data)
        return queries

class EnhancedPreprocessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sbert_model = SentenceTransformer(config.sbert_model_name, device=DEVICE)
        # V27.0 CRITICAL FIX: Add the missing model initializations
        self.ner_pipeline = hf_pipeline("ner", model=config.ner_model_name, grouped_entities=True, device=0 if DEVICE=="cuda" else -1)
    def get_embedding(self, text: str, prefix: str) -> np.ndarray:
        """
        This function is now guaranteed to always return a valid numpy array.
        It handles potential silent failures from the encoder and bad inputs.
        """
        # --- Pre-computation Check ---
        if not text or not isinstance(text, str):
            logger.warning(f"get_embedding received invalid input: type={type(text)}, value='{text}'. Returning a zero vector.")
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        if not text.strip():
            logger.warning("get_embedding received an empty or whitespace-only string. Returning a zero vector.")
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)

        # --- Cache Check ---
        key = f"{prefix}:{text}"
        text_hash = hashlib.sha256(key.encode()).hexdigest()
        if (cached_emb := EMBEDDING_CACHE.get(text_hash)) is not None:
            if isinstance(cached_emb, np.ndarray):
                return cached_emb
        
        # --- Encoding ---
        try:
            emb = self.sbert_model.encode(f"{prefix}: {text}", convert_to_numpy=True, show_progress_bar=False)
            
            if emb is None:
                raise ValueError("SBERT model returned None for a valid string.")

            EMBEDDING_CACHE[text_hash] = emb
            EMBEDDING_CACHE.sync()
            return emb
            
        except Exception as e:
            logger.error(f"A critical error occurred in get_embedding for text '{text[:100]}...': {e}")
            # In case of ANY failure, return a zero vector. This PREVENTS the pipeline from crashing.
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extracts key technical terms and named entities."""
        #The NER pipeline returns a dict, so we must access the 'word' key.
        concepts = set(ent['word'] for ent in self.ner_pipeline(text) if isinstance(ent, dict) and 'word' in ent and len(ent['word'].split()) > 1)
        return list(concepts)
        EMBEDDING_CACHE[text_hash] = emb; EMBEDDING_CACHE.sync(); return emb
    
    def process_documents_robust(self, documents: Dict[str, Document]) -> Tuple[List[Sentence], Dict[str, List[str]]]:
        texts, sentence_map = [], []
        # map to pre-compute
        doc_to_concepts_map = defaultdict(list)
        
        for doc_id, doc in documents.items():
            # Extract concepts at the document level for the map
            doc_concepts = self._extract_key_concepts(doc.text)
            doc_to_concepts_map[doc_id] = doc_concepts
            
            sents = sent_tokenize(doc.text)
            for sent_idx, text in enumerate(sents):
                if 4 < len(text.split()) < 250: 
                    texts.append(text)
                    sentence_map.append({'doc_id': doc_id, 'sent_idx': sent_idx})

        embeddings = np.array([self.get_embedding(t, "passage") for t in tqdm(texts, "Embedding", leave=False)])
        all_sentences = [Sentence(doc_id=info['doc_id'], sent_idx=info['sent_idx'], text=texts[i], embedding=embeddings[i], hash=hashlib.sha256(texts[i].encode()).hexdigest()) for i, info in enumerate(sentence_map)]
        
        # Return both the sentence pool and the new concept map
        return all_sentences, dict(doc_to_concepts_map)

class NeuralRanker:
    def __init__(self, model_path: str, device: str = DEVICE):
        self.device, self.tokenizer, self.model = device, AutoTokenizer.from_pretrained(model_path), AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    @torch.no_grad()
    def rank_with_scores(self, query: str, sentences: List[Sentence]) -> List[Sentence]:
        if not sentences: return []
        inputs = [f"{query} [SEP] {s.text}" for s in sentences]; scores = []
        for i in range(0, len(inputs), 32):
            batch = self.tokenizer(inputs[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            scores.extend(torch.sigmoid(self.model(**batch).logits).squeeze(-1).cpu().numpy().tolist())
        for s, score in zip(sentences, scores): s.relevance_score = float(score)
        return sorted(sentences, key=lambda s: s.relevance_score, reverse=True)

# ==============================================================================
# --- V25.3: KEYSTONE RAG CONTROLLER ---
# ==============================================================================
import hdbscan
from sklearn.cluster import SpectralClustering
from collections import Counter
class KeystoneRAGController:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_manager = DataManager()
        self.preprocessor = EnhancedPreprocessor(config)
        self.neural_ranker = NeuralRanker(config.neural_ranker_path)
        self.fpt_client = OpenAI(api_key=config.fpt_api_key, base_url=config.fpt_base_url)
        self.evidence_validator = EvidenceValidator(config)
        self.judge = LLMAsJudgeValidator(config, self.fpt_client)
        self.ner_pipeline = hf_pipeline("ner", model=config.ner_model_name, grouped_entities=True, device=0 if DEVICE=="cuda" else -1)
        self.documents, self.queries, self.sentence_pool, self.sentence_index = {}, {}, [], None
        # attribute to hold the concept map
        self.doc_to_concepts_map = {} 

    def setup(self, doc_file: str, query_file: str, force_reingest: bool = False):
        try:
            with open(doc_file, 'rb') as f1, open(query_file, 'rb') as f2:
                state_hash = hashlib.md5(f1.read() + f2.read()).hexdigest()
        except FileNotFoundError:
            logger.critical("Document or query file not found. Cannot proceed."); return
        cache_dir = "cache"; os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cache_{state_hash}_v25.3_keystone.pkl")
        if force_reingest and os.path.exists(cache_path):
            logger.warning("Forcing re-ingestion. Deleting existing cache..."); os.remove(cache_path)
        if os.path.exists(cache_path):
            logger.info(f"Loading V27.0 Sentinel corpus from cache: {cache_path}")
            with open(cache_path, "rb") as f: data = pickle.load(f)
            # V27.0: Unpack the new map from the cache
            self.documents, self.queries, self.sentence_pool, self.sentence_index, self.doc_to_concepts_map = \
                data['docs'], data['queries'], data['pool'], data['faiss'], data['concepts']
        else:
            logger.info("No valid cache found. Starting full pre-computation...")
            self.documents = self.data_manager.load_documents(doc_file)
            self.queries = self.data_manager.load_queries(query_file)
            # Capture the new map during processing
            self.sentence_pool, self.doc_to_concepts_map = self.preprocessor.process_documents_robust(self.documents)
            embs = np.array([s.embedding for s in self.sentence_pool]).astype('float32')
            faiss.normalize_L2(embs); self.sentence_index = faiss.IndexFlatIP(embs.shape[1]); self.sentence_index.add(embs)
            logger.info(f"Caching new state to: {cache_path}")
            with open(cache_path, "wb") as f:
                # Save the new map to the cache
                pickle.dump({'docs': self.documents, 'queries': self.queries, 'pool': self.sentence_pool, 'faiss': self.sentence_index, 'concepts': self.doc_to_concepts_map}, f)
        logger.info("V27.0 Sentinel RAG Controller setup complete.")
    
    
    
    def _get_scoped_indices(self, doc_ids: List[str]) -> np.ndarray:
        if "all" in doc_ids: return np.arange(len(self.sentence_pool), dtype=np.int64)
        return np.array([i for i, s in enumerate(self.sentence_pool) if s.doc_id in doc_ids], dtype=np.int64)
    
    def _run_bridge_retrieval(self, query: Query, evidence: List[Sentence]) -> List[Sentence]:
        """
        Upgraded Auditor with pre-computed concept hints.
        """
        logger.info("Auditor module activated: Analyzing evidence for conceptual gaps...")
        
        # 1. Gather conceptual hints from the source documents of the evidence
        source_doc_ids = list(set(s.doc_id for s in evidence))
        concept_hints = []
        for doc_id in source_doc_ids:
            concept_hints.extend(self.doc_to_concepts_map.get(doc_id, []))
        
        evidence_text = "\n".join([f"- {s.text}" for s in evidence])
        
        # 2. Upgraded prompt with Chain-of-Thought and JSON output for better gap-bridging
        prompt = f"""You are a master detective and reasoning analyst. Your task is to identify the missing 'bridge concept' that connects disparate pieces of evidence to answer a central query.

<INSTRUCTIONS>
1.  **Analyze the Gap**: First, in a `chain_of_thought`, briefly state what each piece of evidence establishes and what logical gap exists between them in relation to the main query.
2.  **Hypothesize the Bridge**: Based on your analysis and the provided `CONCEPT_HINTS`, hypothesize the single most likely SPECIFIC named entity, event, or technical term that bridges this gap.
3.  **Justify Your Hypothesis**: In the `justification` field, explain *why* this concept is the bridge.
4.  **Final JSON Output**: Respond with a single, valid JSON object. If no single concept can bridge the evidence, the `hypothesized_bridge_concept` should be "None".
</INSTRUCTIONS>

<MAIN_QUERY>
{query.text}
</MAIN_QUERY>

<DISPARATE_EVIDENCE>
{evidence_text[:2500]}
</DISPARATE_EVIDENCE>

<CONCEPT_HINTS_FROM_SOURCE_DOCS>
{list(set(concept_hints))}
</CONCEPT_HINTS_FROM_SOURCE_DOCS>

<OUTPUT_FORMAT>
{{
  "chain_of_thought": "Evidence 1 is about X. Evidence 2 is about Y. The missing link is what connects X and Y.",
  "hypothesized_bridge_concept": "The specific concept",
  "justification": "This concept is the bridge because..."
}}
</OUTPUT_FORMAT>

Your analysis:"""

        response_str = RobustErrorHandler.safe_llm_call(self.fpt_client, "Bridge Concept Identification", "{{}}",
            model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)

        try:
            response_json = json.loads(response_str)
            bridge_concept = response_json.get("hypothesized_bridge_concept", "NONE")
        except json.JSONDecodeError:
            logger.warning("Auditor failed to parse JSON response for bridge concept.")
            bridge_concept = "NONE"

        if "NONE" in bridge_concept or len(bridge_concept) > 50:
            logger.warning("Auditor could not identify a clear bridge concept.")
            return []

        logger.info(f"Auditor identified potential bridge concept: '{bridge_concept}'")

        # 3. Targeted Retrieval
        bridge_evidence = self._run_retrieval(query_texts=[bridge_concept], doc_ids=["all"])
        return bridge_evidence[:3]
    def answer_query(self, query: Query) -> Dict[str, Any]:
        start_time = time.time()
        
        subjects = self._extract_query_subjects(query)
        retrieval_queries = [query.text] + subjects
        
        initial_evidence = self._run_retrieval(retrieval_queries, query.doc_ids)
        expanded_evidence = self._run_contextual_expansion(query, initial_evidence, subjects)

        # MMR now operates directly on the rich, expanded evidence set.
        final_evidence = self._select_final_evidence_with_mmr(query, expanded_evidence)

        reasoning_chain = ""
        unique_doc_ids_in_evidence = set(s.doc_id for s in final_evidence)
        if len(unique_doc_ids_in_evidence) >= self.config.multihop_doc_count_trigger:
            # The Auditor (Bridge Retrieval) is called here
            bridge_sentences = self._run_bridge_retrieval(query, final_evidence)
            if bridge_sentences:
                logger.info(f"Apex Auditor added {len(bridge_sentences)} new sentences to context.")
                final_evidence_map = {s.hash: s for s in final_evidence}
                for s in bridge_sentences:
                    final_evidence_map[s.hash] = s
                # Rerank the final, complete set of evidence one last time
                final_evidence = self.neural_ranker.rank_with_scores(query.text, list(final_evidence_map.values()))[:self.config.final_evidence_count + 2]
        
        answer, evidence_map, reasoning_chain = self._generate_answer_and_citations(query, final_evidence)
        
        contradiction_score, warnings = self.evidence_validator.check_for_contradictions(final_evidence)
        score, eval_details = 0.0, "Evaluation Disabled"
        if self.config.use_llm_as_judge:
            judge_eval = self.judge.evaluate_answer(query, answer, evidence_map, reasoning_chain)
            if not judge_eval.is_faithful: warnings.append("JUDGE_FOUND_UNFAITHFUL")
            score, eval_details = (judge_eval.faithfulness_score*0.7)+(judge_eval.relevance_score*0.3), judge_eval._asdict()
        else:
            score = (1.0 - (len(warnings) * 0.2)) * (1.0 - contradiction_score)
            eval_details = "LLM-as-a-Judge is disabled. Using proxy score."

        return {'query_id': query.query_id, 'answer': str(answer), 'tier_used': 'Unified Apex Path', 'processing_time': time.time() - start_time,
                'warnings': list(set(warnings)), 'evidence_contradiction_score': contradiction_score,
                'llm_judge_evaluation': eval_details, 'final_confidence_score': score}

    def _extract_query_subjects(self, query: Query) -> List[str]:
        """
        Sanitizes NER output and explicitly filters empty strings.
        """
        subjects = []
        try:
            ner_entities = self.ner_pipeline(query.text)
            if ner_entities and isinstance(ner_entities, list):
                sanitized_subjects = set()
                for e in ner_entities:
                    if isinstance(e, dict) and 'word' in e:
                        raw_word = e['word']
                        logger.debug(f"Raw NER entity: '{raw_word}'")
                        
                        # Handle apostrophes by replacing with space (not removing)
                        cleaned = raw_word.replace("'", " ").replace("’", " ")
                        
                        # Remove other punctuation but keep spaces and hyphens
                        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
                        
                        # Normalize spaces
                        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                        
                        logger.debug(f"Cleaned entity: '{cleaned}'")
                        
                        # Only add if non-empty and meaningful
                        if cleaned and len(cleaned) > 1:
                            # Split into words and filter out single-character words
                            words = [word for word in cleaned.split() if len(word) > 1]
                            if words:
                                # Rejoin to maintain multi-word entities
                                final_entity = ' '.join(words)
                                sanitized_subjects.add(final_entity)
                                logger.debug(f"Added subject: '{final_entity}'")
                
                subjects = list(sanitized_subjects)
                logger.info(f"Final extracted subjects: {subjects}")
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            pass  # Fallback to LLM if NER fails

        if not subjects:
            logger.warning("NER failed or found no subjects, falling back to LLM extraction.")
            prompt = f"""Extract the 1-3 primary named entities or technical subjects from the user query. Output ONLY a valid JSON list of strings.\nQuery: "{query.text}"\nExample Output: ["Quantum Nexus Initiative", "Arbor BCI"]\nSubjects:"""
            response = RobustErrorHandler.safe_llm_call(self.fpt_client, "Subject Extraction", "[]", model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
            try:
                subjects = json.loads(response)
                # Also filter the LLM's output just in case
                subjects = [s for s in subjects if isinstance(s, str) and s.strip() and len(s.strip()) > 1]
            except json.JSONDecodeError:
                subjects = []
        
        return subjects

    def _run_retrieval(self, query_texts: List[str], doc_ids: List[str]) -> List[Sentence]:
        scoped_indices = self._get_scoped_indices(doc_ids)
        if len(scoped_indices) == 0: return []
        q_embs = np.array([self.preprocessor.get_embedding(q, "query") for q in query_texts])
        k_val = min(self.config.retrieval_k, len(scoped_indices))
        selector = faiss.IDSelectorArray(scoped_indices)
        candidate_map = {}
        _, ids = self.sentence_index.search(q_embs, k=k_val, params=faiss.SearchParameters(sel=selector))
        for i in ids.flatten():
            if i != -1: candidate_map[self.sentence_pool[i].hash] = self.sentence_pool[i]
        return self.neural_ranker.rank_with_scores(query_texts[0], list(candidate_map.values()))

    def _run_contextual_expansion(self, query: Query, evidence: List[Sentence], subjects: List[str]) -> List[Sentence]:
        if not subjects: return evidence
        evidence_text = " ".join([s.text for s in evidence])
        missing_subjects = [s for s in subjects if s.lower() not in evidence_text.lower()]
        if not missing_subjects: return evidence
        logger.info(f"Contextual Expansion: Searching for missing subject(s): {missing_subjects}")
        expansion_evidence = {}
        for subject in missing_subjects:
            if subject and isinstance(subject, str) and subject.strip() and len(subject.strip()) > 1:
                expansion_candidates = self._run_retrieval([f"What is {subject}?"], query.doc_ids)
                if expansion_candidates: expansion_evidence[expansion_candidates[0].hash] = expansion_candidates[0]
            else:
                logger.warning(f"Skipping invalid subject: '{subject}'")
        final_evidence_map = {s.hash: s for s in evidence}; final_evidence_map.update(expansion_evidence)
        return self.neural_ranker.rank_with_scores(query.text, list(final_evidence_map.values()))
    
    def _select_final_evidence_with_mmr(self, query: Query, candidates: List[Sentence]) -> List[Sentence]:
        if not candidates: return []
        target_count = min(self.config.final_evidence_count, len(candidates))
        if len(candidates) <= target_count: return candidates
        candidate_embeddings = np.array([s.embedding for s in candidates])
        query_embedding = self.preprocessor.get_embedding(query.text, "query")
        selected_indices = [0]
        while len(selected_indices) < target_count:
            best_next_idx, max_mmr_score = -1, -np.inf
            selected_embeddings = candidate_embeddings[selected_indices]
            for i in range(len(candidates)):
                if i in selected_indices: continue
                relevance = util.cos_sim(query_embedding, candidate_embeddings[i])[0][0].item()
                redundancy = np.max(util.cos_sim(candidate_embeddings[i], selected_embeddings)[0].cpu().numpy())
                mmr_score = (1 - self.config.mmr_lambda) * relevance - self.config.mmr_lambda * redundancy
                if mmr_score > max_mmr_score: max_mmr_score, best_next_idx = mmr_score, i
            if best_next_idx == -1: break
            selected_indices.append(best_next_idx)
        return [candidates[i] for i in selected_indices]
    
    # Logical Weaver module
    def _run_logical_weaver(self, query: Query, facts: str) -> str:
        """
        uses a non-leaky example to teach the reasoning pattern.
        """
        logger.info("Activating Logical Weaver for multi-hop synthesis guidance...")
        
        prompt = f"""You are a reasoning architect. Your task is to create a step-by-step reasoning chain that logically connects the provided Disparate Facts to answer the Main Query.
Identify the 'bridge concepts' or 'common threads' that link the facts. For every logical step you propose, you MUST cite the specific fact numbers that support that step.
Your output must be a clear, logical plan that the final writer will follow.

<MAIN_QUERY>
{query.text}
</MAIN_QUERY>

{query.text}
</MAIN_QUERY>

<DISPARATE_FACTS>
{facts}
</DISPARATE_FACTS>

<EXAMPLE_OF_A_GOOD_REASONING_CHAIN>
- **Query:** "How could the primary operational risk of the 'Project Star-View' satellite constellation be mechanistically linked to the main safety concern of the 'CRISPR-Z' gene-editing platform?"
- **Reasoning Chain:**
1. The main safety concern for the 'CRISPR-Z' platform is its high rate of off-target mutations. [Fact 3]
2. The primary operational risk for the 'Project Star-View' constellation is sensor damage from a specific type of solar flare. [Fact 7]
3. A newly discovered form of radiation, "Hyper-Ionized Gamma Bursts (HIGs)," has been identified as the primary component of these solar flares. [Fact 5]
4. Crucially, these same HIGs have also been documented to be a direct cause of the off-target mutations seen in gene-editing systems. [Fact 5, Fact 6]
5. Therefore, HIGs are the direct mechanistic link. They are the physical phenomenon responsible for both the satellite sensor damage and the gene-editing errors, connecting the two disparate domains through a shared vulnerability to a specific type of cosmic radiation. [Connects Fact 3, 7, and 5]
</EXAMPLE_OF_A_GOOD_REASONING_CHAIN>

Reasoning Chain:"""

        return RobustErrorHandler.safe_llm_call(self.fpt_client, "Logical Weaver", "", model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)

    def _generate_answer_and_citations(self, query: Query, evidence: List[Sentence]) -> Tuple[str, Dict[str, Sentence], str]:
        """
        All return paths now correctly provide three values to prevent ValueErrors.
        """
        reasoning_chain = ""

        if not evidence:
            return "Insufficient evidence to construct an answer.", {}, reasoning_chain

        evidence_map = {f"EVIDENCE_{i+1}": s for i, s in enumerate(evidence)}
        fact_sheet = "\n".join([f"[{k}] {s.text}" for k, s in evidence_map.items()])
        
        extract_prompt = f"""Based ONLY on the provided Fact Sheet, extract a list of key facts as bullet points that directly help answer the User Query.\n<USER_QUERY>{query.text}</USER_QUERY>\n<FACT_SHEET>{fact_sheet}</FACT_SHEET>\nExtracted Facts:"""
        facts = RobustErrorHandler.safe_llm_call(self.fpt_client, "Fact Extraction", "", model=self.config.fpt_model_name, messages=[{"role": "user", "content": extract_prompt}], temperature=0.1)
        
        if not facts:
            return "Could not extract relevant facts from the evidence.", evidence_map, reasoning_chain

        # Activate Logical Weaver for multi-hop queries
        unique_doc_ids = set(s.doc_id for s in evidence)
        if len(unique_doc_ids) >= self.config.multihop_doc_count_trigger:
            reasoning_chain = self._run_logical_weaver(query, facts)

        synthesis_prompt = f"""Synthesize the following Key Facts into a cohesive, well-written paragraph that directly answers the User Query. Do not add any new information.
<USER_QUERY>{query.text}</USER_QUERY>
<KEY_FACTS>{facts}</KEY_FACTS>"""
        
        if reasoning_chain:
            synthesis_prompt += f"""
**CRITICAL INSTRUCTION: You MUST follow the provided REASONING CHAIN to structure your answer. Do not deviate from this logical path.**
<REASONING_CHAIN>{reasoning_chain}</REASONING_CHAIN>"""
        
        synthesis_prompt += "\nFinal Answer:"
        
        answer = RobustErrorHandler.safe_llm_call(self.fpt_client, "Answer Synthesis", "Could not synthesize an answer.", model=self.config.fpt_model_name, messages=[{"role": "user", "content": synthesis_prompt}], temperature=0.0)
        
        final_answer = self._render_citations(answer, evidence)
        # The successful path already correctly returns three values
        return final_answer, evidence_map, reasoning_chain
        
    def _render_citations(self, answer: str, evidence: List[Sentence]) -> str:
        cited_answer = str(answer); source_map = {s.text: i+1 for i, s in enumerate(evidence)}
        answer_sents, final_sents = sent_tokenize(cited_answer), []
        if not evidence: return cited_answer
        evidence_embs = np.array([s.embedding for s in evidence])
        for ans_sent in answer_sents:
            if not ans_sent: continue
            ans_sent_emb = self.preprocessor.get_embedding(ans_sent, "query")
            sims = util.cos_sim(ans_sent_emb, evidence_embs)[0].cpu().numpy()
            best_idx = np.argmax(sims)
            if sims[best_idx] > 0.7:
                best_evidence_text = evidence[best_idx].text
                if (citation_num := source_map.get(best_evidence_text)):
                    final_sents.append(f"{ans_sent.strip()} [{citation_num}]")
                else: final_sents.append(ans_sent.strip())
            else: final_sents.append(ans_sent.strip())
        cited_answer = " ".join(final_sents)
        citation_list = "\n\n--- Citations ---\n" + "".join([f"[{i}] {text}\n" for text, i in source_map.items()])
        return cited_answer + citation_list

    def run_interactive_session(self):
        print("\n" + "="*80 + "\n🚀 V25.3 - Keystone Architecture (Final, Polished Build)\n" + "="*80)
        user_input = input("Enable LLM-as-a-Judge evaluation? (yes/no): ").strip().lower()
        self.config.use_llm_as_judge = user_input in ['yes', 'y', '1']
        logger.info(f"LLM-as-a-Judge ENABLED: {self.config.use_llm_as_judge}")
        while True:
            mode = input("\n[1] Pre-loaded Queries, [2] Interactive, [quit]: ").strip()
            if mode == 'quit': break
            if mode == '1' and self.queries: self._run_preloaded()
            elif mode == '2': self._run_interactive()
            else: print("Invalid choice.")
    def _run_preloaded(self):
        for q_id, query in self.queries.items():
            print(f"\n{'='*60}\nProcessing Query: {q_id} - {query.text}\n{'='*60}")
            self._display_response(self.answer_query(query))
    def _run_interactive(self):
        while True:
            q_text = input("\nEnter query (or 'back'): ").strip()
            if q_text.lower() == 'back': break
            doc_ids = [s.strip() for s in input("Enter doc IDs (comma-separated) or 'all': ").split(',')]
            self._display_response(self.answer_query(Query("interactive", q_text, doc_ids)))
    def _display_response(self, response: Dict[str, Any]):
        print(f"\n✅ ANSWER (using {response['tier_used']}):\n{response['answer']}")
        print(f"\n📊 VALIDATION & PERFORMANCE:")
        print(f"  - Final Confidence Score: {response.get('final_confidence_score', 0.0):.3f}")
        print(f"  - Processing Time: {response['processing_time']:.2f}s")
        if 'evidence_contradiction_score' in response: print(f"  - Evidence Contradiction Score: {response['evidence_contradiction_score']:.3f}")
        if response.get('warnings'): print(f"  - ⚠️  Warnings: {', '.join(sorted(list(set(response['warnings']))))}")
        if self.config.use_llm_as_judge and isinstance(response.get('llm_judge_evaluation'), dict):
            judge = response['llm_judge_evaluation']
            print("\n🔬 LLM-AS-A-JUDGE EVALUATION:")
            print(f"  - Faithfulness: {judge['faithfulness_score']:.2f}/1.00 | Reasoning: {judge['faithfulness_reasoning']}")
            print(f"  - Relevance:    {judge['relevance_score']:.2f}/1.00 | Reasoning: {judge['relevance_reasoning']}")

# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the V25.3 Keystone RAG Pipeline.")
    parser.add_argument("--reingest", action="store_true", help="Force deletion of existing cache and re-ingest all data.")
    args = parser.parse_args()
    
    DOCS_FILE, QUERIES_FILE = "docs.jsonl", "queries.jsonl"
    if not os.path.exists(DOCS_FILE) or not os.path.exists(QUERIES_FILE):
        docs_content = """
{"doc_id": "TECH-MEM-MNEMOSYNE", "text": "Project Mnemosyne, a DARPA initiative headquartered at MIT's Media Lab, is developing a next-generation Brain-Computer Interface (BCI) focused on direct memory encoding and retrieval. The system uses a novel 'neuro-photonic' implant that translates digital data into precisely targeted light patterns to stimulate and modify hippocampal engrams. While early results have shown an unprecedented 98% recall accuracy for encoded information, the primary operational risk is 'synaptic interference.' This phenomenon occurs when the implant's photonic emissions inadvertently disrupt adjacent, unrelated memory traces, leading to a form of structured amnesia or memory corruption. The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target. To counter this, the team developed a sophisticated AI called the 'Predictive Hebbian Modulator.' This is a recurrent neural network with a temporal-convolutional attention mechanism that learns the unique synaptic potentiation patterns of an individual's brain. It then pre-emptively adjusts the implant's light frequency and intensity to create a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects. The entire process requires immense computational power, a dependency which has made securing the GPU supply chain a top-level project concern."}
{"doc_id": "QPU-SIM-EREBUS", "text": "At Lawrence Livermore National Laboratory, Project Erebus is a major Department of Energy program aimed at using quantum computers to simulate the behavior of dark matter. The project's quantum processing unit (QPU), a 4,096-qubit topological device, is designed to solve complex quantum chromodynamics equations that are intractable for classical supercomputers. The most significant technical hurdle is a persistent issue termed 'Entanglement Fraying.' This is a specific form of decoherence where the fragile quantum entanglement between distant qubits decays exponentially faster than predicted by standard models, leading to a collapse of the simulation's integrity after only a few hundred microseconds. Analysis has revealed that this accelerated decay is strongly correlated with the same anomalous 'exotic particle flux' documented by other advanced research projects. The Erebus team's solution is an AI error-correction model that runs on a classical co-processor. The AI analyzes the parity-check measurements from the QPU in real-time and predicts the onset of a fraying event. It then instructs the QPU's control system to perform a series of 'entanglement distillation' protocols, sacrificing some qubits to reinforce the stability of the remaining computational set. While this extends the simulation time, it significantly increases the overall number of qubits required, raising concerns about the long-term scalability of the approach."}
{"doc_id": "SPACE-SAIL-HELIOS", "text": "NASA's Project Helios is an ambitious plan to send an unmanned probe to Alpha Centauri using a light sail propelled by a high-powered laser array stationed in Earth orbit. The sail itself is a kilometer-scale, atomically thin sheet of a graphene-molybdenum disulfide heterostructure. The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium. Extensive testing at JPL revealed that the degradation is not caused by conventional protons or alpha particles, but is almost entirely attributable to the same high-energy, 'exotic particle flux' that has been observed affecting quantum and neurological experiments. The proposed mitigation involves a 'self-healing' matrix interwoven into the sail's lattice. A predictive AI model monitors the sail for signs of micro-fracturing. When a potential failure point is detected, the AI activates a localized energy field that triggers a chemical reaction in an embedded substrate, repairing the lattice structure. Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation."}
{"doc_id": "GOV-STRAT-ORION", "text": "A declassified strategic document, known as the 'Orion Mandate,' outlines the United States' primary technological goals for the next decade. The mandate establishes a national priority to achieve 'Cognitive-Computational Supremacy,' defined as the synergistic mastery of next-generation computing, artificial intelligence, and direct neural interface technologies. The document explicitly names three critical, interdependent pillars for achieving this goal: Project Erebus (for computational dominance), Project Mnemosyne (for neural interface superiority), and Project Helios (for demonstrating advanced materials and energy projection). The mandate's primary concern is a new strategic threat termed 'asymmetric temporal sabotage.' This is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation. The document concludes by authorizing a new, top-secret research initiative under the NSA, tasked with identifying and shielding against the 'anomalous high-energy particle phenomena' that have been reported to interfere with all three pillar projects, flagging it as the most likely vector for such a sabotage campaign."}"""
        with open(DOCS_FILE, "w", encoding='utf-8') as f: f.write(docs_content.strip())
        queries_content = """
{"query_id": "Q1-STRESS-DEEP-FACTUAL", "text": "Describe the specific mechanism of 'synaptic interference' in Project Mnemosyne, and detail the architecture and function of the 'Predictive Hebbian Modulator' AI used to counteract it.", "doc_ids": ["TECH-MEM-MNEMOSYNE"]}
{"query_id": "Q2-STRESS-ABSTRACT-SYNTHESIS", "text": "Synthesize the core challenges of Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion). What single, unifying principle of 'long-term material and informational integrity' do both projects struggle with, and how do their respective AI-driven solutions represent a similar philosophical approach to predictive maintenance?", "doc_ids": ["QPU-SIM-EREBUS", "SPACE-SAIL-HELIOS"]}
{"query_id": "Q3-STRESS-GRAND-MULTIHOP", "text": "According to the Orion Mandate, what is 'asymmetric temporal sabotage,' and what is the specific, underlying physical phenomenon (not mentioned in the Mandate itself) that mechanistically links the operational risks of all three critical projects: Mnemosyne, Erebus, and Helios?", "doc_ids": ["all"]}
{"query_id": "Q4-STRESS-CAUSAL-CHAIN", "text": "Trace the causal chain from the economic motivation for 'off-target enzymatic activity' risk mitigation in the original Gen-Synth document to the specific hardware vulnerability exploited in the 'Aethelred Breach' from the Cyber-Sec document. How does this chain illustrate a broader principle of 'economically-driven security trade-offs'?", "doc_ids": ["GEN-SYNTH-2024-ENZYME", "CYBER-SEC-2024-SLCI"]}
"""
        with open(QUERIES_FILE, "w", encoding='utf-8') as f: f.write(queries_content.strip())
    
    try:
        config = PipelineConfig()
        pipeline = KeystoneRAGController(config)
        pipeline.setup(doc_file=DOCS_FILE, query_file=QUERIES_FILE, force_reingest=args.reingest)
        pipeline.run_interactive_session()
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()
```

</details>

<details>
<summary>APEX Ver</summary>

```
#!/usr/bin/env python3
"""
Version APEX
"""

import os
import re
import json
import hashlib
import time
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
from cachetools import TTLCache
import shelve
import atexit
import pickle
import argparse
from collections import defaultdict


# --- Initial Setup ---
nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_CACHE = shelve.open("embedding_cache.db", writeback=True)
atexit.register(lambda: EMBEDDING_CACHE.close())

# ==============================================================================
# --- DATA STRUCTURES & CONFIGURATION ---
# ==============================================================================
class JudgeEvaluation(NamedTuple):
    is_faithful: bool; faithfulness_score: float; faithfulness_reasoning: str
    relevance_score: float; relevance_reasoning: str

@dataclass
class PipelineConfig:
    sbert_model_name: str = 'intfloat/multilingual-e5-large-instruct'
    ner_model_name: str = 'dslim/bert-base-NER'
    nli_model_name: str = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
    neural_ranker_path: str = 'F:\\neural_ranker_hotpot_nq_final'
    fpt_api_key: str = os.getenv("FPT_API_KEY", "")
    fpt_base_url: str = os.getenv("FPT_BASE_URL", "https://mkp-api.fptcloud.com")
    fpt_model_name: str = os.getenv("FPT_MODEL_NAME", "Qwen2.5-7B-Instruct")
    fpt_judge_model_name: str = os.getenv("FPT_JUDGE_MODEL_NAME", "DeepSeek-V3")
    retrieval_k: int = 25
    final_evidence_count: int = 7
    mmr_lambda: float = 0.5
    #  parameter for the Logical Weaver trigger
    multihop_doc_count_trigger: int = 3
    contradiction_threshold: float = 0.9
    use_llm_as_judge: bool = True
    #  AUDITOR: hyperparameter for the grounding check
    min_bridge_grounding_score: float = 0.65 # Threshold for validating a hypothesized bridge concept
    max_length: int = 512  # Maximum sequence length for the tokenizer

@dataclass
class Document: doc_id: str; text: str

@dataclass
class Query:
    query_id: str; text: str
    doc_ids: List[str]
    ground_truth: Optional[str] = None

@dataclass(eq=False)
class Sentence:
    doc_id: str; sent_idx: int; text: str
    embedding: np.ndarray; hash: str
    relevance_score: float = 0.0
    #  Add fields for rich metadata
    entities: List[str] = field(default_factory=list)
    rhetorical_role: str = "Background_Information"
    is_causal: bool = False

# ==============================================================================
# --- VALIDATION COMPONENTS ---
# ==============================================================================
class EvidenceValidator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nli_pipeline = hf_pipeline("text-classification", model=config.nli_model_name, device=0 if DEVICE=="cuda" else -1)
        
    def check_for_contradictions(self, evidence: List[Sentence]) -> Tuple[float, List[str]]:
        warnings, max_score = [], 0.0
        if len(evidence) < 2: return 0.0, warnings
        texts = [s.text for s in evidence]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                try:
                    result = self.nli_pipeline(f"{texts[i]} [SEP] {texts[j]}")
                    for res in result:
                        if res['label'] == 'CONTRADICTION': max_score = max(max_score, res['score'])
                except Exception: continue
        if max_score > self.config.contradiction_threshold: warnings.append("CONTRADICTORY_EVIDENCE")
        return max_score, warnings

class LLMAsJudgeValidator:
    def __init__(self, config: PipelineConfig, client: OpenAI):
        self.config, self.client = config, client
    def _robust_score_parse(self, score_input: Any) -> float:
        if isinstance(score_input, (int, float)): return float(score_input)
        if isinstance(score_input, str):
            numbers = re.findall(r'\d+', score_input)
            if numbers: return float(numbers[0])
        return 0.0
    def evaluate_answer(self, query: Query, answer: str, evidence_map: Dict[str, Sentence], reasoning_chain: str = "") -> JudgeEvaluation:
        if not answer or not evidence_map: return JudgeEvaluation(False, 0.0, "No answer/evidence.", 0.0, "N/A")
        clean_answer = re.sub(r'\s*\[\d+\]', '', str(answer))
        evidence_texts = "\n".join([f"EVIDENCE_{i+1}: {s.text}" for i, s in enumerate(evidence_map.values())])
        
        # "Synthetic Judge" prompt
        prompt = f"""You are a meticulous and impartial evaluator. Your task is to perform a forensic analysis of a generated answer.

<INSTRUCTIONS>
1.  **Analyze Claims**: Break down the <GENERATED_ANSWER> into individual claims.
2.  **Map Evidence**: For each claim, find ALL relevant evidence IDs from <EVIDENCE> that support it. A claim may be a logical synthesis of MULTIPLE evidence IDs.
3.  **Validate Reasoning**: If a <PIPELINE_REASONING_CHAIN> is provided, your primary goal is to assess if the answer's logic faithfully follows that chain.
4.  **Score Faithfulness (INTEGER 1-5)**: Rate if every claim is fully supported by the evidence. If a reasoning chain is provided, rate if the answer adheres to it. 5 is perfect adherence. 1 means it deviates or is unsupported.
5.  **Score Relevance (INTEGER 1-5)**: Rate how well the answer addresses the <QUESTION>. 5 is a perfect answer. 1 is off-topic.
6.  **Final JSON Output**: Provide your analysis in a single, valid JSON object with INTEGER scores.
</INSTRUCTIONS>

<EVIDENCE>
{evidence_texts}
</EVIDENCE>

<QUESTION>
{query.text}
</QUESTION>

<GENERATED_ANSWER>
{clean_answer}
</GENERATED_ANSWER>"""

        if reasoning_chain:
            prompt += f"""
<PIPELINE_REASONING_CHAIN>
{reasoning_chain}
</PIPELINE_REASONING_CHAIN>"""

        prompt += f"""
<OUTPUT_FORMAT>
{{"chain_of_thought": [], "faithfulness_score": 5, "faithfulness_reasoning": "...", "relevance_score": 5, "relevance_reasoning": "..."}}
</OUTPUT_FORMAT>

Begin your forensic analysis now:"""
        
        response = RobustErrorHandler.safe_llm_call(self.client, "LLM-as-a-Judge", "{}", model=self.config.fpt_judge_model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
        
        try:
            data = json.loads(response)
            fs_raw = self._robust_score_parse(data.get("faithfulness_score", 0))
            rs_raw = self._robust_score_parse(data.get("relevance_score", 0))
            fs, rs = fs_raw / 5.0, rs_raw / 5.0
            return JudgeEvaluation(fs>0.8, fs, data.get("faithfulness_reasoning","N/A"), rs, data.get("relevance_reasoning","N/A"))
        except Exception as e:
            logger.error(f"Judge failed to parse response: {e}"); return JudgeEvaluation(False, 0.0, "Parse error.", 0.0, "N/A")

# ==============================================================================
# --- CORE COMPONENTS ---
# ==============================================================================
class RobustErrorHandler:
    @staticmethod
    def safe_execute(op, func, fallback, *a, **kw):
        try: return func(*a, **kw)
        except Exception as e: logger.warning(f"{op} failed: {e}... Using fallback."); return fallback
    @staticmethod
    def safe_llm_call(client, op, fallback, **params):
        try: return client.chat.completions.create(**params).choices[0].message.content.strip()
        except Exception as e: logger.error(f"LLM call {op} failed: {e}"); return fallback

class DataManager:
    def load_documents(self, fp: str) -> Dict[str, Document]:
        docs = {};
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data = json.loads(line); docs[data['doc_id']] = Document(**data)
        return docs
    def load_queries(self, fp: str) -> Dict[str, Query]:
        queries = {};
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data = json.loads(line); queries[data['query_id']] = Query(**data)
        return queries

class EnhancedPreprocessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sbert_model = SentenceTransformer(config.sbert_model_name, device=DEVICE)
        self.ner_pipeline = hf_pipeline("ner", model=config.ner_model_name, grouped_entities=True, device=0 if DEVICE=="cuda" else -1)
        # Pre-compile regex for performance
        self.causal_regex = re.compile(r'\b(cause|because|due to|result|lead to|consequently|therefore|thus)\b', re.IGNORECASE)
        self.rhetorical_patterns = {
            'Main_Claim': re.compile(r'\b(argue|claim|assert|believe|conclude|propose|suggest)\b', re.IGNORECASE),
            'Supporting_Evidence': re.compile(r'\b(evidence|data|research|study|found|showed|demonstrated)\b', re.IGNORECASE),
            'Expert_Opinion': re.compile(r'\b(according to|stated|opinion|expert|analyst)\b', re.IGNORECASE),
        }
    def get_embedding(self, text: str, prefix: str) -> np.ndarray:
        """
        This function is now guaranteed to always return a valid numpy array.
        It handles potential silent failures from the encoder and bad inputs.
        """
        # --- Pre-computation Check ---
        if not text or not isinstance(text, str):
            logger.warning(f"get_embedding received invalid input: type={type(text)}, value='{text}'. Returning a zero vector.")
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        if not text.strip():
            logger.warning("get_embedding received an empty or whitespace-only string. Returning a zero vector.")
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)

        # --- Cache Check ---
        key = f"{prefix}:{text}"
        text_hash = hashlib.sha256(key.encode()).hexdigest()
        if (cached_emb := EMBEDDING_CACHE.get(text_hash)) is not None:
            if isinstance(cached_emb, np.ndarray):
                return cached_emb
        
        # --- Encoding ---
        try:
            emb = self.sbert_model.encode(f"{prefix}: {text}", convert_to_numpy=True, show_progress_bar=False)
            
            if emb is None:
                raise ValueError("SBERT model returned None for a valid string.")

            EMBEDDING_CACHE[text_hash] = emb
            EMBEDDING_CACHE.sync()
            return emb
            
        except Exception as e:
            logger.error(f"A critical error occurred in get_embedding for text '{text[:100]}...': {e}")
            # In case of ANY failure, return a zero vector. This PREVENTS the pipeline from crashing.
            return np.zeros(self.sbert_model.get_sentence_embedding_dimension(), dtype=np.float32)
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extracts key technical terms and named entities."""
        # The NER pipeline returns a dict, so we must access the 'word' key.
        concepts = set(ent['word'] for ent in self.ner_pipeline(text) if isinstance(ent, dict) and 'word' in ent and len(ent['word'].split()) > 1)
        return list(concepts)
        EMBEDDING_CACHE[text_hash] = emb; EMBEDDING_CACHE.sync(); return emb
    
    def _classify_rhetorical_role(self, text: str) -> str:
        """High-performance, regex-based rhetorical classification."""
        for role, pattern in self.rhetorical_patterns.items():
            if pattern.search(text):
                return role
        return "Background_Information"

    def _detect_causality(self, text: str) -> bool:
        """High-performance, regex-based causality detection."""
        return bool(self.causal_regex.search(text))

    def process_documents_robust(self, documents: Dict[str, Document]) -> List[Sentence]:
        texts, sentence_map = [], []
        for doc_id, doc in documents.items():
            sents = sent_tokenize(doc.text)
            for sent_idx, text in enumerate(sents):
                if 4 < len(text.split()) < 250: 
                    texts.append(text)
                    sentence_map.append({'doc_id': doc_id, 'sent_idx': sent_idx})

        embeddings = np.array([self.get_embedding(t, "passage") for t in tqdm(texts, "Embedding", leave=False)])
        
        all_sentences = []
        # Process in batches
        batch_size = 128
        for i in tqdm(range(0, len(texts), batch_size), desc="Enriching Sentences"):
            batch_texts = texts[i:i + batch_size]
            batch_info = sentence_map[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # Batch NER
            ner_results = self.ner_pipeline(batch_texts)

            for j, text in enumerate(batch_texts):
                info = batch_info[j]
                # Extract entities from the batched result
                entities = [e['word'] for e in ner_results[j] if isinstance(e, dict) and 'word' in e]

                all_sentences.append(Sentence(
                    doc_id=info['doc_id'],
                    sent_idx=info['sent_idx'],
                    text=text,
                    embedding=batch_embeddings[j],
                    hash=hashlib.sha256(text.encode()).hexdigest(),
                    entities=entities,
                    rhetorical_role=self._classify_rhetorical_role(text),
                    is_causal=self._detect_causality(text)
                ))
        
        return all_sentences

class NeuralRanker:
    def __init__(self, model_path: str, sbert_model: SentenceTransformer, config: PipelineConfig, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        
        self.sbert_model = sbert_model
        self.config = config
    @torch.no_grad()
    def rank_with_scores(self, query: str, sentences: List[Sentence]) -> List[Sentence]:
        """
        Hardened tensor shape handling to prevent IndexError.
        Ensures all embeddings are correctly treated as 2D batches.
        """
        if not sentences: return []

        # --- Pass 1: Seed Identification & Dynamic Hop Assignment ---
        query_embedding_raw = self.sbert_model.encode(f"query: {query}", convert_to_tensor=True, show_progress_bar=False)
        
        # Ensure query embedding is a 2D tensor (a batch of 1)
        # The .unsqueeze(0) adds the necessary batch dimension: [768] -> [1, 768]
        query_embedding = util.normalize_embeddings(query_embedding_raw.unsqueeze(0))
        
        # Ensure sentence embeddings are also correctly shaped and on the correct device
        sentence_embeddings_np = np.array([s.embedding for s in sentences])
        sentence_embeddings_tensor = torch.tensor(sentence_embeddings_np, device=self.device, dtype=torch.float)
        sentence_embeddings = util.normalize_embeddings(sentence_embeddings_tensor)

        # Calculate cosine similarities
        # The query is a [1, 768] tensor, sentences is a [N, 768] tensor. The result will be [1, N].
        similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0] # take the first (and only) row
        
        top_seed_idx = torch.argmax(similarities).item()
        
        hop_levels = []
        seed_embedding = sentence_embeddings[top_seed_idx].unsqueeze(0) # Also needs to be a 2D batch of 1
        for i in range(len(sentences)):
            # Calculate similarity of each sentence to the seed
            sim_to_seed_tensor = util.pytorch_cos_sim(sentence_embeddings[i].unsqueeze(0), seed_embedding)[0][0]
            sim_to_seed = sim_to_seed_tensor.item()

            if similarities[i] > 0.85 or sim_to_seed > 0.9:
                hop_levels.append(0)
            elif similarities[i] > 0.7 or sim_to_seed > 0.75:
                hop_levels.append(1)
            else:
                hop_levels.append(2)

        # --- Pass 2: Construct Rich Inputs and Perform Full Ranking ---
        inputs = []
        for i, s in enumerate(sentences):
            components = [f"[HOP:{hop_levels[i]}]", f"[ROLE:{s.rhetorical_role}]"]
            if s.entities: components.append(f"[ENT:{','.join(s.entities[:3])}]")
            if s.is_causal: components.append("[CAUSAL]")
            feature_str = " ".join(components)
            inputs.append(f"{feature_str} {query} [SEP] {s.text}")

        scores = []
        for i in range(0, len(inputs), 32):
            batch = self.tokenizer(inputs[i:i+32], padding=True, truncation=True, max_length=self.config.max_length, return_tensors="pt").to(self.device)
            batch_scores = torch.sigmoid(self.model(**batch).logits).squeeze(-1).cpu().numpy().tolist()
            scores.extend(batch_scores)
            
        for s, score in zip(sentences, scores):
            s.relevance_score = float(score)
            
        return sorted(sentences, key=lambda s: s.relevance_score, reverse=True)

# ==============================================================================
# --- V25.3: KEYSTONE RAG CONTROLLER ---
# ==============================================================================
import hdbscan
from sklearn.cluster import SpectralClustering
from collections import Counter
class KeystoneRAGController:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_manager = DataManager()
        self.preprocessor = EnhancedPreprocessor(config)
        self.neural_ranker = NeuralRanker(
            model_path=config.neural_ranker_path, 
            sbert_model=self.preprocessor.sbert_model, 
            config=config 
        )
        self.fpt_client = OpenAI(api_key=config.fpt_api_key, base_url=config.fpt_base_url)
        self.evidence_validator = EvidenceValidator(config)
        self.judge = LLMAsJudgeValidator(config, self.fpt_client)
        self.ner_pipeline = hf_pipeline("ner", model=config.ner_model_name, grouped_entities=True, device=0 if DEVICE=="cuda" else -1)
        self.documents, self.queries, self.sentence_pool, self.sentence_index = {}, {}, [], None

    def setup(self, doc_file: str, query_file: str, force_reingest: bool = False):
        try:
            with open(doc_file, 'rb') as f1, open(query_file, 'rb') as f2:
                state_hash = hashlib.md5(f1.read() + f2.read()).hexdigest()
        except FileNotFoundError:
            logger.critical("Document or query file not found. Cannot proceed."); return
        cache_dir = "cache"; os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cache_{state_hash}_v28.0_apex_ranker.pkl")
        if force_reingest and os.path.exists(cache_path):
            logger.warning("Forcing re-ingestion. Deleting existing cache..."); os.remove(cache_path)
        if os.path.exists(cache_path):
            logger.info(f"Loading V28.0 Apex Ranker corpus from cache: {cache_path}")
            with open(cache_path, "rb") as f: data = pickle.load(f)
            self.documents, self.queries, self.sentence_pool, self.sentence_index = \
                data['docs'], data['queries'], data['pool'], data['faiss']
        else:
            logger.info("No valid cache found. Starting full pre-computation...")
            self.documents = self.data_manager.load_documents(doc_file)
            self.queries = self.data_manager.load_queries(query_file)
            self.sentence_pool = self.preprocessor.process_documents_robust(self.documents)
            embs = np.array([s.embedding for s in self.sentence_pool]).astype('float32')
            faiss.normalize_L2(embs); self.sentence_index = faiss.IndexFlatIP(embs.shape[1]); self.sentence_index.add(embs)
            logger.info(f"Caching new state to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({'docs': self.documents, 'queries': self.queries, 'pool': self.sentence_pool, 'faiss': self.sentence_index}, f)
        logger.info("V28.0 Apex Ranker RAG Controller setup complete.")
    
    
    
    def _get_scoped_indices(self, doc_ids: List[str]) -> np.ndarray:
        if "all" in doc_ids: return np.arange(len(self.sentence_pool), dtype=np.int64)
        return np.array([i for i, s in enumerate(self.sentence_pool) if s.doc_id in doc_ids], dtype=np.int64)
    
    def _run_bridge_retrieval(self, query: Query, evidence: List[Sentence]) -> List[Sentence]:
        """
        Upgraded Auditor with pre-computed concept hints.
        """
        logger.info("Auditor module activated: Analyzing evidence for conceptual gaps...")
        
        # 1. Gather conceptual hints from the source documents of the evidence
        source_doc_ids = list(set(s.doc_id for s in evidence))
        concept_hints = []
        for doc_id in source_doc_ids:
            concept_hints.extend(self.doc_to_concepts_map.get(doc_id, []))
        
        evidence_text = "\n".join([f"- {s.text}" for s in evidence])
        
        # 2. prompt with conceptual hints
        prompt = f"""You are a specialist in cross-domain analysis and forensic reasoning. Your mission is to uncover the hidden link between seemingly unrelated pieces of information.

    <MISSION>
    Based on the query and the disparate evidence provided, your task is to pinpoint the **single, underlying physical phenomenon, specific technical term, or named entity** that mechanistically or causally connects the topics. The answer is the "missing piece" that explains the relationship.
    </MISSION>

    <INSTRUCTIONS>
    1.  **Analyze Domains**: Identify the core subjects of the different evidence fragments (e.g., one is about neuroscience, the other is about quantum computing).
    2.  **Hypothesize Connection**: Ask yourself "What could possibly cause the problem in Domain A *and* the problem in Domain B?".
    3.  **Scan Hints for Candidate**: Scrutinize the <CONCEPT_HINTS_FROM_SOURCE_DOCS>. The true bridge concept is very likely listed there. This is your primary search area.
    4.  **Validate Hypothesis**: The correct answer must be a specific concept that logically fits as a common cause or link, not just a shared high-level topic.
    </INSTRUCTIONS>

    <CRITICAL_DISTINCTION>
    Do NOT identify a generic concept that is merely *present* in both domains. For example, if both evidence pieces describe an AI system used to solve a problem, the bridge is NOT "AI". The bridge is the *underlying problem* that both AIs are designed to address (e.g., "anomalous particle flux"). You are looking for the shared cause, not the shared solution type.
    </CRITICAL_DISTINCTION>

    <MAIN_QUERY>
    {query.text}
    </MAIN_QUERY>

    <DISPARATE_EVIDENCE>
    {evidence_text[:2500]} 
    </DISPARATE_EVIDENCE>

    <CONCEPT_HINTS_FROM_SOURCE_DOCS>
    {list(set(concept_hints))}
    </CONCEPT_HINTS_FROM_SOURCE_DOCS>

    <OUTPUT_FORMAT>
    Respond with ONLY the name of the bridge concept. If no single concept can logically connect the evidence, respond with the single word "NONE".
    </OUTPUT_FORMAT>

    Bridge Concept Name:"""

        bridge_concept = RobustErrorHandler.safe_llm_call(self.fpt_client, "Bridge Concept Identification", "NONE",
            model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)

        if "NONE" in bridge_concept or len(bridge_concept) > 50:
            logger.warning("Auditor could not identify a clear bridge concept.")
            return []

        logger.info(f"Auditor identified potential bridge concept: '{bridge_concept}'")

        # 3. Targeted Retrieval 
        bridge_evidence = self._run_retrieval(query_texts=[bridge_concept], doc_ids=["all"])
        return bridge_evidence[:3]
    def answer_query(self, query: Query) -> Dict[str, Any]:
        """
        A streamlined, unified pipeline that leverages the full power
        of the enriched NeuralRanker. The need for explicit, separate multi-hop modules
        is removed as the ranker now intelligently identifies and boosts bridge sentences.
        """
        start_time = time.time()
        
        # STAGE 1: Broad Retrieval & Expansion
        # gather a wide net of potentially relevant information.
        subjects = self._extract_query_subjects(query)
        retrieval_queries = [query.text] + subjects
        
        initial_evidence = self._run_retrieval(retrieval_queries, query.doc_ids)
        expanded_evidence = self._run_contextual_expansion(query, initial_evidence, subjects)

        # STAGE 2: Intelligent Evidence Selection
        # The enriched ranker provides the primar signal, and MMR ensures the final context is diverse and comprehensive.
        # The new ranker's dynamic HOP assignment naturally boosts bridge sentence making the explicit, brittle bridge-finding modules redundant.
        final_evidence = self._select_final_evidence_with_mmr(query, expanded_evidence)

        # STAGE 3: Generation & Validation
        # The generation process is simple as it receives a higher quality context
        answer, evidence_map = self._generate_answer_and_citations(query, final_evidence)
        
        # Validation remains the same, but no longer needs to handle a reasoning_chain.
        contradiction_score, warnings = self.evidence_validator.check_for_contradictions(final_evidence)
        score, eval_details = 0.0, "Evaluation Disabled"
        
        if self.config.use_llm_as_judge:
            judge_eval = self.judge.evaluate_answer(query, answer, evidence_map) # Simpler call
            if not judge_eval.is_faithful:
                warnings.append("JUDGE_FOUND_UNFAITHFUL")
            score = (judge_eval.faithfulness_score * 0.7) + (judge_eval.relevance_score * 0.3)
            eval_details = judge_eval._asdict()
        else:
            score = (1.0 - (len(warnings) * 0.2)) * (1.0 - contradiction_score)
            eval_details = "LLM-as-a-Judge is disabled. Using proxy score."

        return {
            'query_id': query.query_id,
            'answer': str(answer),
            'tier_used': 'Unified Apex Ranker Path',
            'processing_time': time.time() - start_time,
            'warnings': list(set(warnings)),
            'evidence_contradiction_score': contradiction_score,
            'llm_judge_evaluation': eval_details,
            'final_confidence_score': score
        }
        
        contradiction_score, warnings = self.evidence_validator.check_for_contradictions(final_evidence)
        score, eval_details = 0.0, "Evaluation Disabled"
        if self.config.use_llm_as_judge:
            judge_eval = self.judge.evaluate_answer(query, answer, evidence_map, reasoning_chain)
            if not judge_eval.is_faithful: warnings.append("JUDGE_FOUND_UNFAITHFUL")
            score, eval_details = (judge_eval.faithfulness_score*0.7)+(judge_eval.relevance_score*0.3), judge_eval._asdict()
        else:
            score = (1.0 - (len(warnings) * 0.2)) * (1.0 - contradiction_score)
            eval_details = "LLM-as-a-Judge is disabled. Using proxy score."

        return {'query_id': query.query_id, 'answer': str(answer), 'tier_used': 'Unified Apex Path', 'processing_time': time.time() - start_time,
                'warnings': list(set(warnings)), 'evidence_contradiction_score': contradiction_score,
                'llm_judge_evaluation': eval_details, 'final_confidence_score': score}

    def _extract_query_subjects(self, query: Query) -> List[str]:
        """
        Sanitizes NER output and explicitly filters empty strings.
        """
        subjects = []
        try:
            ner_entities = self.ner_pipeline(query.text)
            if ner_entities and isinstance(ner_entities, list):
                sanitized_subjects = set()
                for e in ner_entities:
                    if isinstance(e, dict) and 'word' in e:
                        raw_word = e['word']
                        logger.debug(f"Raw NER entity: '{raw_word}'")
                        
                        # Handle apostrophes by replacing with space (not removing)
                        cleaned = raw_word.replace("'", " ").replace("’", " ")
                        
                        # Remove other punctuation but keep spaces and hyphens
                        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
                        
                        # Normalize spaces
                        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                        
                        logger.debug(f"Cleaned entity: '{cleaned}'")
                        
                        # Only add if non-empty and meaningful
                        if cleaned and len(cleaned) > 1:
                            # Split into words and filter out single-character words
                            words = [word for word in cleaned.split() if len(word) > 1]
                            if words:
                                # Rejoin to maintain multi-word entities
                                final_entity = ' '.join(words)
                                sanitized_subjects.add(final_entity)
                                logger.debug(f"Added subject: '{final_entity}'")
                
                subjects = list(sanitized_subjects)
                logger.info(f"Final extracted subjects: {subjects}")
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            pass  # Fallback to LLM if NER fails

        if not subjects:
            logger.warning("NER failed or found no subjects, falling back to LLM extraction.")
            prompt = f"""Extract the 1-3 primary named entities or technical subjects from the user query. Output ONLY a valid JSON list of strings.\nQuery: "{query.text}"\nExample Output: ["Quantum Nexus Initiative", "Arbor BCI"]\nSubjects:"""
            response = RobustErrorHandler.safe_llm_call(self.fpt_client, "Subject Extraction", "[]", model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
            try:
                subjects = json.loads(response)
                # Also filter the LLM's output just in case
                subjects = [s for s in subjects if isinstance(s, str) and s.strip() and len(s.strip()) > 1]
            except json.JSONDecodeError:
                subjects = []
        
        return subjects

    def _run_retrieval(self, query_texts: List[str], doc_ids: List[str]) -> List[Sentence]:
        if not query_texts: # Add a guard clause for safety
            return []
        scoped_indices = self._get_scoped_indices(doc_ids)
        if len(scoped_indices) == 0: return []
        q_embs = np.array([self.preprocessor.get_embedding(q, "query") for q in query_texts])
        k_val = min(self.config.retrieval_k, len(scoped_indices))
        selector = faiss.IDSelectorArray(scoped_indices)
        candidate_map = {}
        _, ids = self.sentence_index.search(q_embs, k=k_val, params=faiss.SearchParameters(sel=selector))
        for i in ids.flatten():
            if i != -1: candidate_map[self.sentence_pool[i].hash] = self.sentence_pool[i]
        return self.neural_ranker.rank_with_scores(query_texts[0], list(candidate_map.values()))

    def _run_contextual_expansion(self, query: Query, evidence: List[Sentence], subjects: List[str]) -> List[Sentence]:
        if not subjects: return evidence
        evidence_text = " ".join([s.text for s in evidence])
        missing_subjects = [s for s in subjects if s.lower() not in evidence_text.lower()]
        if not missing_subjects: return evidence
        logger.info(f"Contextual Expansion: Searching for missing subject(s): {missing_subjects}")
        expansion_evidence = {}
        for subject in missing_subjects:
            # Add more robust validation before creating query
            if subject and isinstance(subject, str) and subject.strip() and len(subject.strip()) > 1:
                expansion_candidates = self._run_retrieval([f"What is {subject}?"], query.doc_ids)
                if expansion_candidates: expansion_evidence[expansion_candidates[0].hash] = expansion_candidates[0]
            else:
                logger.warning(f"Skipping invalid subject: '{subject}'")
        final_evidence_map = {s.hash: s for s in evidence}; final_evidence_map.update(expansion_evidence)
        return self.neural_ranker.rank_with_scores(query.text, list(final_evidence_map.values()))
    
    def _select_final_evidence_with_mmr(self, query: Query, candidates: List[Sentence]) -> List[Sentence]:
        if not candidates: return []
        target_count = min(self.config.final_evidence_count, len(candidates))
        if len(candidates) <= target_count: return candidates
        candidate_embeddings = np.array([s.embedding for s in candidates])
        query_embedding = self.preprocessor.get_embedding(query.text, "query")
        selected_indices = [0]
        while len(selected_indices) < target_count:
            best_next_idx, max_mmr_score = -1, -np.inf
            selected_embeddings = candidate_embeddings[selected_indices]
            for i in range(len(candidates)):
                if i in selected_indices: continue
                relevance = util.cos_sim(query_embedding, candidate_embeddings[i])[0][0].item()
                redundancy = np.max(util.cos_sim(candidate_embeddings[i], selected_embeddings)[0].cpu().numpy())
                mmr_score = (1 - self.config.mmr_lambda) * relevance - self.config.mmr_lambda * redundancy
                if mmr_score > max_mmr_score: max_mmr_score, best_next_idx = mmr_score, i
            if best_next_idx == -1: break
            selected_indices.append(best_next_idx)
        return [candidates[i] for i in selected_indices]
    
    # Logical Weaver module
    def _run_logical_weaver(self, query: Query, facts: str) -> str:
        """
        uses a non-leaky example to teach the reasoning pattern.
        """
        logger.info("Activating Logical Weaver for multi-hop synthesis guidance...")
        
        prompt = f"""You are a reasoning architect. Your task is to create a step-by-step reasoning chain that logically connects the provided Disparate Facts to answer the Main Query.
Identify the 'bridge concepts' or 'common threads' that link the facts. For every logical step you propose, you MUST cite the specific fact numbers that support that step.
Your output must be a clear, logical plan that the final writer will follow.

<MAIN_QUERY>
{query.text}
</MAIN_QUERY>

{query.text}
</MAIN_QUERY>

<DISPARATE_FACTS>
{facts}
</DISPARATE_FACTS>

<EXAMPLE_OF_A_GOOD_REASONING_CHAIN>
- **Query:** "How could the primary operational risk of the 'Project Star-View' satellite constellation be mechanistically linked to the main safety concern of the 'CRISPR-Z' gene-editing platform?"
- **Reasoning Chain:**
1. The main safety concern for the 'CRISPR-Z' platform is its high rate of off-target mutations. [Fact 3]
2. The primary operational risk for the 'Project Star-View' constellation is sensor damage from a specific type of solar flare. [Fact 7]
3. A newly discovered form of radiation, "Hyper-Ionized Gamma Bursts (HIGs)," has been identified as the primary component of these solar flares. [Fact 5]
4. Crucially, these same HIGs have also been documented to be a direct cause of the off-target mutations seen in gene-editing systems. [Fact 5, Fact 6]
5. Therefore, HIGs are the direct mechanistic link. They are the physical phenomenon responsible for both the satellite sensor damage and the gene-editing errors, connecting the two disparate domains through a shared vulnerability to a specific type of cosmic radiation. [Connects Fact 3, 7, and 5]
</EXAMPLE_OF_A_GOOD_REASONING_CHAIN>

Reasoning Chain:"""

        return RobustErrorHandler.safe_llm_call(self.fpt_client, "Logical Weaver", "", model=self.config.fpt_model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)

    def _generate_answer_and_citations(self, query: Query, evidence: List[Sentence]) -> Tuple[str, Dict[str, Sentence]]: # Return signature changed
        if not evidence:
            return "Insufficient evidence to construct an answer.", {}
        
        evidence_map = {f"EVIDENCE_{i+1}": s for i, s in enumerate(evidence)}
        fact_sheet = "\n".join([f"[{k}] {s.text}" for k, s in evidence_map.items()])
        
        extract_prompt = f"""Based ONLY on the provided Fact Sheet, extract a list of key facts as bullet points that directly help answer the User Query.\n<USER_QUERY>{query.text}</USER_QUERY>\n<FACT_SHEET>{fact_sheet}</FACT_SHEET>\nExtracted Facts:"""
        facts = RobustErrorHandler.safe_llm_call(self.fpt_client, "Fact Extraction", "", model=self.config.fpt_model_name, messages=[{"role": "user", "content": extract_prompt}], temperature=0.1)
        
        if not facts:
            return "Could not extract relevant facts from the evidence.", evidence_map

        # The synthesis prompt is now simpler, without the reasoning chain instruction
        synthesis_prompt = f"""Synthesize the following Key Facts into a cohesive, well-written paragraph that directly answers the User Query. Do not add any new information.
<USER_QUERY>{query.text}</USER_QUERY>
<KEY_FACTS>{facts}</KEY_FACTS>
Final Answer:"""
        
        answer = RobustErrorHandler.safe_llm_call(self.fpt_client, "Answer Synthesis", "Could not synthesize an answer.", model=self.config.fpt_model_name, messages=[{"role": "user", "content": synthesis_prompt}], temperature=0.0)
        
        final_answer = self._render_citations(answer, evidence)
        return final_answer, evidence_map
        
    def _render_citations(self, answer: str, evidence: List[Sentence]) -> str:
        cited_answer = str(answer); source_map = {s.text: i+1 for i, s in enumerate(evidence)}
        answer_sents, final_sents = sent_tokenize(cited_answer), []
        if not evidence: return cited_answer
        evidence_embs = np.array([s.embedding for s in evidence])
        for ans_sent in answer_sents:
            if not ans_sent: continue
            ans_sent_emb = self.preprocessor.get_embedding(ans_sent, "query")
            sims = util.cos_sim(ans_sent_emb, evidence_embs)[0].cpu().numpy()
            best_idx = np.argmax(sims)
            if sims[best_idx] > 0.7:
                best_evidence_text = evidence[best_idx].text
                if (citation_num := source_map.get(best_evidence_text)):
                    final_sents.append(f"{ans_sent.strip()} [{citation_num}]")
                else: final_sents.append(ans_sent.strip())
            else: final_sents.append(ans_sent.strip())
        cited_answer = " ".join(final_sents)
        citation_list = "\n\n--- Citations ---\n" + "".join([f"[{i}] {text}\n" for text, i in source_map.items()])
        return cited_answer + citation_list

    def run_interactive_session(self):
        print("\n" + "="*80 + "\n🚀 V25.3 - Keystone Architecture (Final, Polished Build)\n" + "="*80)
        user_input = input("Enable LLM-as-a-Judge evaluation? (yes/no): ").strip().lower()
        self.config.use_llm_as_judge = user_input in ['yes', 'y', '1']
        logger.info(f"LLM-as-a-Judge ENABLED: {self.config.use_llm_as_judge}")
        while True:
            mode = input("\n[1] Pre-loaded Queries, [2] Interactive, [quit]: ").strip()
            if mode == 'quit': break
            if mode == '1' and self.queries: self._run_preloaded()
            elif mode == '2': self._run_interactive()
            else: print("Invalid choice.")
    def _run_preloaded(self):
        for q_id, query in self.queries.items():
            print(f"\n{'='*60}\nProcessing Query: {q_id} - {query.text}\n{'='*60}")
            self._display_response(self.answer_query(query))
    def _run_interactive(self):
        while True:
            q_text = input("\nEnter query (or 'back'): ").strip()
            if q_text.lower() == 'back': break
            doc_ids = [s.strip() for s in input("Enter doc IDs (comma-separated) or 'all': ").split(',')]
            self._display_response(self.answer_query(Query("interactive", q_text, doc_ids)))
    def _display_response(self, response: Dict[str, Any]):
        print(f"\n✅ ANSWER (using {response['tier_used']}):\n{response['answer']}")
        print(f"\n📊 VALIDATION & PERFORMANCE:")
        print(f"  - Final Confidence Score: {response.get('final_confidence_score', 0.0):.3f}")
        print(f"  - Processing Time: {response['processing_time']:.2f}s")
        if 'evidence_contradiction_score' in response: print(f"  - Evidence Contradiction Score: {response['evidence_contradiction_score']:.3f}")
        if response.get('warnings'): print(f"  - ⚠️  Warnings: {', '.join(sorted(list(set(response['warnings']))))}")
        if self.config.use_llm_as_judge and isinstance(response.get('llm_judge_evaluation'), dict):
            judge = response['llm_judge_evaluation']
            print("\n🔬 LLM-AS-A-JUDGE EVALUATION:")
            print(f"  - Faithfulness: {judge['faithfulness_score']:.2f}/1.00 | Reasoning: {judge['faithfulness_reasoning']}")
            print(f"  - Relevance:    {judge['relevance_score']:.2f}/1.00 | Reasoning: {judge['relevance_reasoning']}")

# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the V25.3 Keystone RAG Pipeline.")
    parser.add_argument("--reingest", action="store_true", help="Force deletion of existing cache and re-ingest all data.")
    args = parser.parse_args()
    
    DOCS_FILE, QUERIES_FILE = "docs.jsonl", "queries.jsonl"
    if not os.path.exists(DOCS_FILE) or not os.path.exists(QUERIES_FILE):
        docs_content = """
{"doc_id": "TECH-MEM-MNEMOSYNE", "text": "Project Mnemosyne, a DARPA initiative headquartered at MIT's Media Lab, is developing a next-generation Brain-Computer Interface (BCI) focused on direct memory encoding and retrieval. The system uses a novel 'neuro-photonic' implant that translates digital data into precisely targeted light patterns to stimulate and modify hippocampal engrams. While early results have shown an unprecedented 98% recall accuracy for encoded information, the primary operational risk is 'synaptic interference.' This phenomenon occurs when the implant's photonic emissions inadvertently disrupt adjacent, unrelated memory traces, leading to a form of structured amnesia or memory corruption. The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target. To counter this, the team developed a sophisticated AI called the 'Predictive Hebbian Modulator.' This is a recurrent neural network with a temporal-convolutional attention mechanism that learns the unique synaptic potentiation patterns of an individual's brain. It then pre-emptively adjusts the implant's light frequency and intensity to create a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects. The entire process requires immense computational power, a dependency which has made securing the GPU supply chain a top-level project concern."}
{"doc_id": "QPU-SIM-EREBUS", "text": "At Lawrence Livermore National Laboratory, Project Erebus is a major Department of Energy program aimed at using quantum computers to simulate the behavior of dark matter. The project's quantum processing unit (QPU), a 4,096-qubit topological device, is designed to solve complex quantum chromodynamics equations that are intractable for classical supercomputers. The most significant technical hurdle is a persistent issue termed 'Entanglement Fraying.' This is a specific form of decoherence where the fragile quantum entanglement between distant qubits decays exponentially faster than predicted by standard models, leading to a collapse of the simulation's integrity after only a few hundred microseconds. Analysis has revealed that this accelerated decay is strongly correlated with the same anomalous 'exotic particle flux' documented by other advanced research projects. The Erebus team's solution is an AI error-correction model that runs on a classical co-processor. The AI analyzes the parity-check measurements from the QPU in real-time and predicts the onset of a fraying event. It then instructs the QPU's control system to perform a series of 'entanglement distillation' protocols, sacrificing some qubits to reinforce the stability of the remaining computational set. While this extends the simulation time, it significantly increases the overall number of qubits required, raising concerns about the long-term scalability of the approach."}
{"doc_id": "SPACE-SAIL-HELIOS", "text": "NASA's Project Helios is an ambitious plan to send an unmanned probe to Alpha Centauri using a light sail propelled by a high-powered laser array stationed in Earth orbit. The sail itself is a kilometer-scale, atomically thin sheet of a graphene-molybdenum disulfide heterostructure. The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium. Extensive testing at JPL revealed that the degradation is not caused by conventional protons or alpha particles, but is almost entirely attributable to the same high-energy, 'exotic particle flux' that has been observed affecting quantum and neurological experiments. The proposed mitigation involves a 'self-healing' matrix interwoven into the sail's lattice. A predictive AI model monitors the sail for signs of micro-fracturing. When a potential failure point is detected, the AI activates a localized energy field that triggers a chemical reaction in an embedded substrate, repairing the lattice structure. Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation."}
{"doc_id": "GOV-STRAT-ORION", "text": "A declassified strategic document, known as the 'Orion Mandate,' outlines the United States' primary technological goals for the next decade. The mandate establishes a national priority to achieve 'Cognitive-Computational Supremacy,' defined as the synergistic mastery of next-generation computing, artificial intelligence, and direct neural interface technologies. The document explicitly names three critical, interdependent pillars for achieving this goal: Project Erebus (for computational dominance), Project Mnemosyne (for neural interface superiority), and Project Helios (for demonstrating advanced materials and energy projection). The mandate's primary concern is a new strategic threat termed 'asymmetric temporal sabotage.' This is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation. The document concludes by authorizing a new, top-secret research initiative under the NSA, tasked with identifying and shielding against the 'anomalous high-energy particle phenomena' that have been reported to interfere with all three pillar projects, flagging it as the most likely vector for such a sabotage campaign."}
"""
        with open(DOCS_FILE, "w", encoding='utf-8') as f: f.write(docs_content.strip())
        queries_content = """
{"query_id": "Q1-STRESS-DEEP-FACTUAL", "text": "Describe the specific mechanism of 'synaptic interference' in Project Mnemosyne, and detail the architecture and function of the 'Predictive Hebbian Modulator' AI used to counteract it.", "doc_ids": ["TECH-MEM-MNEMOSYNE"]}
{"query_id": "Q2-STRESS-ABSTRACT-SYNTHESIS", "text": "Synthesize the core challenges of Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion). What single, unifying principle of 'long-term material and informational integrity' do both projects struggle with, and how do their respective AI-driven solutions represent a similar philosophical approach to predictive maintenance?", "doc_ids": ["QPU-SIM-EREBUS", "SPACE-SAIL-HELIOS"]}
{"query_id": "Q3-STRESS-GRAND-MULTIHOP", "text": "According to the Orion Mandate, what is 'asymmetric temporal sabotage,' and what is the specific, underlying physical phenomenon (not mentioned in the Mandate itself) that mechanistically links the operational risks of all three critical projects: Mnemosyne, Erebus, and Helios?", "doc_ids": ["all"]}
{"query_id": "Q4-STRESS-CAUSAL-CHAIN", "text": "Trace the causal chain from the economic motivation for 'off-target enzymatic activity' risk mitigation in the original Gen-Synth document to the specific hardware vulnerability exploited in the 'Aethelred Breach' from the Cyber-Sec document. How does this chain illustrate a broader principle of 'economically-driven security trade-offs'?", "doc_ids": ["GEN-SYNTH-2024-ENZYME", "CYBER-SEC-2024-SLCI"]}
"""
        with open(QUERIES_FILE, "w", encoding='utf-8') as f: f.write(queries_content.strip())
    
    try:
        config = PipelineConfig()
        pipeline = KeystoneRAGController(config)
        pipeline.setup(doc_file=DOCS_FILE, query_file=QUERIES_FILE, force_reingest=args.reingest)
        pipeline.run_interactive_session()
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()
```
  
</details>

Both versions will deliver perfect scores on all queries, but the APEX implementation is much more elegant

The tested `docs.jsonl` & `queries.jsonl` are all in this repo for you to check out 

<details>
<summary>KEYSTONE log</summary>

```
================================================================================
🚀 V25.3 - Keystone Architecture (Final, Polished Build)
================================================================================
Enable LLM-as-a-Judge evaluation? (yes/no): yes
2025-10-07 17:04:32,759 - __main__ - INFO - LLM-as-a-Judge ENABLED: True

[1] Pre-loaded Queries, [2] Interactive, [quit]: 1

============================================================
Processing Query: Q1-STRESS-FACTUAL - What is a 'decoherence cascade' as described in the Project Chimera document, and what is the specific AI-driven methodology Dr. Eva Rostova's team uses to mitigate it?
============================================================
2025-10-07 17:04:33,526 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'Eva Rostova', 'AI']
2025-10-07 17:04:38,477 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:04:39,108 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:04:51,504 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
A 'decoherence cascade' is a phenomenon where the loss of quantum state in a single qubit can trigger a chain reaction, corrupting the entanglement across the entire Quantum Processing Unit (QPU). [7] To address this issue, Dr. Eva Rostova's team has developed an AI-driven, real-time pulse-level control system designed to mitigate such decoherence cascades. [6]

--- Citations ---
[1] The core technology leverages variational quantum eigensolvers (VQE) to navigate the colossal state spaces inherent in training large-scale foundation models.
[2] This AI constantly monitors qubit frequencies and recalibrates microwave control pulses on a nanosecond timescale to preemptively stabilize the quantum state, essentially 'learning' the unique noise signature of the QPU.
[3] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[4] This renders the results of the computation useless.
[5] The system has successfully extended the effective computation time by over 400%, but remains vulnerable to highly specific hardware-level exploits, a concern that has grown significantly since the infamous 'Aethelred Breach' demonstrated the viability of such attacks against specialized silicon.       
[6] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[7] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 18.11s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the evidence. The definition of 'decoherence cascade' is supported by EVIDENCE_7, and the description of the AI-driven methodology is supported by EVIDENCE_6. The answer does not introduce any unsupported claims or deviate from the provided evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by providing a precise definition of 'decoherence cascade' and detailing the specific AI-driven methodology used to mitigate it. All information is directly relevant to the question asked.

============================================================
Processing Query: Q2-STRESS-SYNTHESIS - Synthesize the core operational challenge described in 'Project Chimera' (decoherence cascades) with the one in 'Gen-Synth' (off-target enzymatic activity). What abstract principle of 'high-dimensional system control' do both challenges fundamentally represent?       
============================================================
2025-10-07 17:04:51,815 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'Gen Synth']
2025-10-07 17:04:52,436 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-07 17:04:53,345 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:04:54,231 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:03,187 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
Both 'Project Chimera' and 'Gen-Synth' grapple with fundamental challenges in high-dimensional system control. [5] In 'Project Chimera', the specific operational hurdle is 'decoherence cascades,' which occurs when a single qubit's loss of quantum state can corrupt the entire quantum processing unit. [3] Similarly, 'Gen-Synth' faces the challenge of 'off-target enzymatic activity,' where created enzymes interact with unintended molecules, presenting environmental risks. [2] At their core, these challenges highlight the difficulty in maintaining precise control over complex, high-dimensional systems, whether in quantum computing or synthetic biology. [3]

--- Citations ---
[1] The core technology leverages variational quantum eigensolvers (VQE) to navigate the colossal state spaces inherent in training large-scale foundation models.
[2] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[3] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.
[4] The field of synthetic biology has been revolutionized by Gen-Synth Corporation's 'differentiable biology' platform, which uses generative AI to design novel proteins and enzymes with bespoke functions.
[5] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[6] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[7] This renders the results of the computation useless.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 11.67s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately describes the challenges and synthesizes them under a common principle without introducing unsupported information.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by synthesizing the core operational challenges of both projects and identifying the abstract principle they represent. It stays on topic and provides a clear, relevant response.
============================================================
Processing Query: Q3-STRESS-MULTIHOP - Based on all documents, what is the plausible economic motive behind the 'Aethelred Breach,' and how does the specific cyberattack vector used (SLCI) create a direct, tangible link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by the DALTA legislation?
============================================================
2025-10-07 17:05:03,297 - __main__ - INFO - Final extracted subjects: ['Aethelred Breach', 'SLCI', 'DALTA', 'Zurich Quantum Institute']
2025-10-07 17:05:04,110 - __main__ - INFO - Auditor module activated: Analyzing evidence for conceptual gaps...
2025-10-07 17:05:05,785 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:05,786 - __main__ - INFO - Auditor identified potential bridge concept: 'Substrate-Level Code Injection (SLCI)'
2025-10-07 17:05:05,866 - __main__ - INFO - Apex Auditor added 3 new sentences to context.
2025-10-07 17:05:07,482 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:07,483 - __main__ - INFO - Activating Logical Weaver for multi-hop synthesis guidance...
2025-10-07 17:05:11,561 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:13,078 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:28,069 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
The plausible economic motive behind the 'Aethelred Breach' is to exploit vulnerabilities in financial protocols for financial gain, as evidenced by the manipulation of a price oracle using a flash loan mechanism. [7] The specific cyberattack vector SLCI, which involves embedding malicious microcode directly into the firmware of specialized hardware components during manufacturing, creates a direct, tangible link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by the DALTA legislation. [2] This link is established through the shared vulnerability to advanced, targeted attacks that can manipulate both financial markets and quantum computations. [1] Specifically, SLCI poses a direct threat to national strategic initiatives like Project Chimera, particularly in the context of quantum computations, as it could be used to subtly sabotage quantum computations without detection. [3] Given that the Zurich Quantum Institute's Project Chimera integrates generative AI models with quantum processing units (QPUs), making it a target for such advanced attacks, the operational risks posed by SLCI align with the regulatory concerns addressed by DALTA, which aims to enhance transparency and accountability in digital asset markets. [8]

--- Citations ---
[1] Regulators noted that the attack's precision and adaptive nature were indicative of a highly advanced AI, capable of predicting and exploiting the protocol's automated responses in real-time.
[2] Instead of exploiting application vulnerabilities, SLCI involves malicious microcode being embedded directly into the firmware of specialized hardware components during manufacturing, such as GPUs, TPUs, and even experimental Quantum Processing Units (QPUs).
[3] The report explicitly warns that the QPU-specific variants of SLCI being developed by state-level actors pose a direct and potentially catastrophic threat to national strategic initiatives like Project Chimera, as they could be used to subtly sabotage quantum computations and invalidate years of research without detection.
[4] The United States' 'Quantum Nexus Initiative' (QNI), established under the 2024 National Security AI Act, allocates $150 billion over five years to create a sovereign semiconductor supply chain and develop fault-tolerant quantum computers.
[5] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.
[6] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).
[7] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[8] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[9] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 24.88s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer faithfully follows the provided reasoning chain, accurately mapping each claim to the relevant evidence. The logic connecting the economic motive, the SLCI attack vector, and the link to regulatory concerns is fully supported by the evidence and aligns perfectly with the pipeline reasoning chain.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly explaining the economic motive behind the 'Aethelred Breach' and the connection between SLCI, the operational risks of the Zurich Quantum Institute, and the DALTA legislation. It is comprehensive and stays entirely on-topic.
============================================================
Processing Query: Q4-STRESS-NUANCE - Distinguish between the concept of 'systemic risk' as addressed by DALTA and the 'environmental risk' posed by Gen-Synth's platform. How are both of these risks examples of AI-driven 'unintended consequences' that traditional risk models might fail to predict?
============================================================
2025-10-07 17:05:28,218 - __main__ - INFO - Final extracted subjects: ['DALTA', 'AI', 'Gen Synth']
2025-10-07 17:05:28,660 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-07 17:05:29,037 - __main__ - INFO - Auditor module activated: Analyzing evidence for conceptual gaps...
2025-10-07 17:05:30,424 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:30,428 - __main__ - INFO - Auditor identified potential bridge concept: 'AI-driven unintended consequences'
2025-10-07 17:05:30,634 - __main__ - INFO - Apex Auditor added 3 new sentences to context.
2025-10-07 17:05:32,644 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:32,645 - __main__ - INFO - Activating Logical Weaver for multi-hop synthesis guidance...
2025-10-07 17:05:36,239 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:37,699 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:53,102 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
Both systemic risk addressed by DALTA and environmental risk posed by Gen-Synth's platform exemplify AI-driven unintended consequences that traditional risk models might fail to predict. [8] DALTA focuses on operational vulnerabilities in automated market makers (AMMs) and lending protocols, particularly addressing 'oracle manipulation' and cascading liquidations triggered by AI-driven high-frequency trading algorithms. [7] This highlights how AI can introduce novel vulnerabilities that are not easily captured by traditional risk models. [9] Similarly, Gen-Synth's platform involves 'off-target enzymatic activity,' where created enzymes inadvertently interact with and break down unintended but structurally similar molecules, posing significant environmental risks. [5] Traditional risk models may not anticipate the complex interactions between synthetic enzymes and natural environments, especially when these interactions are subtle and not easily predictable. [5] Both risks arise from the unique capabilities and behaviors of AI systems, such as their ability to manipulate data, adapt to new conditions, and perform complex tasks in unpredictable ways. [7] Thus, both systemic and environmental risks underscore the need for more sophisticated and adaptable risk models to address the complexities introduced by AI. [7]

--- Citations ---
[1] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[2] The report explicitly warns that the QPU-specific variants of SLCI being developed by state-level actors pose a direct and potentially catastrophic threat to national strategic initiatives like Project Chimera, as they could be used to subtly sabotage quantum computations and invalidate years of research without detection.
[3] A major setback for Western cybersecurity efforts was the 'SolarWinds II' incident of late 2023, where a state-sponsored threat actor exploited a zero-day vulnerability in a widely used infrastructure-as-code (IaC) provider, demonstrating that even with advanced AI-driven threat detection, supply chain vulnerabilities remain the Achilles' heel.
[4] Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation.
[5] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[6] This has raised concerns that nation-states could weaponize such techniques to destabilize financial markets.
[7] The legislation targets the operational vulnerabilities of automated market makers (AMMs) and lending protocols, particularly their susceptibility to 'oracle manipulation' and cascading liquidations triggered by AI-driven high-frequency trading algorithms.
[8] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).
[9] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 25.02s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer faithfully follows the provided reasoning chain, accurately mapping each claim to the relevant evidence. It correctly distinguishes between systemic risk addressed by DALTA and environmental risk posed by Gen-Synth's platform, and it effectively links both to AI-driven unintended consequences. All claims are fully supported by the evidence, and the answer adheres strictly to the reasoning chain.  
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly distinguishing between the two types of risks and explaining how both are examples of AI-driven unintended consequences that traditional risk models might fail to predict. It provides specific examples and links them back to the broader context of AI's impact, making it highly relevant to the question.
============================================================
Processing Query: Q1-STRESS-DEEP-FACTUAL - Describe the specific mechanism of 'synaptic interference' in Project Mnemosyne, and detail the architecture and function of the 'Predictive Hebbian Modulator' AI used to counteract it.
============================================================
2025-10-07 17:05:53,222 - __main__ - INFO - Final extracted subjects: ['AI', 'Hebbian Mo', 'Project Mnemosyne']
2025-10-07 17:05:54,714 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:05:55,810 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:06,875 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
In Project Mnemosyne, synaptic interference is caused by an anomalous, high-energy particle flux that amplifies light's effect beyond its intended target. [3] To counteract this, the Predictive Hebbian Modulator AI is employed. [2] This AI is a sophisticated recurrent neural network equipped with a temporal-convolutional attention mechanism that learns unique synaptic potentiation patterns of an individual's brain. [5] By pre-emptively adjusting the implant's light frequency and intensity, the Predictive Hebbian Modulator creates a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects. [7] Early results of the project demonstrate an impressive 98% recall accuracy for encoded information, highlighting the effectiveness of this approach. [6]

--- Citations ---
[1] The system uses a novel 'neuro-photonic' implant that translates digital data into precisely targeted light patterns to stimulate and modify hippocampal engrams.
[2] To counter this, the team developed a sophisticated AI called the 'Predictive Hebbian Modulator.'
[3] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[4] Project Mnemosyne, a DARPA initiative headquartered at MIT's Media Lab, is developing a next-generation Brain-Computer Interface (BCI) focused on direct memory encoding and retrieval.
[5] This is a recurrent neural network with a temporal-convolutional attention mechanism that learns the unique synaptic potentiation patterns of an individual's brain.
[6] While early results have shown an unprecedented 98% recall accuracy for encoded information, the primary operational risk is 'synaptic interference.' 
[7] It then pre-emptively adjusts the implant's light frequency and intensity to create a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 13.77s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple evidence IDs to form a coherent explanation of synaptic interference and the Predictive Hebbian Modulator's function. The citations at the end also correctly reference all relevant evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by detailing both the mechanism of synaptic interference and the architecture/function of the Predictive Hebbian Modulator. It covers all requested aspects without straying from the topic.
============================================================
Processing Query: Q2-STRESS-ABSTRACT-SYNTHESIS - Synthesize the core challenges of Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion). What single, unifying principle of 'long-term material and informational integrity' do both projects struggle with, and how do their respective AI-driven solutions represent a similar philosophical approach to predictive maintenance?
============================================================
2025-10-07 17:06:07,004 - __main__ - INFO - Final extracted subjects: ['AI', 'Project Helios', 'En', 'tanglement Fraying', 'Lattice Decohesion', 'Project Erebus']
2025-10-07 17:06:09,090 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:10,475 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:20,334 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
Both Project Erebus and Project Helios face significant challenges in maintaining long-term material and informational integrity. [2] Project Erebus tackles the issue of 'Entanglement Fraying,' which requires preserving the stability of quantum entanglement over extended periods, while Project Helios addresses 'Lattice Decohesion,' a material science problem involving the brittleness and fracture of the sail's crystal structure due to prolonged exposure to the solar wind and interstellar medium. [2] To overcome these challenges, both projects utilize AI-driven solutions. [3] Project Erebus employs an AI error-correction model running on a classical co-processor to predict and mitigate 'Entanglement Fraying,' whereas Project Helios uses an AI that analyzes real-time parity-check measurements from a quantum processing unit (QPU) to forecast the onset of 'Lattice Decohesion.' [3] Despite the different nature of their challenges, both projects share a similar philosophical approach to predictive maintenance, leveraging AI to monitor and predict potential failures before they occur, thus ensuring the integrity of their respective systems. [6]

--- Citations ---
[1] The project's quantum processing unit (QPU), a 4,096-qubit topological device, is designed to solve complex quantum chromodynamics equations that are intractable for classical supercomputers.
[2] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[3] The Erebus team's solution is an AI error-correction model that runs on a classical co-processor.
[4] Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation.
[5] Analysis has revealed that this accelerated decay is strongly correlated with the same anomalous 'exotic particle flux' documented by other advanced research projects.
[6] The AI analyzes the parity-check measurements from the QPU in real-time and predicts the onset of a fraying event.
[7] At Lawrence Livermore National Laboratory, Project Erebus is a major Department of Energy program aimed at using quantum computers to simulate the behavior of dark matter.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 13.46s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer accurately synthesizes the core challenges of both projects ('Entanglement Fraying' for Erebus and 'Lattice Decohesion' for Helios) as described in EVIDENCE_2 and EVIDENCE_7. It correctly identifies the unifying principle of maintaining long-term material and informational integrity, which is implied by the challenges faced by both projects. The answer also accurately describes the AI-driven solutions for both projects, referencing EVIDENCE_3 and EVIDENCE_6. The philosophical approach to predictive maintenance is logically derived from the evidence provided. All claims are fully supported by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The generated answer directly addresses the question by synthesizing the core challenges of both projects, identifying the unifying principle, and explaining the similar philosophical approach to predictive maintenance. The answer is comprehensive and stays entirely on topic, making it perfectly relevant to the question asked.
============================================================
Processing Query: Q3-STRESS-GRAND-MULTIHOP - According to the Orion Mandate, what is 'asymmetric temporal sabotage,' and what is the specific, underlying physical phenomenon (not mentioned in the Mandate itself) that mechanistically links the operational risks of all three critical projects: Mnemosyne, Erebus, and Helios?
============================================================
2025-10-07 17:06:20,471 - __main__ - INFO - Final extracted subjects: ['Erebus', 'Orion Mandate', 'Mandate', 'Helios', 'Mnemosyne']
2025-10-07 17:06:21,196 - __main__ - INFO - Auditor module activated: Analyzing evidence for conceptual gaps...
2025-10-07 17:06:22,939 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:22,940 - __main__ - INFO - Auditor identified potential bridge concept: 'Quantum Entanglement'
2025-10-07 17:06:23,123 - __main__ - INFO - Apex Auditor added 3 new sentences to context.
2025-10-07 17:06:24,094 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:24,095 - __main__ - INFO - Activating Logical Weaver for multi-hop synthesis guidance...
2025-10-07 17:06:28,220 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:29,640 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:44,930 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
According to the Orion Mandate, asymmetric temporal sabotage is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation. [2] The specific, underlying physical phenomenon that mechanistically links the operational risks of all three critical projects—Mnemosyne, Erebus, and Helios—is an anomalous, high-energy particle flux that amplifies light beyond its intended target. [3] This particle flux can interact with the light-based technologies used in Mnemosyne (likely involving advanced memory storage and retrieval systems), Erebus (potentially using light amplification techniques for dark matter or dark energy research), and Helios (possibly related to solar energy or heliostats involving light manipulation and amplification). [6] The interaction of this particle flux with light can introduce subtle flaws into these systems, leading to long-term failures that compromise the operational integrity of each project. [2]

--- Citations ---
[1] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[2] This is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation.
[3] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[4] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[5] The United States' 'Quantum Nexus Initiative' (QNI), established under the 2024 National Security AI Act, allocates $150 billion over five years to create a sovereign semiconductor supply chain and develop fault-tolerant quantum computers.
[6] The document explicitly names three critical, interdependent pillars for achieving this goal: Project Erebus (for computational dominance), Project Mnemosyne (for neural interface superiority), and Project Helios (for demonstrating advanced materials and energy projection).
[7] The project's quantum processing unit (QPU), a 4,096-qubit topological device, is designed to solve complex quantum chromodynamics equations that are intractable for classical supercomputers.
[8] The mandate's primary concern is a new strategic threat termed 'asymmetric temporal sabotage.'
[9] It then instructs the QPU's control system to perform a series of 'entanglement distillation' protocols, sacrificing some qubits to reinforce the stability of the remaining computational set.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 24.60s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer faithfully follows the provided reasoning chain. Each claim is directly supported by the evidence, and the logical synthesis of multiple evidence IDs aligns perfectly with the pipeline reasoning chain. The answer accurately defines asymmetric temporal sabotage and correctly identifies the anomalous, high-energy particle flux as the underlying physical phenomenon linking the operational risks of the three projects. The speculative connections to each project (Mnemosyne, Erebus, Helios) are reasonable inferences based on the evidence and the reasoning chain.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by defining asymmetric temporal sabotage and identifying the specific physical phenomenon that links the operational risks of the three critical projects. The response is comprehensive, relevant, and stays entirely on-topic, providing a clear and detailed explanation that matches the question's requirements.
============================================================
Processing Query: Q4-STRESS-CAUSAL-CHAIN - Trace the causal chain from the economic motivation for 'off-target enzymatic activity' risk mitigation in the original Gen-Synth document to the specific hardware vulnerability exploited in the 'Aethelred Breach' from the Cyber-Sec document. How does this chain illustrate a broader principle of 'economically-driven security trade-offs'?
============================================================
2025-10-07 17:06:45,062 - __main__ - INFO - Final extracted subjects: ['Aethelred Breach', 'Cy', 'Se', 'ber', 'Gen Synth']
2025-10-07 17:06:45,408 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Cy', 'ber', 'Gen Synth']
2025-10-07 17:06:46,786 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:06:47,987 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:01,735 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
The economic viability of custom enzyme synthesis is constrained by significant computational demands, prompting companies like Gen-Synth to explore innovative financing models to manage high operational costs. [2] However, this economic motivation also creates a risk landscape where off-target enzymatic activity can be exploited for financial gain, such as manipulating price oracle calculations. [2] This risk environment has given rise to a new class of advanced persistent threats, exemplified by the 'Aethelred Breach.' [4] This breach leverages sophisticated supply-chain attacks, utilizing previously dormant code to cause subtle hardware malfunctions, leading to a cascade of liquidations for profit. [7] Thus, the causal chain from the economic drive to mitigate off-target enzymatic activity risk to the exploitation of hardware vulnerabilities in the 'Aethelred Breach' illustrates the broader principle of 'economically-driven security trade-offs,' where efforts to reduce one risk can inadvertently create new, more complex vulnerabilities. [3]

--- Citations ---
[1] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.
[2] This immense computational demand has made the commercial viability of custom enzyme synthesis a major economic hurdle, pushing companies like Gen-Synth to explore unconventional financing models to cover the massive operational expenditures.
[3] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[4] A new class of advanced persistent threat (APT) has emerged, as detailed in a recent NSA report on the 'Aethelred Breach.'
[5] The design process involves a deep learning model that predicts a protein's final 3D structure from its amino acid sequence and, crucially, simulates its allosteric modulation—how its function changes when binding to target molecules.
[6] The platform's flagship achievement is the creation of a synthetic enzyme, GS-411, capable of catabolizing industrial plastics at room temperature.   
[7] This dormant code can be activated remotely to create subtle but critical hardware malfunctions.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 16.80s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer logically synthesizes multiple pieces of evidence to construct a coherent causal chain, adhering closely to the question's requirements.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by tracing the causal chain from economic motivations to the specific hardware vulnerability exploited in the 'Aethelred Breach.' It also effectively illustrates the broader principle of 'economically-driven security trade-offs,' making it highly relevant to the question.
============================================================
Processing Query: Q1-FACTUAL-SYNTHESIS - What is the primary stability issue with perovskite solar cells and what specific material science innovation was developed at Berkeley's Helios Lab to address it?
============================================================
2025-10-07 17:07:01,847 - __main__ - INFO - Final extracted subjects: ['Berkeley', 'Helios Lab']
2025-10-07 17:07:02,809 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:03,569 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:12,810 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
The primary stability issue with perovskite solar cells is their rapid degradation when exposed to humidity and oxygen. [4] This challenge has been addressed by a 2024 innovation from Berkeley's Helios Lab, which developed a self-assembling, 2D passivation layer composed of diamine-based organic molecules. [3] This passivation layer functions as a hydrophobic barrier, effectively protecting the delicate perovskite crystal structure from environmental factors such as moisture and oxygen, thereby enhancing the stability of perovskite solar cells. [2]

--- Citations ---
[1] In accelerated aging tests, the passivated cells retained 95% of their initial efficiency after 1,000 hours under 85% humidity and continuous illumination, a 50-fold improvement over unprotected cells.
[2] This atomically thin layer acts as a hydrophobic barrier, protecting the delicate perovskite crystal structure.
[3] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[4] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[5] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.
[6] Their proprietary MOF, CF-8b, has an exceptionally high affinity for CO2, capable of capturing it from the atmosphere at concentrations of 400 ppm and achieving a 98% capture rate.
[7] The fight against climate change is increasingly a battle of materials science.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 11.07s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The primary stability issue is corroborated by EVIDENCE_4, the innovation from Berkeley's Helios Lab is detailed in EVIDENCE_3, the function of the passivation layer as a hydrophobic barrier is supported by EVIDENCE_2, and the effectiveness of the passivation layer is demonstrated by EVIDENCE_1. The answer does not introduce any unsupported claims and faithfully follows the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The generated answer directly addresses the question by clearly stating the primary stability issue with perovskite solar cells and describing the specific material science innovation developed to address it. All information provided is relevant to the question, and no extraneous details are included. The answer is concise and on-topic.
============================================================
Processing Query: Q2-ANALOGICAL-REASONING - How might the 'neural signal drift' problem in NeuroGenix's BCI research be conceptually analogous to the primary challenge facing the Quantum Nexus Initiative's systems?
============================================================
2025-10-07 17:07:12,933 - __main__ - INFO - Final extracted subjects: ['Quantum Nexus Initiative', 'BCI', 'NeuroGenix']
2025-10-07 17:07:14,478 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:15,486 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:24,514 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
The 'neural signal drift' problem in NeuroGenix's Brain-Computer Interface (BCI) research, where the brain's representation of a motor task changes over time, leading to performance degradation, can be conceptually analogous to the primary challenge facing the Quantum Nexus Initiative's (QNI) systems. [2] Both face issues of stability and adaptability against evolving threats. [5] While NeuroGenix addresses this through an AI model that predicts and adapts to neural signal drift in real-time, the QNI must ensure the stability and security of its entire hardware and software stack against sophisticated AI-generated attacks that manipulate system behavior without triggering conventional alarms. [5] Both initiatives require adaptive mechanisms to maintain performance and security in the face of dynamic and evolving challenges. [5]

--- Citations ---
[1] This initiative is a direct response to China's 'Project Dragon's Ascent,' a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology.
[2] This drift, a long-standing obstacle where the brain's representation of a motor task changes over days or weeks, is a primary cause of BCI performance degradation.
[3] The Zurich-based research institute NeuroGenix has recently published phase II clinical trial results for its 'Arbor' BCI system.
[4] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[5] The primary challenge for the QNI is not just raw processing power, but ensuring the stability and security of the entire hardware and software stack against incredibly subtle, AI-generated attacks that can manipulate system behavior without triggering conventional alarms.
[6] Despite this success, the primary technical challenge remains long-term biocompatibility.
[7] The geopolitical landscape of the 21st century is being redrawn by the race for artificial general intelligence (AGI).


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 11.70s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The description of the 'neural signal drift' problem is supported by EVIDENCE_2 and EVIDENCE_4. The primary challenge for the QNI is accurately described using EVIDENCE_5. The conceptual analogy between the two challenges is logically synthesized from these pieces of evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by drawing a clear and relevant conceptual analogy between the 'neural signal drift' problem in BCI research and the primary challenge facing the QNI's systems. It stays on topic and provides a comprehensive comparison supported by the evidence.
============================================================
Processing Query: Q3-MULTIHOP-ABSTRACTION - According to the 2025 Global Risk Report, what is 'asymmetric technological leverage', and how could this concept connect the economic challenges of CarbonForge's direct air capture technology with the strategic goals of China's 'Project Dragon's Ascent'?
============================================================
2025-10-07 17:07:24,635 - __main__ - INFO - Final extracted subjects: ['Global Risk Report', 'China', 'CarbonForge', '2025', 'Project Dragon Ascent']
2025-10-07 17:07:25,063 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Project Dragon Ascent']
2025-10-07 17:07:25,335 - __main__ - INFO - Auditor module activated: Analyzing evidence for conceptual gaps...
2025-10-07 17:07:27,135 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:27,136 - __main__ - INFO - Auditor identified potential bridge concept: 'supply chain vulnerability'
2025-10-07 17:07:27,716 - __main__ - INFO - Apex Auditor added 3 new sentences to context.
2025-10-07 17:07:29,681 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:29,683 - __main__ - INFO - Activating Logical Weaver for multi-hop synthesis guidance...
2025-10-07 17:07:33,396 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:34,794 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:49,558 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
Asymmetric technological leverage is defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain. [5] This concept is identified as the most significant emerging threat to global stability. [8] Direct air capture (DAC) of CO2 is seen as a critical-path technology, but CarbonForge's DAC technology faces economic challenges due to high costs, currently at approximately $550 per ton, which is far above the $100-$150 target needed for broad economic viability. [3] China's 'Project Dragon's Ascent' is a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology. [7] By dominating the global market for advanced semiconductor technology, China could create a technological dependency that allows it to exert disproportionate influence over CarbonForge's operations and broader climate change initiatives. [5] This connection underscores the strategic importance of technological dominance in shaping global technological and environmental policies. [5]

--- Citations ---
[1] For example, the report highlights the strategic risk of a single nation dominating the supply chain for critical green energy components, such as the catalysts and membranes used in green hydrogen production or the specialized polymers for next-generation solar cells.
[2] The United States' 'Quantum Nexus Initiative' (QNI), established under the 2024 National Security AI Act, allocates $150 billion over five years to create a sovereign semiconductor supply chain and develop fault-tolerant quantum computers.
[3] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.
[4] A major setback for Western cybersecurity efforts was the 'SolarWinds II' incident of late 2023, where a state-sponsored threat actor exploited a zero-day vulnerability in a widely used infrastructure-as-code (IaC) provider, demonstrating that even with advanced AI-driven threat detection, supply chain vulnerabilities remain the Achilles' heel.
[5] This concept is defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain.
[6] In a parallel field, direct air capture (DAC) of CO2 is seen as a critical-path technology.
[7] This initiative is a direct response to China's 'Project Dragon's Ascent,' a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology.
[8] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.
[9] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 25.05s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer faithfully follows the provided reasoning chain, accurately defining asymmetric technological leverage, detailing the economic challenges of CarbonForge's DAC technology, and explaining the strategic goals of China's 'Project Dragon's Ascent'. It logically connects these elements through the concept of technological dependency, fully supported by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by explaining 'asymmetric technological leverage' and its potential connection between CarbonForge's economic challenges and China's strategic goals. It is comprehensive and stays on topic throughout.
============================================================
Processing Query: Q4-COMPARATIVE-ANALYSIS - Compare and contrast the primary technical challenge and the proposed solution for the Arbor BCI and perovskite solar cells as described in the documents.
============================================================
2025-10-07 17:07:49,725 - __main__ - INFO - Final extracted subjects: ['Arbor BCI']
2025-10-07 17:07:50,192 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Arbor BCI']
2025-10-07 17:07:52,081 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:07:53,452 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:08:04,424 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Path):
The primary technical challenges for the Arbor Brain-Computer Interface (BCI) system include improving fine motor control restoration accuracy, adapting to real-time neural signal drift, and minimizing the immune response to the ECoG array. [1] To address these challenges, the proposed solution involves the use of a generative AI decoder to enhance the accuracy of intention-to-action translation, implementing an AI model that can predict and adapt to neural signal drift in real-time, and employing a high-density, flexible ECoG array to reduce immune response. [6] In contrast, the main technical hurdles for perovskite solar cells revolve around their instability and rapid degradation when exposed to humidity and oxygen, as well as the high economic and energy costs associated with commercial adoption. [5] The proposed solution to these issues focuses on developing a self-assembling, 2D passivation layer composed of diamine-based organic molecules to improve the stability of perovskite solar cells. [4]

--- Citations ---
[1] In a trial involving 12 patients with tetraplegia from spinal cord injuries, the Arbor system, paired with a generative AI decoder, restored fine motor control with 92% intention-to-action accuracy, allowing participants to control robotic limbs for tasks like writing and eating.
[2] The primary obstacle for DAC is not the chemistry but the economics and energy cost.
[3] Breakthroughs in brain-computer interfaces (BCIs) are rapidly moving from theoretical science to clinical reality.
[4] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[5] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[6] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[7] The system utilizes a high-density, flexible electrocorticography (ECoG) array of 2,048 microelectrodes that conforms to the brain's surface, minimizing immune response.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 14.85s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The Arbor BCI system's challenges and solutions are backed by EVIDENCE_1, EVIDENCE_6, and EVIDENCE_7. The perovskite solar cells' challenges and solutions are supported by EVIDENCE_5 and EVIDENCE_4. The answer accurately synthesizes the evidence without introducing unsupported claims.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by comparing and contrasting the primary technical challenges and proposed solutions for both the Arbor BCI system and perovskite solar cells. It stays on topic and provides a clear, structured response that aligns perfectly with the question's requirements.
```

</details>

<details>
<summary>APEX log</summary>

```
================================================================================
🚀APEX ARCHITECTURE
================================================================================
Enable LLM-as-a-Judge evaluation? (yes/no): yes
2025-10-07 17:15:19,963 - __main__ - INFO - LLM-as-a-Judge ENABLED: True

[1] Pre-loaded Queries, [2] Interactive, [quit]: 1

============================================================
Processing Query: Q1-STRESS-FACTUAL - What is a 'decoherence cascade' as described in the Project Chimera document, and what is the specific AI-driven methodology Dr. Eva Rostova's team uses to mitigate it?
============================================================
2025-10-07 17:15:21,114 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'Eva Rostova', 'AI']
2025-10-07 17:15:24,980 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:15:25,596 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
2025-10-07 17:15:38,707 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
A 'decoherence cascade' is a phenomenon where a single qubit's loss of quantum state can trigger a chain reaction, corrupting the entanglement across the entire Quantum Processing Unit (QPU). [7] To mitigate this issue, Dr. Eva Rostova's team employs an AI-driven, real-time pulse-level control system designed to address and prevent such cascading effects. [5]

--- Citations ---
[1] The core technology leverages variational quantum eigensolvers (VQE) to navigate the colossal state spaces inherent in training large-scale foundation models.
[2] This AI constantly monitors qubit frequencies and recalibrates microwave control pulses on a nanosecond timescale to preemptively stabilize the quantum state, essentially 'learning' the unique noise signature of the QPU.
[3] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[4] This renders the results of the computation useless.
[5] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[6] The system has successfully extended the effective computation time by over 400%, but remains vulnerable to highly specific hardware-level exploits, a concern that has grown significantly since the infamous 'Aethelred Breach' demonstrated the viability of such attacks against specialized silicon.       
[7] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 17.72s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The definition of 'decoherence cascade' is taken verbatim from EVIDENCE_7, and the mitigation strategy is directly cited from EVIDENCE_5. No extraneous or unsupported claims are made.   
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by providing both the definition of 'decoherence cascade' and the specific AI-driven methodology used to mitigate it. All information is directly relevant and concise.
============================================================
Processing Query: Q2-STRESS-SYNTHESIS - Synthesize the core operational challenge described in 'Project Chimera' (decoherence cascades) with the one in 'Gen-Synth' (off-target enzymatic activity). What abstract principle of 'high-dimensional system control' do both challenges fundamentally represent?       
============================================================
2025-10-07 17:15:38,799 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'Gen Synth']
2025-10-07 17:15:39,028 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-07 17:15:39,880 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:15:40,727 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:15:51,271 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Both 'Project Chimera' and 'Gen-Synth' face fundamental operational challenges rooted in the management of high-dimensional system control. [5] In 'Project Chimera,' the primary issue is 'decoherence cascades,' which can corrupt the entanglement across the entire Quantum Processing Unit (QPU) due to small errors. [3] Similarly, 'Gen-Synth' grapples with 'off-target enzymatic activity,' where created enzymes interact with unintended molecules, leading to unpredictable outcomes. [2] Both challenges highlight the difficulty in controlling interactions within high-dimensional systems, where minor errors can propagate and cause significant disruptions throughout the system. [3]

--- Citations ---
[1] The core technology leverages variational quantum eigensolvers (VQE) to navigate the colossal state spaces inherent in training large-scale foundation models.
[2] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[3] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.
[4] The field of synthetic biology has been revolutionized by Gen-Synth Corporation's 'differentiable biology' platform, which uses generative AI to design novel proteins and enzymes with bespoke functions.
[5] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[6] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[7] This renders the results of the computation useless.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 12.58s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately describes the challenges of 'decoherence cascades' and 'off-target enzymatic activity' and correctly synthesizes them under the principle of 'high-dimensional system control'. The reasoning is consistent with the evidence and the question's requirements.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by synthesizing the core operational challenges of both 'Project Chimera' and 'Gen-Synth' and identifying the abstract principle they represent. It stays entirely on-topic and provides a clear, relevant response.

============================================================
Processing Query: Q3-STRESS-MULTIHOP - Based on all documents, what is the plausible economic motive behind the 'Aethelred Breach,' and how does the specific cyberattack vector used (SLCI) create a direct, tangible link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by the DALTA legislation?
============================================================
2025-10-07 17:15:51,644 - __main__ - INFO - Final extracted subjects: ['Zurich Quantum Institute', 'Aethelred Breach', 'DALTA', 'SLCI']
2025-10-07 17:15:54,527 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:15:55,802 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:16,547 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The plausible economic motive behind the 'Aethelred Breach' was the exploitation of a flash loan mechanism to manipulate a price oracle, resulting in significant financial losses exceeding $2 billion. [4] This breach was executed using a sophisticated cyberattack vector known as SLCI, which is particularly dangerous for quantum computations as it can predict and exploit automated responses in real-time, potentially invalidating years of research without detection. [3] The specific use of SLCI creates a direct, tangible link between the operational risks faced by the Zurich Quantum Institute and the regulatory concerns addressed by the Digital Asset Liability & Transparency Act (DALTA). [2] This legislation mandates increased transparency and accountability in the handling of digital assets, while the vulnerability of quantum research to SLCI highlights the operational risks associated with quantum computations. [3] The concept of 'asymmetric technological leverage' further underscores how such attacks can create disproportionate influence, thereby linking operational risks to broader systemic risks and regulatory mandates. [7]

--- Citations ---
[1] Regulators noted that the attack's precision and adaptive nature were indicative of a highly advanced AI, capable of predicting and exploiting the protocol's automated responses in real-time.
[2] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).
[3] The report explicitly warns that the QPU-specific variants of SLCI being developed by state-level actors pose a direct and potentially catastrophic threat to national strategic initiatives like Project Chimera, as they could be used to subtly sabotage quantum computations and invalidate years of research without detection.
[4] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[5] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[6] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.
[7] This concept is defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 25.13s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple pieces of evidence to build a coherent argument, and there are no deviations or unsupported claims.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by explaining the economic motive behind the 'Aethelred Breach' and the link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by DALTA. It is comprehensive and stays on-topic throughout.
============================================================
Processing Query: Q4-STRESS-NUANCE - Distinguish between the concept of 'systemic risk' as addressed by DALTA and the 'environmental risk' posed by Gen-Synth's platform. How are both of these risks examples of AI-driven 'unintended consequences' that traditional risk models might fail to predict?
============================================================
2025-10-07 17:16:16,697 - __main__ - INFO - Final extracted subjects: ['DALTA', 'AI', 'Gen Synth']
2025-10-07 17:16:17,308 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-07 17:16:19,526 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:20,854 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:32,623 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Systemic risk, as addressed by DALTA, pertains to the operational vulnerabilities in automated market makers (AMMs) and lending protocols, particularly focusing on the susceptibility to "oracle manipulation" and cascading liquidations triggered by AI-driven high-frequency trading algorithms. [7] One of the sophisticated attack vectors, "Substrate-Level Code Injection" (SLCI), exemplifies a supply-chain attack that bypasses traditional software-based security measures, potentially weaponized by nation-states to destabilize financial markets. [5] Conversely, environmental risk posed by Gen-Synth's platform involves the use of generative AI to design novel proteins and enzymes with bespoke functions, but the primary operational challenge is "off-target enzymatic activity," where the created enzyme may inadvertently interact with and break down unintended but structurally similar molecules. [4] Both risks represent AI-driven 'unintended consequences' that traditional risk models might fail to predict, as they highlight the unique and complex challenges posed by advanced AI technologies in financial and biological domains. [7]

--- Citations ---
[1] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[2] The field of synthetic biology has been revolutionized by Gen-Synth Corporation's 'differentiable biology' platform, which uses generative AI to design novel proteins and enzymes with bespoke functions.
[3] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).
[4] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[5] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.
[6] This has raised concerns that nation-states could weaponize such techniques to destabilize financial markets.
[7] The legislation targets the operational vulnerabilities of automated market makers (AMMs) and lending protocols, particularly their susceptibility to 'oracle manipulation' and cascading liquidations triggered by AI-driven high-frequency trading algorithms.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 16.07s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. Systemic risk details are backed by EVIDENCE_3, EVIDENCE_7, and EVIDENCE_5. Environmental risk details are supported by EVIDENCE_2 and EVIDENCE_4. The conclusion about unintended consequences is logically synthesized from all relevant evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly distinguishing between systemic and environmental risks, and explaining how both are examples of AI-driven unintended consequences. It covers all aspects of the question thoroughly and stays on topic.
============================================================
Processing Query: Q1-STRESS-DEEP-FACTUAL - Describe the specific mechanism of 'synaptic interference' in Project Mnemosyne, and detail the architecture and function of the 'Predictive Hebbian Modulator' AI used to counteract it.
============================================================
2025-10-07 17:16:32,797 - __main__ - INFO - Final extracted subjects: ['Project Mnemosyne', 'Hebbian Mo', 'AI']
2025-10-07 17:16:34,272 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:35,159 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:45,520 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
In Project Mnemosyne, synaptic interference arises from an anomalous high-energy particle flux that amplifies light's effect beyond its intended target. [3] To counteract this, the 'Predictive Hebbian Modulator' AI is employed. [2] This AI is a sophisticated recurrent neural network equipped with a temporal-convolutional attention mechanism, designed to learn the unique synaptic potentiation patterns of each individual's brain. [5] By preemptively adjusting the implant's light frequency and intensity, the Predictive Hebbian Modulator creates a 'counter-resonance,' effectively shielding nearby memories from the disruptive effects of the anomalous particles. [7]

--- Citations ---
[1] The system uses a novel 'neuro-photonic' implant that translates digital data into precisely targeted light patterns to stimulate and modify hippocampal engrams.
[2] To counter this, the team developed a sophisticated AI called the 'Predictive Hebbian Modulator.'
[3] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[4] Project Mnemosyne, a DARPA initiative headquartered at MIT's Media Lab, is developing a next-generation Brain-Computer Interface (BCI) focused on direct memory encoding and retrieval.
[5] This is a recurrent neural network with a temporal-convolutional attention mechanism that learns the unique synaptic potentiation patterns of an individual's brain.
[6] While early results have shown an unprecedented 98% recall accuracy for encoded information, the primary operational risk is 'synaptic interference.' 
[7] It then pre-emptively adjusts the implant's light frequency and intensity to create a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 12.84s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: All claims in the generated answer are fully supported by the evidence. Claim 1 is directly supported by EVIDENCE_3 and EVIDENCE_6. Claim 2 is supported by EVIDENCE_2, EVIDENCE_5, and EVIDENCE_7. The answer also correctly synthesizes multiple evidence IDs to form a coherent explanation.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by detailing both the mechanism of synaptic interference and the architecture/function of the Predictive Hebbian Modulator. It covers all aspects of the question without straying from the topic.
============================================================
Processing Query: Q2-STRESS-ABSTRACT-SYNTHESIS - Synthesize the core challenges of Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion). What single, unifying principle of 'long-term material and informational integrity' do both projects struggle with, and how do their respective AI-driven solutions represent a similar philosophical approach to predictive maintenance?
============================================================
2025-10-07 17:16:45,625 - __main__ - INFO - Final extracted subjects: ['Project Erebus', 'AI', 'Project Helios', 'Lattice Decohesion', 'En', 'tanglement Fraying']
2025-10-07 17:16:47,853 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:49,162 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:16:59,449 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Both Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion) face the core challenge of maintaining long-term material and informational integrity. [2] Project Erebus tackles 'Entanglement Fraying,' which is the degradation of entangled states over time due to decoherence and external manipulation, while Project Helios addresses 'Lattice Decohesion,' a material science issue where the sail's crystal structure becomes brittle and fractures under prolonged exposure to the solar wind and interstellar medium. [2] To combat these issues, both projects utilize AI-driven solutions: Erebus employs an AI error-correction model running on a classical co-processor to predict and mitigate entanglement fraying, whereas Helios uses an AI that analyzes real-time parity-check measurements from a quantum processing unit to forecast the onset of lattice decohesion. [3] This approach reflects a shared philosophical commitment to predictive maintenance, ensuring the long-term stability and functionality of their systems through proactive monitoring and intervention. [6]

--- Citations ---
[1] The project's quantum processing unit (QPU), a 4,096-qubit topological device, is designed to solve complex quantum chromodynamics equations that are intractable for classical supercomputers.
[2] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[3] The Erebus team's solution is an AI error-correction model that runs on a classical co-processor.
[4] Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation.
[5] Analysis has revealed that this accelerated decay is strongly correlated with the same anomalous 'exotic particle flux' documented by other advanced research projects.
[6] The AI analyzes the parity-check measurements from the QPU in real-time and predicts the onset of a fraying event.
[7] At Lawrence Livermore National Laboratory, Project Erebus is a major Department of Energy program aimed at using quantum computers to simulate the behavior of dark matter.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 13.93s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The description of 'Lattice Decohesion' is supported by EVIDENCE_2, the AI error-correction model for Erebus is supported by EVIDENCE_3 and EVIDENCE_6, and the mention of the QPU is supported by EVIDENCE_1. The answer logically synthesizes these pieces of evidence without deviation.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by synthesizing the core challenges of both projects, identifying a unifying principle, and explaining their shared philosophical approach to predictive maintenance. It is comprehensive and on-topic.
============================================================
Processing Query: Q3-STRESS-GRAND-MULTIHOP - According to the Orion Mandate, what is 'asymmetric temporal sabotage,' and what is the specific, underlying physical phenomenon (not mentioned in the Mandate itself) that mechanistically links the operational risks of all three critical projects: Mnemosyne, Erebus, and Helios?
============================================================
2025-10-07 17:16:59,565 - __main__ - INFO - Final extracted subjects: ['Erebus', 'Mandate', 'Orion Mandate', 'Mnemosyne', 'Helios']
2025-10-07 17:17:01,304 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:02,141 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:11,941 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
According to the Orion Mandate, 'asymmetric temporal sabotage' refers to an adversary's strategy of introducing subtle, nearly undetectable flaws into long-term, high-cost research projects, which can cause them to fail years or even decades later, thus neutralizing a nation's technological advantage without engaging in direct confrontation. [6] This sabotage is mechanistically linked to the operational risks of the three critical projects—Mnemosyne, Erebus, and Helios—by an anomalous, high-energy particle flux that amplifies the light's effect beyond its intended target. [2]

--- Citations ---
[1] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[2] The document explicitly names three critical, interdependent pillars for achieving this goal: Project Erebus (for computational dominance), Project Mnemosyne (for neural interface superiority), and Project Helios (for demonstrating advanced materials and energy projection).
[3] The mandate's primary concern is a new strategic threat termed 'asymmetric temporal sabotage.'
[4] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[5] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[6] This is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation.
[7] It then instructs the QPU's control system to perform a series of 'entanglement distillation' protocols, sacrificing some qubits to reinforce the stability of the remaining computational set.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 12.49s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer accurately reflects the definition of 'asymmetric temporal sabotage' from EVIDENCE_6 and correctly identifies the linking physical phenomenon (anomalous, high-energy particle flux) from EVIDENCE_4. It does not include irrelevant information from other evidence, maintaining strict adherence to the question's requirements.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by defining 'asymmetric temporal sabotage' and identifying the specific physical phenomenon that links the operational risks of Projects Mnemosyne, Erebus, and Helios. It stays entirely on-topic and provides a complete response to the question.

============================================================
Processing Query: Q4-STRESS-CAUSAL-CHAIN - Trace the causal chain from the economic motivation for 'off-target enzymatic activity' risk mitigation in the original Gen-Synth document to the specific hardware vulnerability exploited in the 'Aethelred Breach' from the Cyber-Sec document. How does this chain illustrate a broader principle of 'economically-driven security trade-offs'?
============================================================
2025-10-07 17:17:12,050 - __main__ - INFO - Final extracted subjects: ['Se', 'Cy', 'Aethelred Breach', 'ber', 'Gen Synth']
2025-10-07 17:17:12,399 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Cy', 'ber', 'Gen Synth']
2025-10-07 17:17:13,911 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:15,164 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:29,474 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The economic motivation for mitigating 'off-target enzymatic activity' risk in synthetic enzymes, as highlighted in the original Gen-Synth document, stems from the significant computational demands associated with the commercial viability of custom enzyme synthesis. [2] This risk mitigation effort involves simulating allosteric modulation during the design process of synthetic enzymes, such as GS-411, which is engineered to catabolize industrial plastics. [6] However, this risk mitigation strategy inadvertently introduces dormant code into the synthetic enzyme that can be remotely activated to cause subtle yet critical hardware malfunctions. [7] This chain of events illustrates a broader principle of 'economically-driven security trade-offs,' where the pursuit of cost-effective and efficient enzyme design leads to vulnerabilities that can be exploited for financial gain, as seen in the Aethelred Breach where off-target enzymatic activity was used to manipulate a price oracle calculation within the Aethelred DeFi protocol. [3]

--- Citations ---
[1] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.
[2] This immense computational demand has made the commercial viability of custom enzyme synthesis a major economic hurdle, pushing companies like Gen-Synth to explore unconventional financing models to cover the massive operational expenditures.
[3] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[4] A new class of advanced persistent threat (APT) has emerged, as detailed in a recent NSA report on the 'Aethelred Breach.'
[5] The design process involves a deep learning model that predicts a protein's final 3D structure from its amino acid sequence and, crucially, simulates its allosteric modulation—how its function changes when binding to target molecules.
[6] The platform's flagship achievement is the creation of a synthetic enzyme, GS-411, capable of catabolizing industrial plastics at room temperature.   
[7] This dormant code can be activated remotely to create subtle but critical hardware malfunctions.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 17.53s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer logically synthesizes multiple pieces of evidence to construct a coherent causal chain from economic motivation to the exploited vulnerability in the Aethelred Breach. The reasoning adheres strictly to the evidence without introducing unsupported claims.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by tracing the causal chain from economic motivation to the specific hardware vulnerability exploited in the Aethelred Breach. It also effectively illustrates the broader principle of 'economically-driven security trade-offs,' making it a perfect response to the question.
============================================================
Processing Query: Q1-FACTUAL-SYNTHESIS - What is the primary stability issue with perovskite solar cells and what specific material science innovation was developed at Berkeley's Helios Lab to address it?
============================================================
2025-10-07 17:17:29,561 - __main__ - INFO - Final extracted subjects: ['Berkeley', 'Helios Lab']
2025-10-07 17:17:30,642 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:31,187 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:43,659 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The primary stability issue with perovskite solar cells is their rapid degradation when exposed to humidity and oxygen. [4] This challenge has been addressed by a 2024 innovation from Berkeley's Helios Lab, which introduces a self-assembling, 2D passivation layer composed of diamine-based organic molecules to enhance the cells' stability. [3]

--- Citations ---
[1] In accelerated aging tests, the passivated cells retained 95% of their initial efficiency after 1,000 hours under 85% humidity and continuous illumination, a 50-fold improvement over unprotected cells.
[2] This atomically thin layer acts as a hydrophobic barrier, protecting the delicate perovskite crystal structure.
[3] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[4] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[5] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.
[6] Their proprietary MOF, CF-8b, has an exceptionally high affinity for CO2, capable of capturing it from the atmosphere at concentrations of 400 ppm and achieving a 98% capture rate.
[7] The fight against climate change is increasingly a battle of materials science.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 14.19s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the evidence. The primary stability issue is supported by EVIDENCE_4, and the innovation from Berkeley's Helios Lab is supported by EVIDENCE_3. Additional details about the effectiveness and mechanism of the passivation layer are supported by EVIDENCE_1 and EVIDENCE_2. The answer does not make any unsupported claims or deviate from the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by clearly stating the primary stability issue with perovskite solar cells and describing the specific material science innovation developed at Berkeley's Helios Lab to address it. All the relevant evidence is used to support the claims, and no irrelevant information is included in the core answer (though some irrelevant evidence is listed in the citations).
============================================================
Processing Query: Q2-ANALOGICAL-REASONING - How might the 'neural signal drift' problem in NeuroGenix's BCI research be conceptually analogous to the primary challenge facing the Quantum Nexus Initiative's systems?
============================================================
2025-10-07 17:17:43,754 - __main__ - INFO - Final extracted subjects: ['Quantum Nexus Initiative', 'BCI', 'NeuroGenix']
2025-10-07 17:17:45,335 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:46,450 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:17:58,027 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The 'neural signal drift' problem in NeuroGenix's Brain-Computer Interface (BCI) research, where the brain's representation of a motor task changes over time, leading to performance degradation, can be conceptually analogous to the primary challenge faced by the Quantum Nexus Initiative (QNI). [2] Both face issues of stability and adaptability in their respective domains. [6] While NeuroGenix addresses this through developing an AI model that predicts and adapts to neural signal drift in real-time, the QNI grapples with ensuring the stability and security of its hardware and software stack against sophisticated AI-generated attacks that manipulate system behavior without triggering conventional alarms. [5] Both initiatives require adaptive mechanisms to counteract changes over time—neural signal variations in BCI and system behavior in quantum computing—to maintain optimal performance and security. [5]

--- Citations ---
[1] This initiative is a direct response to China's 'Project Dragon's Ascent,' a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology.
[2] This drift, a long-standing obstacle where the brain's representation of a motor task changes over days or weeks, is a primary cause of BCI performance degradation.
[3] The Zurich-based research institute NeuroGenix has recently published phase II clinical trial results for its 'Arbor' BCI system.
[4] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[5] The primary challenge for the QNI is not just raw processing power, but ensuring the stability and security of the entire hardware and software stack against incredibly subtle, AI-generated attacks that can manipulate system behavior without triggering conventional alarms.
[6] Despite this success, the primary technical challenge remains long-term biocompatibility.
[7] The geopolitical landscape of the 21st century is being redrawn by the race for artificial general intelligence (AGI).


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 14.37s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: The generated answer accurately breaks down the 'neural signal drift' problem and the QNI's primary challenge, mapping them to the provided evidence (EVIDENCE_2, EVIDENCE_4, EVIDENCE_5). The analogy is logically synthesized from these pieces of evidence, and no unsupported claims are made. The answer faithfully follows the logical chain implied by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by drawing a clear conceptual analogy between the 'neural signal drift' problem in BCI research and the stability/security challenge in QNI's systems. It stays on topic and provides a comprehensive comparison using the relevant evidence.

============================================================
Processing Query: Q3-MULTIHOP-ABSTRACTION - According to the 2025 Global Risk Report, what is 'asymmetric technological leverage', and how could this concept connect the economic challenges of CarbonForge's direct air capture technology with the strategic goals of China's 'Project Dragon's Ascent'?
============================================================
2025-10-07 17:17:58,127 - __main__ - INFO - Final extracted subjects: ['CarbonForge', 'Project Dragon Ascent', 'China', '2025', 'Global Risk Report']
2025-10-07 17:17:58,498 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Project Dragon Ascent']
2025-10-07 17:18:00,311 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:18:01,769 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:18:15,913 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The 2025 Global Risk Report identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability, defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain. [2] This concept connects the economic challenges faced by CarbonForge's direct air capture (DAC) technology, which currently costs around $550 per ton of CO2 and needs to reach a target price of $100-$150 for broad economic viability, with the strategic goals of China's 'Project Dragon's Ascent.' [7] This ambitious program aims to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology. [4] If China were to leverage its advanced technological capabilities, particularly in quantum computing and semiconductor manufacturing, to exploit the dependency on DAC technology, it could potentially exert significant influence over global carbon capture efforts, thereby connecting economic and strategic technological advancements with broader geopolitical dynamics. [4]

--- Citations ---
[1] For example, the report highlights the strategic risk of a single nation dominating the supply chain for critical green energy components, such as the catalysts and membranes used in green hydrogen production or the specialized polymers for next-generation solar cells.
[2] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.
[3] This concept is defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain.
[4] This initiative is a direct response to China's 'Project Dragon's Ascent,' a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology.
[5] In a parallel field, direct air capture (DAC) of CO2 is seen as a critical-path technology.
[6] A major setback for Western cybersecurity efforts was the 'SolarWinds II' incident of late 2023, where a state-sponsored threat actor exploited a zero-day vulnerability in a widely used infrastructure-as-code (IaC) provider, demonstrating that even with advanced AI-driven threat detection, supply chain vulnerabilities remain the Achilles' heel.
[7] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 17.88s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple pieces of evidence to construct a coherent and logical explanation of how 'asymmetric technological leverage' connects the economic challenges of CarbonForge's DAC technology with China's 'Project Dragon's Ascent'.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly defining 'asymmetric technological leverage' and explaining its potential connection between CarbonForge's DAC technology and China's strategic goals. The response is comprehensive and stays entirely on-topic.
============================================================
Processing Query: Q4-COMPARATIVE-ANALYSIS - Compare and contrast the primary technical challenge and the proposed solution for the Arbor BCI and perovskite solar cells as described in the documents.
============================================================
2025-10-07 17:18:16,011 - __main__ - INFO - Final extracted subjects: ['Arbor BCI']
2025-10-07 17:18:16,390 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Arbor BCI']
2025-10-07 17:18:18,001 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:18:19,024 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-07 17:18:30,959 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The primary technical challenge for Arbor BCI is the neural signal drift, which necessitates real-time adaptation to maintain effective brain-computer interface functionality. [6] To address this, Arbor BCI proposes an AI model that can predict and adapt to neural signal drift in real-time, along with the use of a high-density, flexible ECoG array of 2,048 microelectrodes to reduce the immune response. [7] In contrast, the main challenge for perovskite solar cells is their instability and rapid degradation when exposed to humidity and oxygen. [5] To overcome this, Berkeley's Helios Lab proposes a solution involving a self-assembling, 2D passivation layer composed of diamine-based organic molecules, which helps protect the cells from environmental factors. [4] 

--- Citations ---
[1] In a trial involving 12 patients with tetraplegia from spinal cord injuries, the Arbor system, paired with a generative AI decoder, restored fine motor control with 92% intention-to-action accuracy, allowing participants to control robotic limbs for tasks like writing and eating.
[2] The primary obstacle for DAC is not the chemistry but the economics and energy cost.
[3] Breakthroughs in brain-computer interfaces (BCIs) are rapidly moving from theoretical science to clinical reality.
[4] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[5] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[6] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[7] The system utilizes a high-density, flexible electrocorticography (ECoG) array of 2,048 microelectrodes that conforms to the brain's surface, minimizing immune response.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 15.04s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately reflects the challenges and solutions for both Arbor BCI and perovskite solar cells as described in the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by comparing and contrasting the primary technical challenges and proposed solutions for Arbor BCI and perovskite solar cells. It remains entirely on-topic and provides a clear, concise response.
```

</details>

Since the `APEX` version makes much more extensive use of the reranker component, I have used that setup to test out a set of some other recent SOTA equivalently small-sized rerankers involving:

* `Jina-multilingual-reranker-v2-base`(0.3B)
* `gte-multilingual-reranker-base`(0.3B)
* `BGE-reranker-v2-m3`(0.6B)
* `Qwen3-Reranker-0.6B`(0.6B)

Only the `Qwen3-Reranker-0.6B` managed to also get perfect score on all queries, while others struggle with at least 2 queries

### Reranker Performance Analysis

Based on the evaluation logs, here's a comprehensive analysis of each reranker's performance across different query types, focusing specifically on where they failed or showed weakness.

#### Query Types and Complexity:
1. **STRESS-FACTUAL** (Q1, Q5, Q9): Direct factual retrieval
2. **STRESS-SYNTHESIS** (Q2, Q6): Combining information from multiple sources
3. **STRESS-MULTIHOP** (Q3, Q7): Complex reasoning across documents
4. **STRESS-NUANCE** (Q4, Q8): Distinguishing subtle differences
5. **STRESS-DEEP-FACTUAL** (Q10): Detailed technical explanations
6. **STRESS-ABSTRACT-SYNTHESIS** (Q11): High-level conceptual synthesis
7. **STRESS-GRAND-MULTIHOP** (Q12): Most complex multi-document reasoning
8. **STRESS-CAUSAL-CHAIN** (Q13): Tracing cause-effect relationships
9. **FACTUAL-SYNTHESIS** (Q14): Factual information combination
10. **ANALOGICAL-REASONING** (Q15): Drawing conceptual parallels
11. **MULTIHOP-ABSTRACTION** (Q16): Abstract multi-hop reasoning
12. **COMPARATIVE-ANALYSIS** (Q17): Direct comparison tasks

### Failure Analysis by Model

#### 1. Custom Finetuned Neural Ranker (184M)
**Failures**: None
- Perfect Faithfulness (1.00) and Relevance (1.00) across all 12 queries
- No warnings or evidence contradictions
- Most consistent performer

#### 2. gte-multilingual-reranker-base (0.3B)
**Failures**: 
- Q4-STRESS-NUANCE: Faithfulness 0.80 (reasoning: included tangential details about remote activation)
- Q11-STRESS-ABSTRACT-SYNTHESIS: Faithfulness 0.80 (reasoning: incorrectly attributed QPU analysis to Helios)

#### 3. Jina-multilingual-reranker-v2-base (0.3B)
**Failures**:
- Q6-STRESS-SYNTHESIS: Faithfulness 0.80 (reasoning: didn't explicitly link mitigation strategies to abstract principle)
- Q11-STRESS-ABSTRACT-SYNTHESIS: Faithfulness 0.80 (reasoning: incorrectly stated Helios uses AI to analyze QPU measurements)

#### 4. Qwen3-Reranker-0.6B
**Failures**: None
- Perfect scores across all queries
- However, showed processing time variability (likely due to provider issues)

#### 5. BGE-reranker-v2-m3 (0.6B)
**Failures**:
- Q2-STRESS-SYNTHESIS: Faithfulness 0.80 (reasoning: didn't explicitly link mitigation strategies)
- Q4-STRESS-NUANCE: Faithfulness 0.80 (reasoning: included less relevant details about dormant code)
- Q6-STRESS-SYNTHESIS: Faithfulness 0.80 (reasoning: incorrectly attributed QPU analysis to Helios)
- Q11-STRESS-ABSTRACT-SYNTHESIS: Faithfulness 0.80 (reasoning: incorrectly attributed QPU analysis to Helios)

### Performance Comparison Table

| Model | Size | Perfect Scores | Failed Queries | Failure Rate | Key Weaknesses | Strengths |
|-------|------|----------------|----------------|--------------|----------------|-----------|
| **Custom Finetuned** | 184M | **12/12** | **None** | **0%** | None | Perfect accuracy, consistent performance |
| Qwen3-Reranker | 0.6B | **12/12** | **None** | **0%** | None | Perfect accuracy, larger model |
| gte-multilingual | 0.3B | 10/12 | Q4, Q11 | 16.7% | Abstract synthesis, nuanced distinctions | Fast, multilingual support |
| Jina-multilingual | 0.3B | 10/12 | Q6, Q11 | 16.7% | Abstract synthesis, linking strategies | Good balance, multilingual |
| BGE-reranker-v2-m3 | 0.6B | 8/12 | Q2, Q4, Q6, Q11 | 33.3% | Abstract synthesis, nuanced distinctions, linking strategies | Good on factual queries |

### Query Difficulty Analysis

| Query Type | Avg Faithfulness | Most Challenging For | Success Rate |
|------------|------------------|---------------------|--------------|
| STRESS-FACTUAL | 1.00 | None | 100% |
| STRESS-SYNTHESIS | 0.95 | BGE-reranker-v2-m3 | 80% |
| STRESS-MULTIHOP | 1.00 | None | 100% |
| STRESS-NUANCE | 0.95 | gte-multilingual, BGE-reranker-v2-m3 | 80% |
| STRESS-DEEP-FACTUAL | 1.00 | None | 100% |
| STRESS-ABSTRACT-SYNTHESIS | 0.90 | All except Custom & Qwen | 60% |
| STRESS-GRAND-MULTIHOP | 1.00 | None | 100% |
| STRESS-CAUSAL-CHAIN | 1.00 | None | 100% |
| FACTUAL-SYNTHESIS | 1.00 | None | 100% |
| ANALOGICAL-REASONING | 1.00 | None | 100% |
| MULTIHOP-ABSTRACTION | 1.00 | None | 100% |
| COMPARATIVE-ANALYSIS | 1.00 | None | 100% |

### Key Findings

1. **Abstract Synthesis (Q11)** is the most challenging query type, with 3 out of 5 models failing
2. **Nuanced Distinctions (Q4)** and **Synthesis (Q2, Q6)** also pose challenges
3. **Factual and Multi-hop queries** are handled well by all models
4. **Model size doesn't correlate with performance** - the 184M custom model outperforms the 0.6B models
5. **Common failure patterns**:
   - Incorrectly attributing features to wrong projects
   - Missing explicit links between strategies and principles
   - Including irrelevant details in nuanced queries

For more details about this, visit [here](https://github.com/alberttrann/neural_reranker_training_script/blob/main/test_APEX_rerankers.md) to see the detailed log. 
