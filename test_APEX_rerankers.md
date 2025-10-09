<details>
<summary>`BGE-reranker-v2-m3`</summary>  

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
    
    # === MODIFIED SECTION START (1/3) ===
    # Updated the neural ranker path to the desired GTE model.
    neural_ranker_path: str = 'BAAI/bge-reranker-v2-m3'
    # === MODIFIED SECTION END (1/3) ===

    fpt_api_key: str = os.environ.get("FPT_API_KEY", "sk-0g0pXwF4HXrqu-CoydFkWA") # Use environment variable or fallback

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

# === MODIFIED SECTION START (2/3) ===
# This entire class has been replaced to support the GTE reranker model.
class NeuralRanker:
    def __init__(self, model_path: str, config: PipelineConfig, device: str = DEVICE):
        self.device = device
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # GTE model requires trust_remote_code=True. Using float16 for performance on CUDA.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(device).eval()
        # The sbert_model from the previous implementation is no longer needed in this class.

    @torch.no_grad()
    def rank_with_scores(self, query: str, sentences: List[Sentence]) -> List[Sentence]:
        """
        Reranks sentences using the Alibaba GTE reranker model.
        This model expects simple [query, sentence] pairs and does not need
        the complex feature strings of the previous reranker.
        """
        if not sentences:
            return []

        # Create pairs of [query, sentence_text] for the reranker model.
        pairs = [[query, s.text] for s in sentences]
        
        all_scores = []
        # Use batching to process a large number of sentences without memory issues.
        batch_size = 32

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.config.max_length
            ).to(self.device)
            
            # The GTE reranker's output logits are the relevance scores.
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        # Assign the calculated scores back to each sentence object.
        for s, score in zip(sentences, all_scores):
            s.relevance_score = float(score)
            
        # Return the sentences sorted by their new relevance score in descending order.
        return sorted(sentences, key=lambda s: s.relevance_score, reverse=True)
# === MODIFIED SECTION END (2/3) ===

# ==============================================================================
# --- APEX RAG CONTROLLER ---
# ==============================================================================
import hdbscan
from sklearn.cluster import SpectralClustering
from collections import Counter
class KeystoneRAGController:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_manager = DataManager()
        self.preprocessor = EnhancedPreprocessor(config)
        
        # === MODIFIED SECTION START (3/3) ===
        # The instantiation of NeuralRanker is simplified as it no longer needs the sbert_model.
        self.neural_ranker = NeuralRanker(
            model_path=config.neural_ranker_path, 
            config=config 
        )
        # === MODIFIED SECTION END (3/3) ===
        
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
        print("\n" + "="*80 + "\n🚀APEX ARCHITECTURE\n" + "="*80)
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

<details>
<summary>BGE log</summary>

```
================================================================================
🚀APEX ARCHITECTURE
================================================================================
Enable LLM-as-a-Judge evaluation? (yes/no): yes
2025-10-09 21:20:59,652 - __main__ - INFO - LLM-as-a-Judge ENABLED: True

[1] Pre-loaded Queries, [2] Interactive, [quit]: 1

============================================================
Processing Query: Q1-STRESS-FACTUAL - What is a 'decoherence cascade' as described in the Project Chimera document, and what is the specific AI-driven methodology Dr. Eva Rostova's team uses to mitigate it?
============================================================
2025-10-09 21:21:00,793 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'AI', 'Eva Rostova']
2025-10-09 21:21:03,301 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:21:04,213 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
2025-10-09 21:23:11,857 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 524 <none>"
2025-10-09 21:23:11,859 - openai._base_client - INFO - Retrying request to /chat/completions in 0.432307 seconds
2025-10-09 21:23:21,879 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
A 'decoherence cascade' is a phenomenon where a single qubit's loss of quantum state can trigger a chain reaction, corrupting the entanglement across the entire Quantum Processing Unit (QPU). [2] To mitigate this issue, Dr. Eva Rostova's team employs an AI-driven, real-time pulse-level control system. [1] This system continuously monitors qubit frequencies and recalibrates microwave control pulses on a nanosecond timescale to preemptively stabilize the quantum state. [4] By learning the unique noise signature of the QPU, the AI enhances overall stability and effectively mitigates potential 'decoherence cascades.' [4]

--- Citations ---
[1] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[2] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.
[3] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[4] This AI constantly monitors qubit frequencies and recalibrates microwave control pulses on a nanosecond timescale to preemptively stabilize the quantum state, essentially 'learning' the unique noise signature of the QPU.
[5] The system has successfully extended the effective computation time by over 400%, but remains vulnerable to highly specific hardware-level exploits, a concern that has grown significantly since the infamous 'Aethelred Breach' demonstrated the viability of such attacks against specialized silicon.
[6] The core technology leverages variational quantum eigensolvers (VQE) to navigate the colossal state spaces inherent in training large-scale foundation models.
[7] This renders the results of the computation useless.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 141.18s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The definition of 'decoherence cascade' comes from EVIDENCE_2, and the AI-driven mitigation strategy is described in EVIDENCE_1, EVIDENCE_4, and EVIDENCE_5. The answer does not introduce any unsupported claims or deviate from the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by clearly defining 'decoherence cascade' and detailing the specific AI-driven methodology used to mitigate it. All information is directly relevant to the question and sourced from the provided evidence.

============================================================
Processing Query: Q2-STRESS-SYNTHESIS - Synthesize the core operational challenge described in 'Project Chimera' (decoherence cascades) with the one in 'Gen-Synth' (off-target enzymatic activity). What abstract principle of 'high-dimensional system control' do both challenges fundamentally represent?
============================================================
2025-10-09 21:23:21,987 - __main__ - INFO - Final extracted subjects: ['Project Chimera', 'Gen Synth']
2025-10-09 21:23:22,235 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-09 21:23:23,494 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:23:24,443 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:23:37,260 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Both 'Project Chimera' and 'Gen-Synth' face significant operational challenges in maintaining system integrity and preventing unintended interactions. [3] Specifically, 'Project Chimera' grapples with 'decoherence cascades,' where a loss of quantum state in one qubit can corrupt the entire quantum processing unit, while 'Gen-Synth' deals with 'off-target enzymatic activity,' where created enzymes interact with unintended molecules, posing environmental risks. [3] These challenges underscore the difficulty in controlling high-dimensional systems, where small errors can lead to widespread failures. [3] Both projects require sophisticated control mechanisms, such as AI-driven real-time pulse-level control in 'Project Chimera,' to mitigate these issues. [4]

--- Citations ---
[1] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[2] Project Chimera, spearheaded by the Zurich Quantum Institute, represents a paradigm shift in artificial intelligence by directly integrating generative AI models with quantum processing units (QPUs).
[3] A primary challenge, as detailed in their latest internal report, is the management of 'decoherence cascades,' a phenomenon where a single qubit's loss of quantum state triggers a chain reaction that corrupts the entanglement across the entire QPU.
[4] Dr. Eva Rostova's team has pioneered a mitigation strategy using an AI-driven, real-time pulse-level control system.
[5] The field of synthetic biology has been revolutionized by Gen-Synth Corporation's 'differentiable biology' platform, which uses generative AI to design novel proteins and enzymes with bespoke functions.
[6] This renders the results of the computation useless.
[7] The design process involves a deep learning model that predicts a protein's final 3D structure from its amino acid sequence and, crucially, simulates its allosteric modulation—how its function changes when binding to target molecules.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 0.860
  - Processing Time: 15.38s
  - Evidence Contradiction Score: 0.000
  - ⚠️  Warnings: JUDGE_FOUND_UNFAITHFUL

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 0.80/1.00 | Reasoning: The answer accurately captures the core challenges from the evidence (EVIDENCE_1, EVIDENCE_3) and synthesizes them into a coherent principle of 'high-dimensional system control.' However, it does not explicitly link the mitigation strategies (e.g., EVIDENCE_4 for Project Chimera) to the abstract principle, nor does it provide a similar example for Gen-Synth. The citations are listed but not directly referenced in the text, which slightly detracts from faithfulness.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by synthesizing the two challenges into an abstract principle of 'high-dimensional system control.' It stays on-topic and provides a clear, relevant response to the query.

============================================================
Processing Query: Q3-STRESS-MULTIHOP - Based on all documents, what is the plausible economic motive behind the 'Aethelred Breach,' and how does the specific cyberattack vector used (SLCI) create a direct, tangible link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by the DALTA legislation?
============================================================
2025-10-09 21:23:37,375 - __main__ - INFO - Final extracted subjects: ['Zurich Quantum Institute', 'DALTA', 'Aethelred Breach', 'SLCI']
2025-10-09 21:23:39,423 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:23:40,902 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:25:47,460 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 524 <none>"
2025-10-09 21:25:47,461 - openai._base_client - INFO - Retrying request to /chat/completions in 0.381469 seconds
2025-10-09 21:26:01,494 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The plausible economic motive behind the 'Aethelred Breach' is rooted in the manipulation of price oracles within the decentralized finance (DeFi) sector, as evidenced by the Aethelred protocol incident where a sophisticated actor exploited a flash loan mechanism to cause over $2 billion in losses. [3] This breach highlights the systemic risks in DeFi, which prompted the passage of the Digital Asset Liability & Transparency Act (DALTA). [2] The specific cyberattack vector used, Substrate-Level Code Injection (SLCI), serves as a direct, tangible link between the operational risks of the Zurich Quantum Institute and the regulatory concerns addressed by DALTA. [1] SLCI is a sophisticated supply-chain attack that bypasses traditional software-based security measures, posing a significant threat to national strategic initiatives like Project Chimera. [1] Given that QPU-specific variants of SLCI could potentially sabotage quantum computations without detection, the Zurich Quantum Institute's involvement in such initiatives makes it a prime target. [7] Furthermore, the 'anomalous high-energy particle phenomena' are flagged as the most likely vector for sabotage campaigns against strategic initiatives, underscoring the critical nature of securing these operations from such threats. [5]

--- Citations ---
[1] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.
[2] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).        
[3] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[4] Dr. Aris Thorne, DARPA's lead on the project, stated, 'The nation that controls the quantum information space controls the future of strategic intelligence.'
[5] The document concludes by authorizing a new, top-secret research initiative under the NSA, tasked with identifying and shielding against the 'anomalous high-energy particle phenomena' that have been reported to interfere with all three pillar projects, flagging it as the most likely vector for such a sabotage campaign.
[6] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.   
[7] The report explicitly warns that the QPU-specific variants of SLCI being developed by state-level actors pose a direct and potentially catastrophic threat to national strategic initiatives like Project Chimera, as they could be used to subtly sabotage quantum computations and invalidate years of research without detection.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 144.23s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple pieces of evidence to form coherent and logical claims, adhering closely to the reasoning implied by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by explaining the economic motive behind the 'Aethelred Breach' and the link between SLCI, the Zurich Quantum Institute, and DALTA. It covers all aspects of the question comprehensively and stays on topic throughout.

============================================================
Processing Query: Q4-STRESS-NUANCE - Distinguish between the concept of 'systemic risk' as addressed by DALTA and the 'environmental risk' posed by Gen-Synth's platform. How are both of these risks examples of AI-driven 'unintended consequences' that traditional risk models might fail to predict?
============================================================
2025-10-09 21:26:01,632 - __main__ - INFO - Final extracted subjects: ['DALTA', 'AI', 'Gen Synth']
2025-10-09 21:26:02,003 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Gen Synth']
2025-10-09 21:26:04,184 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:26:05,561 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:26:14,707 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Systemic risk in the decentralized finance (DeFi) sector, as addressed by the Digital Asset Liability & Transparency Act (DALTA), primarily stems from operational vulnerabilities in automated market makers (AMMs) and lending protocols, particularly their susceptibility to 'oracle manipulation' and cascading liquidations. [7] An example of such a risk is the 'Substrate-Level Code Injection' (SLCI) attack vector, which can bypass traditional software-based security measures. [4] On the other hand, Gen-Synth Corporation's 'differentiable biology' platform uses generative AI to design novel proteins and enzymes, posing an environmental risk through off-target enzymatic activity, where created enzymes may inadvertently interact with and break down unintended but structurally similar molecules. [1] Both of these risks exemplify AI-driven unintended consequences that traditional risk models might struggle to predict due to the complexity and novel nature of AI-driven systems. [7] The SLCI attack underscores how sophisticated supply-chain attacks can exploit vulnerabilities in AI systems, while off-target enzymatic activity highlights the potential for environmental harm from biotechnology. [4]

--- Citations ---
[1] The field of synthetic biology has been revolutionized by Gen-Synth Corporation's 'differentiable biology' platform, which uses generative AI to design novel proteins and enzymes with bespoke functions.
[2] This has raised concerns that nation-states could weaponize such techniques to destabilize financial markets.
[3] In response to escalating systemic risk in the decentralized finance (DeFi) sector, the US Treasury has passed the Digital Asset Liability & Transparency Act (DALTA).        
[4] The attack vector, dubbed 'Substrate-Level Code Injection' (SLCI), represents a sophisticated supply-chain attack that bypasses traditional software-based security.
[5] While incredibly powerful, the primary operational challenge is 'off-target enzymatic activity,' where the created enzyme inadvertently interacts with and breaks down unintended but structurally similar molecules, posing a significant environmental risk if released.
[6] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[7] The legislation targets the operational vulnerabilities of automated market makers (AMMs) and lending protocols, particularly their susceptibility to 'oracle manipulation' and cascading liquidations triggered by AI-driven high-frequency trading algorithms.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 13.21s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately maps to EVIDENCE_3, EVIDENCE_4, EVIDENCE_7 for systemic risk and EVIDENCE_1, EVIDENCE_5 for environmental risk. The synthesis of these points into AI-driven unintended consequences is logically sound and supported by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly distinguishing between systemic risk and environmental risk, and it effectively explains how both are examples of AI-driven unintended consequences. The answer stays on topic and provides a comprehensive response to the question.

============================================================
Processing Query: Q1-STRESS-DEEP-FACTUAL - Describe the specific mechanism of 'synaptic interference' in Project Mnemosyne, and detail the architecture and function of the 'Predictive Hebbian Modulator' AI used to counteract it.
============================================================
2025-10-09 21:26:14,875 - __main__ - INFO - Final extracted subjects: ['Project Mnemosyne', 'Hebbian Mo', 'AI']
2025-10-09 21:26:16,013 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:26:16,890 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:27:27,895 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 502 Bad Gateway"
2025-10-09 21:27:27,896 - openai._base_client - INFO - Retrying request to /chat/completions in 0.478250 seconds
2025-10-09 21:27:38,323 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
In Project Mnemosyne, synaptic interference arises from an anomalous, high-energy particle flux that amplifies light's effect beyond its intended target. [4] To counteract this, the system employs a sophisticated AI known as the Predictive Hebbian Modulator. [1] This AI is a recurrent neural network equipped with a temporal-convolutional attention mechanism, designed to learn the unique synaptic potentiation patterns of an individual's brain. [3] By doing so, it can pre-emptively adjust the implant's light frequency and intensity to create a 'counter-resonance,' thereby shielding nearby memories from the detrimental effects of the anomalous particles. [5]

--- Citations ---
[1] To counter this, the team developed a sophisticated AI called the 'Predictive Hebbian Modulator.'
[2] Project Mnemosyne, a DARPA initiative headquartered at MIT's Media Lab, is developing a next-generation Brain-Computer Interface (BCI) focused on direct memory encoding and retrieval.
[3] This is a recurrent neural network with a temporal-convolutional attention mechanism that learns the unique synaptic potentiation patterns of an individual's brain.
[4] The lead researcher, Dr. Aris Thorne, noted that the interference is caused by an anomalous, high-energy particle flux that seems to amplify the light's effect beyond its intended target.
[5] It then pre-emptively adjusts the implant's light frequency and intensity to create a 'counter-resonance,' effectively shielding nearby memories from the anomalous particle effects.
[6] While early results have shown an unprecedented 98% recall accuracy for encoded information, the primary operational risk is 'synaptic interference.'
[7] The system uses a novel 'neuro-photonic' implant that translates digital data into precisely targeted light patterns to stimulate and modify hippocampal engrams.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 83.61s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple pieces of evidence to describe both the synaptic interference mechanism and the Predictive Hebbian Modulator's architecture and function.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by detailing both the synaptic interference mechanism and the Predictive Hebbian Modulator's architecture and function. All provided information is directly relevant to the question asked.

============================================================
Processing Query: Q2-STRESS-ABSTRACT-SYNTHESIS - Synthesize the core challenges of Project Erebus (Entanglement Fraying) and Project Helios (Lattice Decohesion). What single, unifying principle of 'long-term material and informational integrity' do both projects struggle with, and how do their respective AI-driven solutions represent a similar philosophical approach to predictive maintenance?
============================================================
2025-10-09 21:27:38,484 - __main__ - INFO - Final extracted subjects: ['Project Helios', 'Lattice Decohesion', 'En', 'Project Erebus', 'tanglement Fraying', 'AI']
2025-10-09 21:27:39,957 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:27:41,083 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:27:52,224 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Project Erebus and Project Helios both face significant challenges in maintaining long-term material and informational integrity. [6] Erebus grapples with 'Entanglement Fraying,' while Helios deals with 'Lattice Decohesion,' a material science issue affecting the probe's crystal structure. [6] To address these challenges, both projects employ AI-driven solutions that share a similar philosophical approach to predictive maintenance. [7] Erebus utilizes an AI error-correction model running on a classical co-processor to predict and mitigate Entanglement Fraying, whereas Helios employs AI to analyze parity-check measurements from the Quantum Processing Unit (QPU) in real-time to forecast the onset of Lattice Decohesion. [7] This approach focuses on early detection and prevention of critical failures, ensuring the integrity and reliability of their respective systems over extended periods. [7]

--- Citations ---
[1] The most significant technical hurdle is a persistent issue termed 'Entanglement Fraying.'
[2] NASA's Project Helios is an ambitious plan to send an unmanned probe to Alpha Centauri using a light sail propelled by a high-powered laser array stationed in Earth orbit.   
[3] Dr. Lena Petrova, a consultant from the Zurich Institute, has warned that this AI-dependent repair mechanism represents a novel and critical point of failure that must be protected against external manipulation.
[4] The Erebus team's solution is an AI error-correction model that runs on a classical co-processor.
[5] Analysis has revealed that this accelerated decay is strongly correlated with the same anomalous 'exotic particle flux' documented by other advanced research projects.       
[6] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[7] The AI analyzes the parity-check measurements from the QPU in real-time and predicts the onset of a fraying event.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 0.860
  - Processing Time: 13.89s
  - Evidence Contradiction Score: 0.000
  - ⚠️  Warnings: JUDGE_FOUND_UNFAITHFUL

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 0.80/1.00 | Reasoning: The answer accurately reflects the evidence for the challenges and solutions of both projects. However, it incorrectly states that Helios uses AI to analyze QPU measurements (EVIDENCE_7 refers to Erebus, not Helios). This minor deviation affects the faithfulness score.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by synthesizing the core challenges and AI-driven solutions of both projects, highlighting their shared philosophical approach to predictive maintenance. It remains fully on-topic and comprehensive.

============================================================
Processing Query: Q3-STRESS-GRAND-MULTIHOP - According to the Orion Mandate, what is 'asymmetric temporal sabotage,' and what is the specific, underlying physical phenomenon (not mentioned in the Mandate itself) that mechanistically links the operational risks of all three critical projects: Mnemosyne, Erebus, and Helios?
============================================================
2025-10-09 21:27:52,389 - __main__ - INFO - Final extracted subjects: ['Mandate', 'Mnemosyne', 'Helios', 'Orion Mandate', 'Erebus']
2025-10-09 21:27:54,256 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:27:55,138 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:28:07,396 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
According to the Orion Mandate, asymmetric temporal sabotage refers to an adversary's strategy of introducing subtle, nearly undetectable flaws into long-term, high-cost research projects, causing them to fail years or even decades later, thus neutralizing a nation's technological advantage without direct confrontation. [6] This concept is underpinned by a specific physical phenomenon known as 'exotic particle flux,' which degrades the components of the three critical projects—Mnemosyne, Erebus, and Helios. [2] This flux leads to 'Lattice Decohesion' and memory corruption issues, thereby mechanistically linking the operational risks of these projects. [7]

--- Citations ---
[1] The mandate's primary concern is a new strategic threat termed 'asymmetric temporal sabotage.'
[2] The document explicitly names three critical, interdependent pillars for achieving this goal: Project Erebus (for computational dominance), Project Mnemosyne (for neural interface superiority), and Project Helios (for demonstrating advanced materials and energy projection).
[3] Extensive testing at JPL revealed that the degradation is not caused by conventional protons or alpha particles, but is almost entirely attributable to the same high-energy, 'exotic particle flux' that has been observed affecting quantum and neurological experiments.
[4] The primary challenge threatening the mission's viability is 'Lattice Decohesion,' a material science problem where the sail's crystal structure becomes brittle and fractures after prolonged exposure to the solar wind and interstellar medium.
[5] The act was fast-tracked following a systemic event in the Aethelred protocol, where a sophisticated actor exploited a flash loan mechanism to manipulate a price oracle, causing a chain of liquidations that led to over $2 billion in losses.
[6] This is defined as an adversary's ability to introduce subtle, almost undetectable flaws into long-term, high-cost research projects, causing them to fail years or decades in the future, thereby neutralizing a nation's technological advantage without a direct confrontation.
[7] This phenomenon occurs when the implant's photonic emissions inadvertently disrupt adjacent, unrelated memory traces, leading to a form of structured amnesia or memory corruption.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 15.17s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The definition of 'asymmetric temporal sabotage' is directly from EVIDENCE_6. The identification of 'exotic particle flux' as the underlying phenomenon is from EVIDENCE_3. The mention of 'Lattice Decohesion' and 'memory corruption' as risks is supported by EVIDENCE_4 and EVIDENCE_7, respectively. The answer logically connects all these elements to address the question.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by defining 'asymmetric temporal sabotage' and identifying the specific physical phenomenon ('exotic particle flux') that links the operational risks of all three critical projects. It stays on-topic and provides a comprehensive response based on the evidence.

============================================================
Processing Query: Q4-STRESS-CAUSAL-CHAIN - Trace the causal chain from the economic motivation for 'off-target enzymatic activity' risk mitigation in the original Gen-Synth document to the specific hardware vulnerability exploited in the 'Aethelred Breach' from the Cyber-Sec document. How does this chain illustrate a broader principle of 'economically-driven security trade-offs'?
============================================================
2025-10-09 21:28:07,517 - __main__ - INFO - Final extracted subjects: ['Cy', 'Se', 'ber', 'Aethelred Breach', 'Gen Synth']
2025-10-09 21:28:07,939 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Cy', 'ber', 'Gen Synth']
2025-10-09 21:28:09,486 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:28:10,521 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:18,100 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 524 <none>"
2025-10-09 21:30:18,102 - openai._base_client - INFO - Retrying request to /chat/completions in 0.444471 seconds
2025-10-09 21:30:30,986 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The economic motivation for mitigating off-target enzymatic activity risk in the original Gen-Synth document, driven by high computational demands and the need for unconventional financing models, led to the development of synthetic enzyme GS-411. [2] This enzyme, designed to address economic pressures, was later exploited in the Aethelred Breach through dormant code that could be remotely activated to cause subtle hardware malfunctions. [6] By manipulating a price oracle calculation, the attackers triggered a cascade of liquidations for significant profit, illustrating how economic incentives can lead to security trade-offs. [3] This chain demonstrates the broader principle of 'economically-driven security trade-offs,' where efforts to optimize economic efficiency inadvertently create vulnerabilities that can be exploited by sophisticated actors. [3]

--- Citations ---
[1] A new class of advanced persistent threat (APT) has emerged, as detailed in a recent NSA report on the 'Aethelred Breach.'
[2] This immense computational demand has made the commercial viability of custom enzyme synthesis a major economic hurdle, pushing companies like Gen-Synth to explore unconventional financing models to cover the massive operational expenditures.
[3] While insignificant for most tasks, these errors were sufficient to manipulate the outcome of a sensitive price oracle calculation within the Aethelred DeFi protocol, allowing the attacker to trigger a cascade of liquidations for immense profit.
[4] The report explicitly warns that the QPU-specific variants of SLCI being developed by state-level actors pose a direct and potentially catastrophic threat to national strategic initiatives like Project Chimera, as they could be used to subtly sabotage quantum computations and invalidate years of research without detection.
[5] The design process involves a deep learning model that predicts a protein's final 3D structure from its amino acid sequence and, crucially, simulates its allosteric modulation—how its function changes when binding to target molecules.
[6] This dormant code can be activated remotely to create subtle but critical hardware malfunctions.
[7] The platform's flagship achievement is the creation of a synthetic enzyme, GS-411, capable of catabolizing industrial plastics at room temperature.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 143.59s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer logically synthesizes multiple pieces of evidence to construct a coherent causal chain from economic motivations to the exploitation of vulnerabilities, adhering closely to the question's requirements.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by tracing the causal chain from economic motivations to the specific hardware vulnerability exploited in the Aethelred Breach. It also effectively illustrates the broader principle of 'economically-driven security trade-offs,' making it highly relevant to the question asked.

============================================================
Processing Query: Q1-FACTUAL-SYNTHESIS - What is the primary stability issue with perovskite solar cells and what specific material science innovation was developed at Berkeley's Helios Lab to address it?
============================================================
2025-10-09 21:30:31,098 - __main__ - INFO - Final extracted subjects: ['Helios Lab', 'Berkeley']
2025-10-09 21:30:32,432 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:33,424 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:41,586 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The primary stability issue with perovskite solar cells is their rapid degradation when exposed to humidity and oxygen. [1] To address this, researchers at Berkeley's Helios Lab developed a self-assembling, 2D passivation layer composed of diamine-based organic molecules. [2] This innovative layer acts as a hydrophobic barrier, effectively protecting the delicate perovskite crystal structure from environmental damage. [3] In accelerated aging tests, cells treated with this passivation layer retained 95% of their initial efficiency after 1,000 hours under 85% humidity and continuous illumination, demonstrating a significant 50-fold improvement in stability compared to unprotected cells. [4]

--- Citations ---
[1] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[2] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[3] This atomically thin layer acts as a hydrophobic barrier, protecting the delicate perovskite crystal structure.
[4] In accelerated aging tests, the passivated cells retained 95% of their initial efficiency after 1,000 hours under 85% humidity and continuous illumination, a 50-fold improvement over unprotected cells.
[5] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.
[6] Their proprietary MOF, CF-8b, has an exceptionally high affinity for CO2, capable of capturing it from the atmosphere at concentrations of 400 ppm and achieving a 98% capture rate.
[7] The fight against climate change is increasingly a battle of materials science.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 10.60s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The primary stability issue is supported by EVIDENCE_1. The description of the innovation is supported by EVIDENCE_2 and EVIDENCE_3. The test results are supported by EVIDENCE_4. There are no unsupported claims or deviations from the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by clearly stating the primary stability issue of perovskite solar cells and detailing the specific material science innovation developed to address it. All information provided is relevant to the question asked.

============================================================
Processing Query: Q2-ANALOGICAL-REASONING - How might the 'neural signal drift' problem in NeuroGenix's BCI research be conceptually analogous to the primary challenge facing the Quantum Nexus Initiative's systems?
============================================================
2025-10-09 21:30:41,685 - __main__ - INFO - Final extracted subjects: ['BCI', 'NeuroGenix', 'Quantum Nexus Initiative']
2025-10-09 21:30:42,673 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:43,631 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:53,459 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The primary challenge for NeuroGenix's Brain-Computer Interface (BCI) research is the 'neural signal drift' problem, which involves the brain's changing representation of a motor task over days or weeks, leading to a decline in BCI performance. [3] This issue is conceptually analogous to the main challenge faced by the Quantum Nexus Initiative, as both involve maintaining system performance in the face of evolving conditions. [1] To address this, NeuroGenix has developed an AI model that predicts and adapts to 'neural signal drift' in real-time, though the initiative still grapples with ensuring long-term biocompatibility. [4]

--- Citations ---
[1] The primary challenge for the QNI is not just raw processing power, but ensuring the stability and security of the entire hardware and software stack against incredibly subtle, AI-generated attacks that can manipulate system behavior without triggering conventional alarms.
[2] The Zurich-based research institute NeuroGenix has recently published phase II clinical trial results for its 'Arbor' BCI system.
[3] This drift, a long-standing obstacle where the brain's representation of a motor task changes over days or weeks, is a primary cause of BCI performance degradation.
[4] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[5] Despite this success, the primary technical challenge remains long-term biocompatibility.
[6] The geopolitical landscape of the 21st century is being redrawn by the race for artificial general intelligence (AGI).
[7] Dr. Aris Thorne, DARPA's lead on the project, stated, 'The nation that controls the quantum information space controls the future of strategic intelligence.'


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 0.860
  - Processing Time: 11.87s
  - Evidence Contradiction Score: 0.000
  - ⚠️  Warnings: JUDGE_FOUND_UNFAITHFUL

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 0.80/1.00 | Reasoning: The answer accurately reflects the evidence about NeuroGenix's BCI research (EVIDENCE_2, EVIDENCE_3, EVIDENCE_4, EVIDENCE_5). However, it does not explicitly state the QNI's primary challenge (EVIDENCE_1, EVIDENCE_7), instead implying it through the analogy. The analogy itself is logically sound but could be more explicitly supported by the evidence.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by drawing a clear conceptual analogy between the challenges faced by NeuroGenix's BCI research and the QNI. It stays focused on the topic and provides relevant details from the evidence.

============================================================
Processing Query: Q3-MULTIHOP-ABSTRACTION - According to the 2025 Global Risk Report, what is 'asymmetric technological leverage', and how could this concept connect the economic challenges of CarbonForge's direct air capture technology with the strategic goals of China's 'Project Dragon's Ascent'?
============================================================
2025-10-09 21:30:53,562 - __main__ - INFO - Final extracted subjects: ['Global Risk Report', 'China', 'Project Dragon Ascent', 'CarbonForge', '2025']
2025-10-09 21:30:53,939 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Project Dragon Ascent']
2025-10-09 21:30:55,915 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:30:57,335 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:33:03,882 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 524 <none>"
2025-10-09 21:33:03,884 - openai._base_client - INFO - Retrying request to /chat/completions in 0.376506 seconds
2025-10-09 21:35:09,328 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 524 <none>"
2025-10-09 21:35:09,330 - openai._base_client - INFO - Retrying request to /chat/completions in 0.848875 seconds
2025-10-09 21:35:23,071 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
Asymmetric technological leverage, as defined in the 2025 Global Risk Report, refers to the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated domain. [4] This concept connects the economic challenges faced by CarbonForge's direct air capture (DAC) technology with the strategic goals of China's 'Project Dragon's Ascent' by highlighting the risk of a single nation dominating the supply chain for critical green energy components. [5] Given that China's 'Project Dragon's Ascent' aims to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology, the report suggests that technological dependencies in green energy components can be exploited to exert strategic influence. [3] This underscores the potential for China to leverage its technological dominance in green energy to achieve broader geopolitical objectives, thereby linking the economic viability of DAC technology with broader strategic considerations. [5]

--- Citations ---
[1] The 2025 Global Risk Report, published by the Zurich Institute, identifies 'asymmetric technological leverage' as the most significant emerging threat to global stability.   
[2] In a parallel field, direct air capture (DAC) of CO2 is seen as a critical-path technology.
[3] This initiative is a direct response to China's 'Project Dragon's Ascent,' a state-funded program aiming to achieve a 10,000-qubit quantum processor by 2028 and dominate the global market for 2nm-and-below lithography technology.
[4] This concept is defined as the ability of a state or non-state actor to weaponize a technological dependency in one domain to exert disproportionate influence in another, seemingly unrelated, domain.
[5] For example, the report highlights the strategic risk of a single nation dominating the supply chain for critical green energy components, such as the catalysts and membranes used in green hydrogen production or the specialized polymers for next-generation solar cells.
[6] A major setback for Western cybersecurity efforts was the 'SolarWinds II' incident of late 2023, where a state-sponsored threat actor exploited a zero-day vulnerability in a widely used infrastructure-as-code (IaC) provider, demonstrating that even with advanced AI-driven threat detection, supply chain vulnerabilities remain the Achilles' heel.      
[7] At current energy prices, this puts the cost of capture at approximately $550 per ton, far above the $100-$150 target needed for broad economic viability.


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 269.61s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the provided evidence. The answer accurately synthesizes multiple pieces of evidence to build a coherent argument about asymmetric technological leverage and its implications for CarbonForge's DAC technology and China's 'Project Dragon's Ascent'.
  - Relevance:    1.00/1.00 | Reasoning: The answer directly addresses the question by defining asymmetric technological leverage and explaining how it connects the economic challenges of CarbonForge's DAC technology with the strategic goals of China's 'Project Dragon's Ascent'. The response is comprehensive and stays on topic throughout.

============================================================
Processing Query: Q4-COMPARATIVE-ANALYSIS - Compare and contrast the primary technical challenge and the proposed solution for the Arbor BCI and perovskite solar cells as described in the documents.
============================================================
2025-10-09 21:35:23,165 - __main__ - INFO - Final extracted subjects: ['Arbor BCI']
2025-10-09 21:35:23,400 - __main__ - INFO - Contextual Expansion: Searching for missing subject(s): ['Arbor BCI']
2025-10-09 21:35:24,790 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:35:25,937 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"
2025-10-09 21:35:34,981 - httpx - INFO - HTTP Request: POST https://mkp-api.fptcloud.com/chat/completions "HTTP/1.1 200 OK"

✅ ANSWER (using Unified Apex Ranker Path):
The primary technical challenge for perovskite solar cells is their instability and rapid degradation when exposed to humidity and oxygen. [1] To address this issue, researchers at Berkeley's Helios Lab have developed a self-assembling, 2D passivation layer composed of diamine-based organic molecules, which helps protect the cells from environmental factors. [3] In contrast, the main technical challenge for Arbor BCI (Brain-Computer Interface) systems is ensuring long-term biocompatibility. [5] To overcome this, Arbor BCI employs an advanced AI model that predicts and adapts to 'neural signal drift' in real-time, combined with a generative AI decoder. [6] Additionally, they utilize a high-density, flexible ECoG array of 2,048 microelectrodes that conform to the brain's surface, thereby reducing the immune response and enhancing long-term stability. [7]

--- Citations ---
[1] While perovskite solar cells have achieved efficiencies exceeding 30% in laboratory settings, their widespread commercial adoption has been crippled by their notorious instability, particularly their rapid degradation when exposed to humidity and oxygen.
[2] Breakthroughs in brain-computer interfaces (BCIs) are rapidly moving from theoretical science to clinical reality.
[3] A 2024 innovation from Berkeley's Helios Lab has provided a potential solution: a self-assembling, 2D passivation layer of diamine-based organic molecules.
[4] In a trial involving 12 patients with tetraplegia from spinal cord injuries, the Arbor system, paired with a generative AI decoder, restored fine motor control with 92% intention-to-action accuracy, allowing participants to control robotic limbs for tasks like writing and eating.
[5] Despite this success, the primary technical challenge remains long-term biocompatibility.
[6] The lead researcher, Dr. Lena Petrova, highlighted that the key innovation was an AI model that predicts and adapts to 'neural signal drift' in real-time.
[7] The system utilizes a high-density, flexible electrocorticography (ECoG) array of 2,048 microelectrodes that conforms to the brain's surface, minimizing immune response.     


📊 VALIDATION & PERFORMANCE:
  - Final Confidence Score: 1.000
  - Processing Time: 11.91s
  - Evidence Contradiction Score: 0.000

🔬 LLM-AS-A-JUDGE EVALUATION:
  - Faithfulness: 1.00/1.00 | Reasoning: Every claim in the generated answer is fully supported by the evidence. The answer accurately reflects the challenges and solutions for both perovskite solar cells (EVIDENCE_1, EVIDENCE_3) and Arbor BCI (EVIDENCE_4, EVIDENCE_5, EVIDENCE_6, EVIDENCE_7). The logical synthesis of multiple evidence IDs is correct and complete.
  - Relevance:    1.00/1.00 | Reasoning: The answer perfectly addresses the question by comparing and contrasting the primary technical challenges and proposed solutions for both technologies. It stays entirely on topic and provides a clear, concise comparison.
```

</details>
