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
    fpt_api_key: str = os.environ.get("FPT_API_KEY", "sk-0g0pXwF4HXrqu-CoydFkWA") # Use environment variable or fallback

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