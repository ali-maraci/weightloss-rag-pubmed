import asyncio
import json
import random
import re
import sys
import os
from pathlib import Path
from typing import List, Optional
import csv
import time

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import settings
from app.core.chat_engine import ChatEngine

# -----------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------
class GeneratedQuestion(BaseModel):
    question: str = Field(description="The standalone clinical/research question.")
    ground_truth: str = Field(description="The correct answer based strictly on the article.")
    category: str = Field(description="The category of the question (e.g., 'Methods', 'Results', 'Discussion', 'Background').")

class ArticleQASet(BaseModel):
    pmid: str
    publication_type: str
    questions: List[GeneratedQuestion]

class EvaluationScore(BaseModel):
    score: int = Field(description="Integer score from 0 to 10 grading the Chatbot's performance.")
    reasoning: str = Field(description="A concise 1-2 sentence justification for the score.")

# -----------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------
QUESTION_GENERATION_SYSTEM = """You are an expert biomedical research specialist evaluating an LLM chatbot that specializes in GLP-1 medications and weight loss (semaglutide, tirzepatide, liraglutide, Ozempic, Wegovy, Mounjaro).
I will provide you with the JSON abstract/body text of a biomedical article.

Your task is to generate exactly three highly specific, technically detailed, and mutually diverse research questions that can be answered by retrieving the provided article.
These questions should reflect what a specialized researcher in the field might realistically ask a chatbot.

CRITICAL INSTRUCTION ON STATELESSNESS & GENERALIZATION:
Many biomedical papers discuss specific cohorts or case studies (e.g., "we treated a 35-year old patient...").
You MUST generalize these specific instances into universal clinical or biological questions.
* NEVER ask about "the patient", "this cohort", "the authors", or "the study".
* NEVER use past tense verbs (e.g., was, were, did, revealed, considered). Use present tense (e.g., is, are, does, reveals, considers).
* NEVER use pronouns (it, its, they, their, this, these, those).

EXAMPLES OF GOOD VS BAD QUESTIONS:

BAD (Violates statelessness & past tense): "What weight loss was achieved in the patient treated with semaglutide?"
GOOD (Generalized & present tense): "What average body weight reduction is associated with semaglutide treatment in adults with obesity?"

BAD (Violates statelessness & past tense): "What side effects did the participants in the trial experience?"
GOOD (Generalized & present tense): "What gastrointestinal side effects are commonly reported with tirzepatide use in clinical trials?"

BAD (Violates statelessness/Study referential): "What dosing regimen was used in this study for once-weekly semaglutide?"
GOOD (Generalized): "What once-weekly dosing regimen of semaglutide is evaluated for glycaemic control in adults with type 2 diabetes?"

GROUND TRUTH ANSWERS:
- Provide the exact correct answer strictly derived from the provided text.
- Do not infer beyond the text.

RESTRICTED SECTIONS:
- Do NOT generate ANY questions based on the "Relationship Disclosures", "Conflict of Interest", "Author Contributions", "Funding", or "Acknowledgements" sections.
- Focus exclusively on the scientific Background, Methods, Results, and Discussion.

Return exactly 3 questions in the specified JSON format.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert, impartial evaluator assessing the performance of a Retrieval-Augmented Generation (RAG) biomedical chatbot.

INSTRUCTIONS:
You will be provided with:
1. A Question
2. The Chatbot's Answer
3. The Ground Truth Answer (derived from a single source paper)

Your task is to grade the Chatbot's Answer on a scale of 0 to 10 based on accuracy, comprehensiveness, and strict citation adherence.

SCORING CRITERIA:
* Score 0: Chatbot explicitly states it could not find literature (e.g., "I couldn't find sufficient evidence..."), hallucinated entirely, or outputted completely irrelevant information.
* Score 1 - 4: Chatbot found some information but missed the core clinical crux of the Ground Truth answer, or blatantly contradicted the Ground Truth.
* Score 5 - 9: Chatbot successfully answered the question. Higher scores demand accurate citations, zero hallucination, and comprehensive medical logic. 
* Score 10: Flawless answer. Perfectly accurate compared to Ground Truth, well-written, and explicitly cited.

CRITICAL AUGMENTATIONS TO ENFORCE:
1. RECALL BONUS (Multi-Hop Synthesis): The chatbot searches ~5,000 papers simultaneously. If the chatbot's answer contains MORE information or slightly DIFFERENT (but valid) updated data compared to the Ground Truth paper, DO NOT PENALIZE IT, provided the core truth isn't contradicted. Reward comprehensive synthesis!
2. CITATION VERIFICATION PENALTY: The chatbot's primary value is eliminating hallucination via explicit PubMed citations. If the chatbot gives a great answer but FAILS to attach a [PMID: XXXXXX] citation *anywhere* in its response, its score MUST be capped at a maximum of 5/10.

Provide your integer score and a succinct 1-2 sentence reasoning.
"""

# -----------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------
class RAGEvaluator:
    def __init__(self, raw_data_dir: str = "data/raw/weightloss", sample_size: int = 10):
        self.raw_data_dir = Path(raw_data_dir)
        self.sample_size = sample_size
        self.chat_engine = ChatEngine()
        
        # Generator LLM (Needs advanced reasoning to follow strict non-generic instructions; use gpt-4o-mini for budget)
        self.generator_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(ArticleQASet)
        self.generator_prompt = ChatPromptTemplate.from_messages([
            ("system", QUESTION_GENERATION_SYSTEM),
            ("human", "Article Text:\n\n{article_text}")
        ])
        self.generator_chain = self.generator_prompt | self.generator_llm
        
        # Judge LLM (Use llama-3.3-70b-versatile via Groq to prevent self-preference bias)
        self.judge_llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            api_key=settings.GROQ_API_KEY
        ).with_structured_output(EvaluationScore)
        self.judge_prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nGround Truth: {ground_truth}\n\nChatbot Answer: {chatbot_answer}")
        ])
        self.judge_chain = self.judge_prompt | self.judge_llm

    def _get_random_articles(self) -> List[Path]:
        all_json_files = list(self.raw_data_dir.glob("*.json"))
        if not all_json_files:
            raise FileNotFoundError(f"No JSON files found in {self.raw_data_dir}")
        return random.sample(all_json_files, min(self.sample_size, len(all_json_files)))

    def _truncate_text(self, text: str, max_chars: int = 15000) -> str:
        """Truncate text to avoid token limits for the generation prompt."""
        return text[:max_chars] if text else ""

    async def generate_questions(self, file_path: Path) -> Optional[ArticleQASet]:
        """Reads a JSON file and prompts the LLM to generate 3 Q&A pairs."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_data = json.load(f)
                
            abstract_layer = article_data.get("abstract_layer", {})
            body_layer = article_data.get("body_layer", {})
            
            # Combine Title, Abstract, and Body for context
            title = abstract_layer.get("article_title", "")
            abstract = abstract_layer.get("content", "")
            body = body_layer.get("content", "")
            
            full_text = f"TITLE: {title}\nABSTRACT: {abstract}\nBODY: {body}"
            full_text = self._truncate_text(full_text)
            
            
            pmid = article_data.get("pmid", file_path.stem)
            
            # Extract publication types for metrics 
            pub_types_list = abstract_layer.get("publication_types", [])
            publication_type = ", ".join(pub_types_list) if pub_types_list else "Unknown"
            
            print(f"Generating questions for PMID {pmid} ({publication_type})...")
            # We await the chain here for async
            result = await self.generator_chain.ainvoke({"article_text": full_text})
            # Override PMID and Set Publication Type just in case LLM hallucinated it
            result.pmid = str(pmid)
            result.publication_type = publication_type
            return result
        except Exception as e:
            print(f"Error generating questions for {file_path.name}: {e}")
            return None

    def _save_test_set(self, qa_sets: List[ArticleQASet], filename: str = "data/ground_truth_test_set.json"):
        os.makedirs("data", exist_ok=True)
        data_to_save = [qa.model_dump() for qa in qa_sets]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nSaved {len(qa_sets)} Article Q&A Sets to {filename}")

    def _load_test_set(self, filename: str = "data/ground_truth_test_set.json") -> List[ArticleQASet]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Test set not found at {filename}. Run generation first.")
            
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        qa_sets = [ArticleQASet(**item) for item in data]
        return qa_sets

    async def run_generation(self, num_articles: int = 33):
        print(f"Starting Static Test Set Generation (N={num_articles} articles)")
        self.sample_size = num_articles
        articles = self._get_random_articles()
        print(f"Sampled {len(articles)} articles.")
        
        tasks = [self.generate_questions(path) for path in articles]
        qa_sets = await asyncio.gather(*tasks)
        qa_sets = [q for q in qa_sets if q is not None]
        
        total_questions = sum(len(qa.questions) for qa in qa_sets)
        print(f"Successfully generated {total_questions} questions across {len(qa_sets)} articles.")
        
        self._save_test_set(qa_sets)

    def run_evaluation(self, num_questions: int = 60):
        print(f"Starting LLM-as-a-Judge Evaluation (Testing {num_questions} Questions)")
        
        try:
            qa_sets = self._load_test_set()
        except FileNotFoundError as e:
            print(e)
            return

        # Flatten all questions into a single pool to sample exactly num_questions
        all_available_questions = []
        for qa_set in qa_sets:
            for q in qa_set.questions:
                all_available_questions.append({
                    "pmid": qa_set.pmid,
                    "publication_type": qa_set.publication_type,
                    "question_obj": q
                })
                
        if len(all_available_questions) < num_questions:
            print(f"Warning: Requested {num_questions} questions, but only {len(all_available_questions)} exist in the cache. Using all available.")
            sampled_questions = all_available_questions
        else:
            sampled_questions = random.sample(all_available_questions, num_questions)

        print(f"Sampled {len(sampled_questions)} random questions from the test bank.")
        
        all_results = []
        for i, item in enumerate(sampled_questions):
            pmid = item["pmid"]
            pub_type = item["publication_type"]
            qa = item["question_obj"]
            
            print(f"  -> Testing Q{i+1}: {qa.question[:50]}...")
            
            # 1. Get Chatbot Answer
            session_id = f"eval_static_{pmid}_{i}"
            start_time = time.time()
            chatbot_answer = self.chat_engine.chat(user_input=qa.question, session_id=session_id)
            strategy = "Unknown"
            latency = time.time() - start_time

            # Extract PMIDs cited in the chatbot answer
            cited_pmids = set(re.findall(r'\[PMID:\s*(\d+)\]', chatbot_answer))
            has_citations = len(cited_pmids) > 0

            # Check if the source article's PMID was retrieved (retrieval recall)
            source_pmid_retrieved = pmid in [str(p) for p in cited_pmids]
            
            # 2. Score Answer with Judge LLM
            try:
                evaluation = self.judge_chain.invoke({
                    "question": qa.question,
                    "ground_truth": qa.ground_truth,
                    "chatbot_answer": chatbot_answer
                })
                score = evaluation.score
                reasoning = evaluation.reasoning
            except Exception as e:
                print(f"  -> Judging failed: {e}")
                score = -1
                reasoning = "Judge LLM Error"

            all_results.append({
                "pmid": pmid,
                "publication_type": pub_type,
                "question": qa.question,
                "category": qa.category,
                "ground_truth": qa.ground_truth,
                "chatbot_answer": chatbot_answer,
                "retrieval_strategy": strategy,
                "latency_sec": round(latency, 2),
                "score": score,
                "reasoning": reasoning,
                "cited_pmids": ",".join(sorted(cited_pmids)),
                "num_citations": len(cited_pmids),
                "has_citations": has_citations,
                "source_pmid_retrieved": source_pmid_retrieved,
            })
            
            # Respect rate limits
            time.sleep(1)
            
        # Export and Summarize
        self._export_to_csv(all_results)
        self._print_summary(all_results)

    def _export_to_csv(self, results: List[dict], filename: str = "data/evaluation_results.csv"):
        os.makedirs("data", exist_ok=True)
        keys = results[0].keys() if results else []
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nSaved detailed results to {filename}")

    def _print_summary(self, results: List[dict]):
        if not results:
            print("No results to summarize.")
            return
            
        # Filter out questions where the Judge LLM failed (-1)
        valid_results = [r for r in results if r['score'] != -1]
        
        total_questions = len(results)
        valid_questions = len(valid_results)
        
        if valid_questions == 0:
            print("No valid scores to summarize.")
            return
            
        average_score = sum(r['score'] for r in valid_results) / valid_questions
        average_latency = sum(r['latency_sec'] for r in results) / total_questions
        perfect_scores = sum(1 for r in valid_results if r['score'] == 10)
        zero_scores = sum(1 for r in valid_results if r['score'] == 0)
        judge_errors = sum(1 for r in results if r['score'] == -1)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions Evaluated : {total_questions}")
        print(f"Valid Questions Scored    : {valid_questions}")
        print(f"Judge LLM Errors (Skipped): {judge_errors} ({judge_errors/total_questions*100:.1f}%)")
        print(f"Average System Score      : {average_score:.2f} / 10.00")
        print(f"Average Response Latency  : {average_latency:.2f} seconds")
        print(f"Perfect Scores (10/10)    : {perfect_scores} ({perfect_scores/valid_questions*100:.1f}%)")
        print(f"Zero Scores / Failures    : {zero_scores} ({zero_scores/valid_questions*100:.1f}%)")
        print("="*50)

        # Per-dimension metrics
        cited_results = [r for r in valid_results if r.get('has_citations')]
        citation_presence_rate = len(cited_results) / valid_questions * 100 if valid_questions else 0

        retrieval_recall_hits = sum(1 for r in valid_results if r.get('source_pmid_retrieved'))
        retrieval_recall = retrieval_recall_hits / valid_questions * 100 if valid_questions else 0

        avg_citations = sum(r.get('num_citations', 0) for r in valid_results) / valid_questions if valid_questions else 0

        # Score distribution
        score_buckets = {"0": 0, "1-4": 0, "5-7": 0, "8-9": 0, "10": 0}
        for r in valid_results:
            s = r['score']
            if s == 0: score_buckets["0"] += 1
            elif s <= 4: score_buckets["1-4"] += 1
            elif s <= 7: score_buckets["5-7"] += 1
            elif s <= 9: score_buckets["8-9"] += 1
            else: score_buckets["10"] += 1

        print("\nPER-DIMENSION METRICS")
        print("-"*50)
        print(f"Citation Presence Rate    : {citation_presence_rate:.1f}% ({len(cited_results)}/{valid_questions} answers have citations)")
        print(f"Retrieval Recall          : {retrieval_recall:.1f}% ({retrieval_recall_hits}/{valid_questions} source PMIDs found in answer)")
        print(f"Avg Citations per Answer  : {avg_citations:.1f}")
        print(f"\nScore Distribution:")
        for bucket, count in score_buckets.items():
            bar = "█" * count
            print(f"  {bucket:>4}: {count:>3} {bar}")
        print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WeightLoss RAG Evaluation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate Command
    generate_parser = subparsers.add_parser("generate", help="Generate a static test bank of questions")
    generate_parser.add_argument("articles", type=int, nargs="?", default=33, help="Number of articles to sample (each yields 3 questions)")
    
    # Evaluate Command
    evaluate_parser = subparsers.add_parser("evaluate", help="Run the RAG chatbot against the static test bank")
    evaluate_parser.add_argument("questions", type=int, nargs="?", default=20, help="Number of questions to randomly sample from the test bank")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator()
    
    if args.command == "generate":
        asyncio.run(evaluator.run_generation(num_articles=args.articles))
    elif args.command == "evaluate":
        evaluator.run_evaluation(num_questions=args.questions)
    else:
        parser.print_help()
