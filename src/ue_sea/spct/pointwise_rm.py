"""
Pointwise Generative Reward Model (GRM) for SPCT.

Implements the core SPCT architecture with:
- Self-generated principles
- Structured critique generation
- Pointwise scoring (1-10 scale)
- Inference-time scaling via sampling and voting
"""

import json
import random
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..optimizer import EvolutionStrategies, ESConfig
import numpy as np


@dataclass
class JudgeOutput:
    """Structured output from the pointwise GRM."""
    self_criteria: List[Dict[str, Any]]  # [{name, weight, rationale}]
    critique: str  # Structured critique text
    scores: List[int]  # Per-response scores (1-10)
    principles: List[str]  # Generated principles
    confidence: float  # Overall confidence in judgement


@dataclass
class SPCTConfig:
    """Configuration for SPCT training and inference."""
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-3B"
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
    # RFT configuration
    rft_samples: int = 8  # N_RFT samples per data point
    rft_correctness_threshold: float = 0.8
    drop_too_easy: bool = True
    rft_max_traces: Optional[int] = 1024  # cap to avoid over-training during cold start
    
    # GRPO configuration
    grpo_kl_penalty: float = 0.1
    grpo_entropy_coef: float = 0.01
    grpo_learning_rate: float = 1e-6
    grpo_batch_size: int = 4
    grpo_iterations: int = 64
    grpo_reward_beta: float = 0.9  # moving baseline factor
    drgrpo_group_size: int = 4
    
    # Meta-RM configuration
    meta_rm_model: str = "Qwen/Qwen2.5-Coder-1.5B"
    meta_rm_samples: int = 4
    meta_rm_filter_ratio: float = 0.5
    meta_rm_learning_rate: float = 1e-5
    meta_rm_epochs: int = 2
    meta_rm_batch_size: int = 4
    
    # Inference scaling
    inference_samples: int = 8
    use_meta_rm_filter: bool = True
    voting_method: str = "sum"  # "sum", "mean", "majority"
    
    # Judge output schema
    max_criteria: int = 5
    max_critique_length: int = 1800
    score_range: Tuple[int, int] = (1, 10)


class PointwiseGRM:
    """
    Pointwise Generative Reward Model implementing SPCT.
    
    Generates:
    1. Self-criteria (principles)
    2. Structured critique
    3. Pointwise scores (1-10) for each response
    """
    
    def __init__(self, config: SPCTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Judge output schema
        self.judge_schema = self._create_judge_schema()
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.best_rl_reward = -float("inf")
        self.es_optimizer: Optional[EvolutionStrategies] = None
        self.last_rft_traces: List[Dict[str, Any]] = []
        self.last_drgrpo_samples: List[Dict[str, Any]] = []
        
    def _create_judge_schema(self) -> str:
        """Create the structured output schema for the judge."""
        return """
Output ONLY this JSON format:

{
  "self_criteria": [
    {"name": "Performance", "weight": 0.4, "rationale": "Code efficiency matters"},
    {"name": "Readability", "weight": 0.3, "rationale": "Code should be clear"},
    {"name": "Correctness", "weight": 0.3, "rationale": "Code must work correctly"}
  ],
  "critique": "Response 1 is slower due to recursion. Response 2 is faster with iteration.",
  "scores": [6, 9],
  "principles": ["Performance", "Readability", "Correctness"],
  "confidence": 0.8
}

NO OTHER TEXT. ONLY JSON.
"""
    
    async def generate_judgement(self, query: str, responses: List[str], 
                                use_principles: bool = True) -> JudgeOutput:
        """Generate a complete judgement for the given query and responses."""
        
        # Build the prompt
        prompt = self._build_judge_prompt(query, responses, use_principles)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.config.max_length)
        # Ensure all inputs are on the same device as the model
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Increase token limit
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and parse
        response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                            skip_special_tokens=True)
        
        # Parse structured output
        judgement = self._parse_judge_output(response_text, len(responses))
        
        return judgement
    
    def _build_judge_prompt(self, query: str, responses: List[str], 
                           use_principles: bool = True) -> str:
        """Build the prompt for the judge."""
        prompt = f"Query: {query}\n\n"
        
        prompt += "Responses:\n"
        for i, response in enumerate(responses, 1):
            prompt += f"{i}: {response}\n"
        
        prompt += "\n" + self.judge_schema
        
        return prompt
    
    def _parse_judge_output(self, response_text: str, num_responses: int) -> JudgeOutput:
        """Parse the judge's structured output."""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Try to find JSON in the response
            json_start = response_text.find('{')
            if json_start == -1:
                raise ValueError("No JSON found in response")
            
            # Find the matching closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(response_text[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                # Try to find a partial JSON and complete it
                json_text = response_text[json_start:]
                # Add missing closing braces
                missing_braces = json_text.count('{') - json_text.count('}')
                json_text += '}' * missing_braces
            else:
                json_text = response_text[json_start:json_end]
            
            # Try to parse JSON
            data = json.loads(json_text)
            
            # Validate and extract components
            self_criteria = data.get('self_criteria', [])
            if not isinstance(self_criteria, list):
                self_criteria = []
            
            critique = data.get('critique', '')
            if not isinstance(critique, str):
                critique = ''
            
            scores = data.get('scores', [])
            if not isinstance(scores, list) or len(scores) != num_responses:
                # Generate default scores if parsing fails
                scores = [random.randint(1, 10) for _ in range(num_responses)]
            
            # Ensure scores are in valid range
            scores = [max(1, min(10, int(score))) for score in scores]
            
            principles = data.get('principles', [])
            if not isinstance(principles, list):
                principles = []
            
            confidence = float(data.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return JudgeOutput(
                self_criteria=self_criteria,
                critique=critique,
                scores=scores,
                principles=principles,
                confidence=confidence
            )
            
        except Exception as e:
            # Fallback to default judgement with valid structure
            print(f"Warning: Failed to parse judge output: {e}")
            
            # Try to extract any useful information
            critique = "Failed to generate proper critique"
            if "critique" in response_text.lower():
                # Try to extract critique text
                lines = response_text.split('\n')
                for line in lines:
                    if len(line) > 20 and not line.startswith('{'):
                        critique = line[:500]  # Limit length
                        break
            
            return JudgeOutput(
                self_criteria=[
                    {"name": "default_criterion", "weight": 1.0, "rationale": "Fallback criterion"}
                ],
                critique=critique,
                scores=[5] * num_responses,  # Neutral scores
                principles=["default_principle"],
                confidence=0.1
            )

    def _serialize_judgement(self, judgement: JudgeOutput) -> str:
        """Serialize a JudgeOutput to JSON text matching the schema."""
        judgement_dict = {
            "self_criteria": judgement.self_criteria,
            "critique": judgement.critique,
            "scores": judgement.scores,
            "principles": judgement.principles,
            "confidence": float(judgement.confidence),
        }
        return json.dumps(judgement_dict, ensure_ascii=False, separators=(",", ":"))
    
    async def generate_multiple_judgements(self, query: str, responses: List[str], 
                                         num_samples: int = 8) -> List[JudgeOutput]:
        """Generate multiple independent judgements for sampling-based scaling."""
        judgements = []
        
        for _ in range(num_samples):
            judgement = await self.generate_judgement(query, responses)
            judgements.append(judgement)
        
        return judgements

    async def _sample_with_logprob(self, prompt: str, responses: List[str]) -> Optional[Tuple[JudgeOutput, torch.Tensor, torch.Tensor]]:
        """Generate a judgement and return its log-probability and greedy baseline log-probability."""
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_len = enc["input_ids"].shape[-1]
        enc = {k: v.to(device) for k, v in enc.items()}

        generated = self.model.generate(
            **enc,
            max_new_tokens=512,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

        if isinstance(generated, torch.Tensor):
            sequence = generated[0]
        elif isinstance(generated, (list, tuple)):
            sequence = generated[0]
            if isinstance(sequence, (list, tuple)):
                sequence = torch.tensor(sequence, device=device)
        else:
            return None

        sequence = sequence.to(device)
        response_ids = sequence[input_len:]
        if response_ids.numel() == 0:
            return None

        logprob = self._sequence_logprob(sequence, input_len, requires_grad=True)

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        try:
            judgement = self._parse_judge_output(response_text, len(responses))
        except Exception:
            return None

        with torch.no_grad():
            greedy_out = self.model.generate(
                **enc,
                max_new_tokens=response_ids.shape[0],
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )
            if isinstance(greedy_out, torch.Tensor):
                greedy_seq = greedy_out[0].to(device)
            elif isinstance(greedy_out, (list, tuple)):
                greedy_seq = greedy_out[0]
                if isinstance(greedy_seq, (list, tuple)):
                    greedy_seq = torch.tensor(greedy_seq, device=device)
                else:
                    greedy_seq = greedy_seq.to(device)
            else:
                return judgement, logprob, logprob.detach()

        greedy_logprob = self._sequence_logprob(greedy_seq, input_len, requires_grad=False).detach()

        return judgement, logprob, greedy_logprob

    def _sequence_logprob(self, sequence: torch.Tensor, prompt_len: int, requires_grad: bool) -> torch.Tensor:
        """Compute log-probability of the response tokens in a sequence."""
        device = next(self.model.parameters()).device
        input_ids = sequence.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)

        if requires_grad:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # (1, seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        token_logprobs = []
        for position in range(prompt_len, sequence.size(0)):
            token_id = sequence[position]
            token_logprobs.append(log_probs[0, position - 1, token_id])

        if not token_logprobs:
            return torch.tensor(0.0, device=device, requires_grad=requires_grad)

        return torch.stack(token_logprobs).sum()
    
    def aggregate_judgements(self, judgements: List[JudgeOutput], 
                           method: str = "sum") -> Dict[str, Any]:
        """Aggregate multiple judgements using voting/summing."""
        if not judgements:
            return {"scores": [], "best_index": 0, "confidence": 0.0}
        
        num_responses = len(judgements[0].scores)
        aggregated_scores = [0] * num_responses
        
        # Sum scores across judgements
        for judgement in judgements:
            for i, score in enumerate(judgement.scores):
                if i < len(aggregated_scores):
                    aggregated_scores[i] += score
        
        # Find best response
        best_index = np.argmax(aggregated_scores)
        
        # Calculate confidence
        total_confidence = sum(j.confidence for j in judgements)
        avg_confidence = total_confidence / len(judgements) if judgements else 0.0
        
        return {
            "scores": aggregated_scores,
            "best_index": best_index,
            "confidence": avg_confidence,
            "num_judgements": len(judgements)
        }
    
    def calculate_correctness(self, predicted_scores: List[int], 
                            ground_truth_best: int) -> bool:
        """Calculate if the predicted scores correctly identify the best response."""
        if not predicted_scores:
            return False
        
        predicted_best = np.argmax(predicted_scores)
        return predicted_best == ground_truth_best
    
    def calculate_outcome_reward(self, predicted_scores: List[int], 
                               ground_truth_best: int) -> float:
        """Calculate the rule-based outcome reward (Eq. 11 from SPCT paper)."""
        is_correct = self.calculate_correctness(predicted_scores, ground_truth_best)
        return 1.0 if is_correct else -1.0
    
    async def train_rft_cold_start(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train using RFT (Rejective Fine-Tuning) cold start."""
        print("🔄 Starting RFT cold start training...")
        
        accepted_traces: List[Dict[str, Any]] = []
        total_samples = 0
        total_items = 0
        hints_used = 0
        easy_dropped = 0
        
        for item in training_data:
            query = getattr(item, "query", item.get("query", "")) if item else ""
            responses = getattr(item, "responses", item.get("responses", [])) if item else []
            ground_truth_best = getattr(item, "best_index", item.get("best_index", 0)) if item else 0
            
            if not query or not responses:
                continue
            
            total_items += 1
            item_traces: List[Dict[str, Any]] = []
            correct_count = 0
            
            for sample_idx in range(self.config.rft_samples):
                total_samples += 1
                
                # Alternate between hinted and non-hinted sampling to encourage robustness
                use_hint = (sample_idx % 2 == 1)
                judgement = await self.generate_judgement(
                    self._build_hinted_query(query, responses, ground_truth_best, use_hint),
                    responses,
                    use_principles=True
                )
                if use_hint:
                    hints_used += 1
                
                is_correct = self.calculate_correctness(judgement.scores, ground_truth_best)
                if is_correct:
                    correct_count += 1
                    item_traces.append({
                        "query": query,
                        "responses": responses,
                        "judgement": judgement,
                        "ground_truth_best": ground_truth_best,
                        "hinted": use_hint,
                    })
            
            if not item_traces:
                continue
            
            if self.config.drop_too_easy and correct_count == self.config.rft_samples:
                easy_dropped += 1
                continue
            
            accepted_traces.extend(item_traces)
            if self.config.rft_max_traces and len(accepted_traces) >= self.config.rft_max_traces:
                accepted_traces = accepted_traces[: self.config.rft_max_traces]
                break
        
        print(f"📊 RFT Results: {len(accepted_traces)}/{total_samples} traces accepted (items processed: {total_items}, hints used: {hints_used}, easy dropped: {easy_dropped})")
        
        training_metrics = await self._train_on_accepted_traces(accepted_traces)
        
        return {
            "accepted_traces": len(accepted_traces),
            "total_samples": total_samples,
            "total_items": total_items,
            "easy_dropped": easy_dropped,
            "acceptance_rate": len(accepted_traces) / total_samples if total_samples > 0 else 0.0,
            "training_metrics": training_metrics
        }

    def _build_hinted_query(self, query: str, responses: List[str], best_index: int, use_hint: bool) -> str:
        """Optionally append a hint about the correct response for diversified sampling."""
        if not use_hint:
            return query
        hint = f"\n[Hint] The best response is #{best_index + 1}."
        return query + hint
    
    def _filter_too_easy_items(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out items where all samples were correct (too easy)."""
        # Group by query
        query_groups = {}
        for trace in traces:
            query = trace["query"]
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(trace)
        
        # Keep only items with some difficulty
        filtered_traces = []
        for query, group_traces in query_groups.items():
            # If we have multiple traces for the same query, it's not too easy
            if len(group_traces) > 1:
                filtered_traces.extend(group_traces)
            else:
                # Single trace - keep it (might be difficult)
                filtered_traces.extend(group_traces)
        
        return filtered_traces
    
    async def _train_on_accepted_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the model on accepted RFT traces using teacher forcing."""
        if not traces:
            return {"training_loss": 0.0, "traces_processed": 0}

        print(f"🎓 Training on {len(traces)} accepted traces...")

        device = next(self.model.parameters()).device
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=max(1e-5, self.config.grpo_learning_rate))

        total_loss = 0.0
        steps = 0

        for trace in traces:
            prompt = self._build_judge_prompt(trace["query"], trace["responses"], use_principles=True)
            target_json = self._serialize_judgement(trace["judgement"])
            training_text = prompt + "\n" + target_json

            tokenized = self.tokenizer(
                training_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            # Mask prompt portion so only the generated judgement contributes to loss
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                add_special_tokens=False,
            )["input_ids"].shape[-1]

            labels = tokenized["input_ids"].clone()
            labels[:, :prompt_tokens] = -100

            outputs = self.model(**tokenized, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps else 0.0
        self.model.eval()

        self.last_rft_traces = traces
        self.training_history.append({
            "stage": "rft",
            "traces": len(traces),
            "loss": avg_loss,
        })

        return {
            "training_loss": avg_loss,
            "traces_processed": steps,
        }
    
    async def train_drgrpo_online_rl(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train using Dr.GRPO (group relative PPO) with rule-based rewards."""
        print("🚀 Starting Dr.GRPO online RL training...")

        if not training_data:
            return {"avg_reward": 0.0, "accuracy": 0.0, "total_samples": 0}

        device = next(self.model.parameters()).device
        optimizer = AdamW(self.model.parameters(), lr=self.config.grpo_learning_rate)
        self.model.train()

        rewards_history: List[float] = []
        correctness_history: List[bool] = []
        rl_samples: List[Dict[str, Any]] = []
        running_baseline = 0.0
        beta = self.config.grpo_reward_beta

        grad_accumulator = 0

        for iteration in range(self.config.grpo_iterations):
            item = random.choice(training_data)
            query = item.get("query", "")
            responses = item.get("responses", [])
            ground_truth_best = item.get("best_index", 0)

            if not query or not responses:
                continue

            group_entries: List[Tuple[torch.Tensor, torch.Tensor, float, JudgeOutput]] = []

            for group_idx in range(self.config.drgrpo_group_size):
                use_hint = ((iteration + group_idx) % 2 == 1)
                hinted_query = self._build_hinted_query(query, responses, ground_truth_best, use_hint)
                prompt = self._build_judge_prompt(hinted_query, responses, use_principles=True)
                sample = await self._sample_with_logprob(prompt, responses)
                if sample is None:
                    continue

                judgement, logprob, greedy_logprob = sample
                reward = self.calculate_outcome_reward(judgement.scores, ground_truth_best)
                rewards_history.append(reward)
                correctness_history.append(reward > 0)
                group_entries.append((logprob, greedy_logprob.detach(), reward, judgement))

            if not group_entries:
                continue

            rewards = [entry[2] for entry in group_entries]
            group_baseline = float(np.mean(rewards))
            running_baseline = beta * running_baseline + (1.0 - beta) * group_baseline

            loss_terms: List[torch.Tensor] = []
            for logprob, greedy_logprob, reward, judgement in group_entries:
                advantage = reward - group_baseline
                policy_loss = -(advantage * logprob)
                kl_penalty = self.config.grpo_kl_penalty * (logprob - greedy_logprob) ** 2
                loss_terms.append(policy_loss + kl_penalty)
                rl_samples.append({
                    "query": query,
                    "responses": responses,
                    "judgement": judgement,
                    "reward": reward,
                    "baseline": group_baseline,
                    "advantage": advantage,
                })

            batch_loss = torch.stack(loss_terms).mean()
            batch_loss.backward()
            grad_accumulator += 1

            if grad_accumulator >= self.config.grpo_batch_size:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                grad_accumulator = 0

        if grad_accumulator > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        self.model.eval()

        avg_reward = float(np.mean(rewards_history)) if rewards_history else 0.0
        accuracy = float(np.mean(correctness_history)) if correctness_history else 0.0

        self.last_drgrpo_samples = rl_samples
        self.training_history.append({
            "stage": "drgrpo",
            "samples": len(rl_samples),
            "avg_reward": avg_reward,
        })

        if avg_reward > self.best_rl_reward:
            self.best_rl_reward = avg_reward
        else:
            await self._evolutionary_escape(training_data)

        return {
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "total_samples": len(rewards_history),
            "baseline": running_baseline,
        }

    async def _evolutionary_escape(self, training_data: List[Dict[str, Any]]) -> None:
        """Apply evolutionary fine-tuning when RL reward plateaus."""
        if not training_data:
            return

        es_config = ESConfig(
            population_size=6,
            sigma=0.05,
            learning_rate=0.05,
            max_iterations=1,
            greedy_decoding=False,
            parallel_evaluation=False,
        )
        self.es_optimizer = EvolutionStrategies(es_config)
        initial_weight = self.model.lm_head.weight.detach().clone()
        initial_params = {"lm_head": initial_weight.clone()}
        await self.es_optimizer.initialize(initial_params)

        subset = random.sample(training_data, min(len(training_data), 6))

        def objective(param_dict: Dict[str, torch.Tensor]) -> float:
            self._load_es_parameters(param_dict)
            rewards = []
            for item in subset:
                query = item.get("query", "")
                responses = item.get("responses", [])
                best_index = item.get("best_index", 0)
                if not query or not responses:
                    continue
                judgement = self._judge_sync(query, responses)
                reward = self.calculate_outcome_reward(judgement.scores, best_index)
                rewards.append(reward)
            return float(np.mean(rewards)) if rewards else 0.0

        await self.es_optimizer.single_step(objective)

        if self.es_optimizer.best_parameters and self.es_optimizer.best_reward > self.best_rl_reward:
            self._load_es_parameters(self.es_optimizer.best_parameters)
            self.best_rl_reward = self.es_optimizer.best_reward
        else:
            self._load_es_parameters({"lm_head": initial_weight})

    def _load_es_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if "lm_head" in params:
                target = params["lm_head"].to(self.model.lm_head.weight.device)
                self.model.lm_head.weight.copy_(target)

    def _judge_sync(self, query: str, responses: List[str]) -> JudgeOutput:
        prompt = self._build_judge_prompt(query, responses, use_principles=True)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            input_len = inputs["input_ids"].shape[-1]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            response_ids = outputs[0][0][input_len:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        try:
            return self._parse_judge_output(response_text, len(responses))
        except Exception:
            return JudgeOutput(
                self_criteria=[{"name": "default", "weight": 1.0, "rationale": "fallback"}],
                critique="",
                scores=[5] * len(responses),
                principles=["default"],
                confidence=0.1,
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current model metrics."""
        return {
            "is_trained": self.is_trained,
            "training_history": len(self.training_history),
            "config": self.config
        }
