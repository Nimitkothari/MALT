import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore")

# Output schemas
class EvaluationResult(BaseModel):
    """Schema for LLM evaluation results"""
    correctness_score: float = Field(description="Score between 0 and 1 indicating answer correctness")
    reasoning_score: float = Field(description="Score between 0 and 1 for reasoning quality")
    explanation: List[str] = Field(description="List of reasons for the scores")
    key_matches: List[str] = Field(description="Key concepts that matched between answer and ground truth")

@dataclass
class Trajectory:
    """Class to store a complete solution trajectory"""
    question: str
    generator_output: str
    verifier_output: str
    refiner_output: str
    ground_truth: str
    value: float = 0.0
    generator_value: float = 0.0
    verifier_value: float = 0.0
    evaluation_details: dict = None

class LLMEvaluator:
    """Evaluates answers using LLM"""
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        self.llm = ChatGroq(model_name=model_name, temperature=temperature, api_key = userdata.get('groq'))
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        self._setup_prompt()

    def _setup_prompt(self):
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator. Analyze answers based on correctness and quality of reasoning.
            Score both the factual accuracy and the reasoning quality."""),
            ("human", """Question: {question}
            Ground Truth: {ground_truth}
            Given Answer: {answer}
            
            Evaluate based on:
            1. Correctness (factual accuracy)
            2. Quality of reasoning
            3. Key concept matches
            
            {format_instructions}
            """)
        ])

    async def evaluate_answer(self, question: str, ground_truth: str, answer: str) -> EvaluationResult:
        """Evaluate a single answer"""
        messages = self.eval_prompt.format_messages(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            format_instructions=self.parser.get_format_instructions()
        )
        
        response = await self.llm.agenerate([messages])
        return self.parser.parse(response.generations[0][0].text)

class TrainingDataGenerator:
    """Main class for generating MALT training data"""
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        branching_factor: int = 2,
        use_llm_eval: bool = True
    ):
        self.llm = ChatGroq(model_name=model_name, temperature=temperature, api_key = userdata.get('groq'))
        self.n = branching_factor
        self.evaluator = LLMEvaluator(model_name=model_name) if use_llm_eval else None
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompts for each agent"""
        self.generator_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a solution generator. Provide detailed solutions for problems."),
            ("human", "{question}")
        ])

        self.verifier_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a solution verifier. Analyze solutions critically."),
            ("human", """Question: {question}
            Proposed Solution: {solution}
            
            Verify this solution and provide detailed feedback.""")
        ])

        self.refiner_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a solution refiner. Improve solutions based on feedback."),
            ("human", """Question: {question}
            Original Solution: {solution}
            Verification Feedback: {feedback}
            
            Provide an improved solution.""")
        ])

    async def generate_single_output(self, prompt, **kwargs) -> str:
        """Generate a single output from the LLM"""
        messages = prompt.format_messages(**kwargs)
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text

    async def generate_trajectories(self, question: str, ground_truth: str) -> List[Trajectory]:
        """Generate multiple solution trajectories"""
        trajectories = []
        
        # Generate initial solutions
        generator_outputs = []
        for _ in range(self.n):
            gen_output = await self.generate_single_output(
                self.generator_prompt,
                question=question
            )
            generator_outputs.append(gen_output)

        # Generate verifier outputs
        for gen_output in generator_outputs:
            verifier_outputs = []
            for _ in range(self.n):
                ver_output = await self.generate_single_output(
                    self.verifier_prompt,
                    question=question,
                    solution=gen_output
                )
                verifier_outputs.append(ver_output)

            # Generate refinements
            for ver_output in verifier_outputs:
                for _ in range(self.n):
                    ref_output = await self.generate_single_output(
                        self.refiner_prompt,
                        question=question,
                        solution=gen_output,
                        feedback=ver_output
                    )

                    trajectories.append(Trajectory(
                        question=question,
                        generator_output=gen_output,
                        verifier_output=ver_output,
                        refiner_output=ref_output,
                        ground_truth=ground_truth
                    ))

        return trajectories

    async def evaluate_trajectory(self, trajectory: Trajectory) -> Dict:
        """Evaluate a complete trajectory using LLM"""
        if not self.evaluator:
            return self._simple_evaluate(trajectory)
            
        generator_eval = await self.evaluator.evaluate_answer(
            trajectory.question,
            trajectory.ground_truth,
            trajectory.generator_output
        )
        
        verifier_eval = await self.evaluator.evaluate_answer(
            trajectory.question,
            trajectory.ground_truth,
            trajectory.verifier_output
        )
        
        refiner_eval = await self.evaluator.evaluate_answer(
            trajectory.question,
            trajectory.ground_truth,
            trajectory.refiner_output
        )
        
        return {
            "generator": generator_eval.model_dump(),
            "verifier": verifier_eval.model_dump(),
            "refiner": refiner_eval.model_dump()
        }

    def _simple_evaluate(self, trajectory: Trajectory) -> Dict:
        """Simple string-based evaluation"""
        def simple_score(text: str, ground_truth: str) -> float:
            text = text.strip().lower()
            ground_truth = ground_truth.strip().lower()
            if text == ground_truth:
                return 1.0
            elif ground_truth in text:
                return 0.8
            return 0.0

        return {
            "generator": {"correctness_score": simple_score(trajectory.generator_output, trajectory.ground_truth)},
            "verifier": {"correctness_score": simple_score(trajectory.verifier_output, trajectory.ground_truth)},
            "refiner": {"correctness_score": simple_score(trajectory.refiner_output, trajectory.ground_truth)}
        }

    async def compute_values(self, trajectories: List[Trajectory]):
        """Compute values for all trajectories"""
        print("\nEvaluating trajectories...")
        for trajectory in tqdm(trajectories, desc="Evaluating"):
            eval_results = await self.evaluate_trajectory(trajectory)
            
            trajectory.generator_value = eval_results["generator"]["correctness_score"]
            trajectory.verifier_value = eval_results["verifier"]["correctness_score"]
            trajectory.value = eval_results["refiner"]["correctness_score"]
            trajectory.evaluation_details = eval_results

    def create_training_datasets(self, trajectories: List[Trajectory]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create training datasets for each agent"""
        generator_data = []
        verifier_data = []
        refiner_data = []

        for t in trajectories:
            # Generator data
            if t.generator_value > 0.5:
                generator_data.append({
                    "question": t.question,
                    "solution": t.generator_output,
                    "ground_truth": t.ground_truth,
                    "score": float(t.generator_value),
                    "evaluation": t.evaluation_details["generator"]
                })

            # Verifier data
            verifier_data.append({
                "question": t.question,
                "solution": t.generator_output,
                "feedback": t.verifier_output,
                "score": float(t.verifier_value),
                "evaluation": t.evaluation_details["verifier"]
            })

            # Refiner data
            if t.value > 0.5:
                refiner_data.append({
                    "question": t.question,
                    "original_solution": t.generator_output,
                    "feedback": t.verifier_output,
                    "refined_solution": t.refiner_output,
                    "score": float(t.value),
                    "evaluation": t.evaluation_details["refiner"]
                })

        return generator_data, verifier_data, refiner_data

async def main():
    # Example training data
    training_data = [
        {
            "question": "What is the capital of France and why is it significant?",
            "ground_truth": "Paris is the capital of France. It is significant as a global center of art, culture, and history."
        },
        {
            "question": "Explain why water boils at 100 degrees Celsius at sea level.",
            "ground_truth": "Water boils at 100°C at sea level because at this temperature, the vapor pressure equals atmospheric pressure."
        }
    ]

    # Initialize generator
    generator = TrainingDataGenerator(
        branching_factor=2,
        use_llm_eval=True
    )
    
    all_trajectories = []
    
    # Generate and evaluate trajectories
    for item in tqdm(training_data, desc="Generating trajectories"):
        try:
            trajectories = await generator.generate_trajectories(
                item["question"],
                item["ground_truth"]
            )
            await generator.compute_values(trajectories)
            all_trajectories.extend(trajectories)
        except Exception as e:
            print(f"Error processing question '{item['question']}': {str(e)}")

    # Create datasets
    gen_data, ver_data, ref_data = generator.create_training_datasets(all_trajectories)
    
    # Save datasets with metadata
    for name, data in [
        ("generator_training.json", gen_data),
        ("verifier_training.json", ver_data),
        ("refiner_training.json", ref_data)
    ]:
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "llama-3.3-70b-versatile",
                "total_examples": len(data)
            },
            "examples": data
        }
        
        with open(name, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(data)} examples to {name}")

if __name__ == "__main__":
    import asyncio
    import warnings
    warnings.filterwarnings("ignore")
    import nest_asyncio 
    nest_asyncio.apply() # This line is added to allow nested event loops.
    # Get the current event loop or create a new one if none exists
    loop = asyncio.get_event_loop()
    # Run the 'main' coroutine until it completes
    loop.run_until_complete(main())