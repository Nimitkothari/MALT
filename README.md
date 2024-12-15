# MALT: Multi-Agent LLM Training Implementation

## Overview
This repository implements the MALT (Multi-Agent LLM Training) framework for creating collaborative AI systems. MALT enables effective collaboration among Language Models through specialized roles and joint training.

## Key Features
- Multi-agent setup with specialized roles (Generator, Verifier, Refiner)
- LLM-based evaluation system
- Training data generation pipeline
- Asynchronous processing
- Detailed evaluation metrics

## System Architecture

### 1. Agent Roles
- **Generator**: Creates initial solutions for given problems
- **Verifier**: Critically analyzes proposed solutions
- **Refiner**: Improves solutions based on verification feedback

### 2. Main Components

#### `malt_implementation.py` & `malt_training_data_generation.py`
Main implementation file containing:
- TrainingDataGenerator class
- Multi-agent interaction logic
- Trajectory generation
- Dataset creation

```python
class TrainingDataGenerator:
    def __init__(self, model_name="model_name", temperature=0.3, branching_factor=2):
        # Initialize MALT system
        ...

    async def generate_trajectories(self, question, ground_truth):
        # Generate solution trajectories
        ...
```
LLM-based evaluation system:
- Answer correctness assessment
- Reasoning quality evaluation
- Detailed feedback generation

```python
class LLMEvaluator:
    def __init__(self, model_name="model_name", temperature=0.1):
        # Initialize evaluator
        ...

    async def evaluate_answer(self, question, ground_truth, answer):
        # Evaluate using LLM
        ...
```

## Training Data Generation

### Process Flow
1. **Initial Generation**:
   - Generator creates multiple solution attempts
   - Branching factor determines number of attempts
   - Each solution is structured and detailed

2. **Verification Phase**:
   - Verifier analyzes each generated solution
   - Provides critical feedback
   - Identifies potential improvements

3. **Refinement Stage**:
   - Refiner processes verification feedback
   - Improves initial solutions
   - Produces final refined answers

4. **Evaluation**:
   - LLM evaluator assesses all outputs
   - Computes quality scores
   - Generates detailed feedback

### Data Structure
Generated datasets are saved in JSON format:
```json
{
    "metadata": {
        "timestamp": "...",
        "model": "model_name",
        "total_examples": 100
    },
    "examples": [
        {
            "question": "...",
            "solution": "...",
            "score": 0.85,
            "evaluation": {
                "correctness_score": 0.9,
                "reasoning_score": 0.8,
                "explanation": ["..."],
                "key_matches": ["..."]
            }
        }
        // ... more examples
    ]
}
```

## Usage

### Prerequisites
```bash
pip install langchain_groq tqdm pydantic
```

### Basic Usage
```python
import asyncio
from malt_implementation import TrainingDataGenerator

# Initialize
generator = TrainingDataGenerator(
    model_name="model_name",
    branching_factor=2,
    use_llm_eval=True
)

# Generate training data
async def generate_data():
    training_data = [
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France."
        }
        # Add more examples...
    ]
    
    await generator.generate_and_evaluate(training_data)

# Run
asyncio.run(generate_data())
```

## Output Files
The system generates three dataset files:
1. `generator_training.json`: Training data for the generator
2. `verifier_training.json`: Training data for the verifier
3. `refiner_training.json`: Training data for the refiner

Each file contains evaluated examples with quality scores and detailed feedback.

## Customization

### Modifying Evaluation Criteria
Adjust the LLMEvaluator class to modify evaluation criteria:
```python
class CustomEvaluator(LLMEvaluator):
    def _setup_prompt(self):
        # Customize evaluation prompt
        self.eval_prompt = ChatPromptTemplate.from_messages([...])
```

### Adjusting Generation Parameters
Modify generation settings through TrainingDataGenerator:
```python
generator = TrainingDataGenerator(
    temperature=0.5,  # Increase creativity
    branching_factor=3,  # More solution attempts
    use_llm_eval=True  # Enable LLM evaluation
)
```

## Limitations and Considerations
- Requires GROQ API key
- Processing time increases with branching factor
- API costs scale with number of evaluations
- Memory usage grows with dataset size

## Future Improvements
- Implement batch processing for larger datasets
- Add support for multiple LLM providers
- Enhance evaluation metrics
- Implement parallel processing
- Add support for different task types

## Citation

```bibtex
@article{malt2024,
    title={MALT: Improving Reasoning with Multi-Agent LLM Training},
    author={Motwani et al.},
    year={2024}
}
```

## License
MIT License