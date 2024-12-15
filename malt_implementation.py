from langchain.prompts import PromptTemplate
#from langchain.llms import OpenAI
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

# Output parsers for structured outputs
class GeneratorOutput(BaseModel):
    reasoning_steps: List[str] = Field(description="Steps taken to reach the solution")
    proposed_solution: str = Field(description="The proposed solution")
    confidence_score: float = Field(description="Confidence in the solution (0-1)")

class VerifierOutput(BaseModel):
    is_correct: bool = Field(description="Whether the solution appears correct")
    issues_found: List[str] = Field(description="List of potential issues identified")
    improvement_suggestions: List[str] = Field(description="Suggested improvements")
    verification_score: float = Field(description="Verification confidence score (0-1)")

class RefinerOutput(BaseModel):
    refined_solution: str = Field(description="The refined solution")
    improvements_made: List[str] = Field(description="List of improvements made")
    final_confidence: float = Field(description="Confidence in final solution (0-1)")

# Initialize parsers
generator_parser = PydanticOutputParser(pydantic_object=GeneratorOutput)
verifier_parser = PydanticOutputParser(pydantic_object=VerifierOutput)
refiner_parser = PydanticOutputParser(pydantic_object=RefinerOutput)

# MALT Agent class
class MALTAgent:
    def __init__(self, role: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.role = role
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory()
        self._setup_prompts()
        self._setup_chains()

    def _setup_prompts(self):
        """Setup role-specific prompts"""
        if self.role == "generator":
            self.prompt = PromptTemplate(
                template="""You are a solution generator. Given a problem, provide a detailed solution.
                Problem: {problem}
                
                Provide output in the following format:
                {format_instructions}
                """,
                input_variables=["problem"],
                partial_variables={"format_instructions": generator_parser.get_format_instructions()}
            )
        
        elif self.role == "verifier":
            self.prompt = PromptTemplate(
                template="""You are a solution verifier. Analyze the proposed solution critically.
                Problem: {problem}
                Proposed Solution: {solution}
                
                Provide output in the following format:
                {format_instructions}
                """,
                input_variables=["problem", "solution"],
                partial_variables={"format_instructions": verifier_parser.get_format_instructions()}
            )
        
        elif self.role == "refiner":
            self.prompt = PromptTemplate(
                template="""You are a solution refiner. Improve the solution based on verification feedback.
                Problem: {problem}
                Original Solution: {solution}
                Verification Feedback: {feedback}
                
                Provide output in the following format:
                {format_instructions}
                """,
                input_variables=["problem", "solution", "feedback"],
                partial_variables={"format_instructions": refiner_parser.get_format_instructions()}
            )

    def _setup_chains(self):
        """Setup LLM chains"""
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            output_key=f"{self.role}_output"
        )

    def process(self, **inputs) -> dict:
        """Process inputs according to role"""
        return self.chain(inputs)

# MALT System class
class MALTSystem:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.generator = MALTAgent("generator", model_name, temperature)
        self.verifier = MALTAgent("verifier", model_name, temperature)
        self.refiner = MALTAgent("refiner", model_name, temperature)
        
        # Setup sequential chain
        self.chain = SequentialChain(
            chains=[
                self.generator.chain,
                self.verifier.chain,
                self.refiner.chain
            ],
            input_variables=["problem"],
            output_variables=["generator_output", "verifier_output", "refiner_output"]
        )
    
    def solve(self, problem: str) -> dict:
        """Solve a problem using the MALT system"""
        # Generate initial solution
        gen_output = self.generator.process(problem=problem)
        gen_solution = generator_parser.parse(gen_output["generator_output"])
        
        # Verify solution
        ver_output = self.verifier.process(
            problem=problem,
            solution=gen_solution.proposed_solution
        )
        verification = verifier_parser.parse(ver_output["verifier_output"])
        
        # Refine solution
        ref_output = self.refiner.process(
            problem=problem,
            solution=gen_solution.proposed_solution,
            feedback=str(verification.dict())
        )
        final_solution = refiner_parser.parse(ref_output["refiner_output"])
        
        return {
            "initial_solution": gen_solution.dict(),
            "verification": verification.dict(),
            "final_solution": final_solution.dict()
        }

# Example usage with value propagation
def calculate_value(outputs: dict, threshold: float = 0.5) -> float:
    """Calculate value of the solution based on confidence scores"""
    gen_conf = outputs["initial_solution"]["confidence_score"]
    ver_conf = outputs["verification"]["verification_score"]
    ref_conf = outputs["final_solution"]["final_confidence"]
    
    # Value propagation similar to paper
    value = (gen_conf + ver_conf + ref_conf) / 3
    return 1.0 if value > threshold else 0.0

# Example usage
def main():
    # Initialize MALT system
    malt = MALTSystem()
    
    # Example problem
    problem = "What is 2 + 2, and why?"
    
    # Solve problem
    outputs = malt.solve(problem)
    
    # Calculate value
    solution_value = calculate_value(outputs)
    
    print(f"Solution value: {solution_value}")
    print("\nDetailed outputs:")
    print(f"Initial solution: {outputs['initial_solution']}")
    print(f"Verification: {outputs['verification']}")
    print(f"Final solution: {outputs['final_solution']}")

if __name__ == "__main__":
    main()



###
# Add at the top of the file
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from google.colab import userdata
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

# Output parsers remain the same
class GeneratorOutput(BaseModel):
    reasoning_steps: List[str] = Field(description="Steps taken to reach the solution")
    proposed_solution: str = Field(description="The proposed solution")
    confidence_score: float = Field(description="Confidence in the solution (0-1)")

class VerifierOutput(BaseModel):
    is_correct: bool = Field(description="Whether the solution appears correct")
    issues_found: List[str] = Field(description="List of potential issues identified")
    improvement_suggestions: List[str] = Field(description="Suggested improvements")
    verification_score: float = Field(description="Verification confidence score (0-1)")

class RefinerOutput(BaseModel):
    refined_solution: str = Field(description="The refined solution")
    improvements_made: List[str] = Field(description="List of improvements made")
    final_confidence: float = Field(description="Confidence in final solution (0-1)")

# Initialize parsers
generator_parser = PydanticOutputParser(pydantic_object=GeneratorOutput)
verifier_parser = PydanticOutputParser(pydantic_object=VerifierOutput)
refiner_parser = PydanticOutputParser(pydantic_object=RefinerOutput)

class MALTAgent:
    def __init__(self, role: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.role = role
        self.llm = ChatGroq(model_name=model_name, temperature=temperature, api_key=userdata.get('groq'))
        self._setup_prompts()
        self._setup_chains()

    def _setup_prompts(self):
        if self.role == "generator":
            self.prompt = PromptTemplate(
                template="""You are a solution generator. Given a problem, provide a detailed solution.
                Problem: {query}
                
                Provide the following:
                1. A list of reasoning steps
                2. The proposed solution
                3. A confidence score between 0 and 1
                
                {format_instructions}
                """,
                input_variables=["query"],
                partial_variables={"format_instructions": generator_parser.get_format_instructions()}
            )
        
        elif self.role == "verifier":
            self.prompt = PromptTemplate(
                template="""You are a solution verifier. Analyze the proposed solution critically.
                Problem: {query}
                Proposed Solution: {proposed_solution}
                
                Verify the solution and provide:
                1. Whether it's correct (true/false)
                2. List any issues found
                3. Suggested improvements
                4. A verification confidence score between 0 and 1
                
                {format_instructions}
                """,
                input_variables=["query", "proposed_solution"],
                partial_variables={"format_instructions": verifier_parser.get_format_instructions()}
            )
        
        elif self.role == "refiner":
            self.prompt = PromptTemplate(
                template="""You are a solution refiner. Improve the solution based on verification feedback.
                Problem: {query}
                Original Solution: {proposed_solution}
                Verification Feedback: {feedback}
                
                Provide:
                1. A refined solution
                2. List of improvements made
                3. Final confidence score between 0 and 1
                
                {format_instructions}
                """,
                input_variables=["query", "proposed_solution", "feedback"],
                partial_variables={"format_instructions": refiner_parser.get_format_instructions()}
            )

    def _setup_chains(self):
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    async def process(self, **inputs):
        """Process inputs based on role"""
        try:
            # Use invoke() instead of run() to handle multiple inputs
            result = await self.chain.ainvoke(inputs)
            return result['text']
        except Exception as e:
            print(f"Error in {self.role} processing: {str(e)}")
            return None

class MALTSystem:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.generator = MALTAgent("generator", model_name, temperature)
        self.verifier = MALTAgent("verifier", model_name, temperature)
        self.refiner = MALTAgent("refiner", model_name, temperature)
    
    async def solve(self, problem: str) -> dict:
        """Solve a problem using the MALT system"""
        try:
            # Step 1: Generate initial solution
            gen_output = await self.generator.process(query=problem)
            if not gen_output:
                return None
            gen_solution = generator_parser.parse(gen_output)
            
            # Step 2: Verify solution
            ver_output = await self.verifier.process(
                query=problem,
                proposed_solution=gen_solution.proposed_solution
            )
            if not ver_output:
                return None
            verification = verifier_parser.parse(ver_output)
            
            # Step 3: Refine solution
            ref_output = await self.refiner.process(
                query=problem,
                proposed_solution=gen_solution.proposed_solution,
                feedback=str(verification.dict())
            )
            if not ref_output:
                return None
            final_solution = refiner_parser.parse(ref_output)
            
            return {
                "initial_solution": gen_solution.dict(),
                "verification": verification.dict(),
                "final_solution": final_solution.dict()
            }
        except Exception as e:
            print(f"Error in MALT processing: {str(e)}")
            return None

def calculate_value(outputs: dict, threshold: float = 0.5) -> float:
    if not outputs:
        return 0.0
    
    gen_conf = outputs["initial_solution"]["confidence_score"]
    ver_conf = outputs["verification"]["verification_score"]
    ref_conf = outputs["final_solution"]["final_confidence"]
    
    value = (gen_conf + ver_conf + ref_conf) / 3
    return 1.0 if value > threshold else 0.0

async def main():
    # Initialize MALT system
    malt = MALTSystem()
    
    # Example problem
    problem = "What is 2 + 2, and why?"
    
    # Solve problem
    outputs = await malt.solve(problem)
    
    if outputs:
        solution_value = calculate_value(outputs)
        print(f"Solution value: {solution_value}")
        print("\nDetailed outputs:")
        print(f"Initial solution: {outputs['initial_solution']}")
        print(f"Verification: {outputs['verification']}")
        print(f"Final solution: {outputs['final_solution']}")
    else:
        print("Failed to generate solution")

if __name__ == "__main__":
    import asyncio
    import nest_asyncio 
    nest_asyncio.apply() # This line is added to allow nested event loops.
    # Get the current event loop or create a new one if none exists
    loop = asyncio.get_event_loop()
    # Run the 'main' coroutine until it completes
    loop.run_until_complete(main())


### Output ###
# Solution value: 1.0

# Detailed outputs:
# Initial solution: {'reasoning_steps': ['Start with the basic arithmetic operation of addition', 'Recall the definition of addition as combining two or more numbers to get a total or a sum', 'Apply this definition to the numbers 2 and 2', 'Use the concept of counting or combining quantities to find the sum', 'Recognize that 2 + 2 is a basic arithmetic fact that is widely known and accepted'], 'proposed_solution': '4', 'confidence_score': 1.0}

# Verification: {'is_correct': True, 'issues_found': ["Lack of explanation for the 'why' part of the question", 'Does not address the underlying mathematical principle'], 'improvement_suggestions': ['Include an explanation of the mathematical principle behind addition', "Address the 'why' part of the question to provide a comprehensive answer"], 'verification_score': 0.8}

# Final solution: {'refined_solution': 'The answer to 2 + 2 is 4 because of the fundamental mathematical principle of addition, which states that the sum of two or more numbers is the total amount obtained when they are combined. In this case, when we add 2 + 2, we are counting the total number of units we have. Since we have 2 units in the first set and 2 units in the second set, the total number of units is 4. This principle is based on the concept of one-to-one correspondence, where each unit in one set corresponds to exactly one unit in the other set, resulting in a total of 4 units.', 'improvements_made': ["Provided a clear explanation for the 'why' part of the question", 'Addressed the underlying mathematical principle of addition', 'Included a description of the concept of one-to-one correspondence', 'Offered a step-by-step breakdown of the addition process'], 'final_confidence': 0.95}
