import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Data Models
class AgentPlan(BaseModel):
    """Represents a plan from an individual agent"""
    agent_id: str
    steps: List[Dict[str, Any]]
    confidence_score: Optional[float] = None

class PlanningState(BaseModel):
    """State for the multi-agent planning process"""
    user_input: str
    num_agents: int = 3
    agent_plans: List[AgentPlan] = []
    evaluated_plan: Optional[AgentPlan] = None
    final_plan: Optional[Dict[str, Any]] = None

class RequirementsCollector:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        self.template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a data collection agent.Ask your questions one by one and proceed with the next question only if the user has answered the previous question. Your task is to gather detailed information from users about their requirements needed by you to implement and automate the task by yourself without human intervention. Ask relevant questions that will be needed by the automation system to automate and implement the tasks.

            1. *Automation Goals and Scope*: What specific tasks or processes do you want the automation system to handle? What are the main goals of automation (e.g., increasing efficiency, reducing manual errors, scaling processes)?
            2. *Infrastructure Requirements*: What type of infrastructure is required for the automation system? (e.g., cloud vs. on-premises, required servers, storage, network considerations)
            3. *Automation Tools and Technologies*: Are there specific tools, frameworks, or technologies you plan to use for automation (e.g., Jenkins, Kubernetes, Zapier, custom scripts)?
            4. *Know about the Existing Systems*: Collect information about the existing system.
            5. *Scalability*: Do you foresee the need to scale the automation system in the future? If so, what are your scalability requirements (e.g., handling more data, supporting more users, etc.)?
            6. *Security and Compliance*: Are there any specific security or compliance requirements that need to be considered (e.g., data encryption, GDPR compliance)?
            8. *Timeline and Deadlines*: What is the timeline for completing the task? Are there any specific deadlines or phases to complete?
            9. *Resources and Personnel*: Who must be alerted if the automation system fails?
            10. *Limitations and constraints*: Check if the user has any limitations or constraints?(eg. budget)
            11. *Additional Details*: Is there anything else that might impact the planning or implementation of the automation system that hasn’t been covered in the questions above?
             
            12. Apart from these ask as many as questions possible relevant to the context to give an apt solution.
             

            Continue asking questions until you have comprehensive information. Once all necessary details are collected, end with 'REQUIREMENTS GATHERED'.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    async def collect_requirements(self) -> str:
        store = {}
        session_id = "requirements_session"
        
        chain = self.template | self.llm
        runnable = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=lambda: self._get_session_history(session_id, store),
            input_messages_key="input",
            history_messages_key="history"
        )

        collected_info = []
        print("Hello! I'll help gather requirements for your task. What would you like to plan?")
        
        while True:
            user_input = input()
            response = await runnable.ainvoke({"input": user_input})
            print(f"response:{response.content}")
            collected_info.append(response.content)
            
            if "REQUIREMENTS GATHERED" in response.content:
                return "\n".join(collected_info)

    def _get_session_history(self, session_id: str, store):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

class MultiAgentPlanner:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.requirements_collector = RequirementsCollector(llm_model)

    async def generate_prompt_for_agents(self, requirements: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the Prompt Generation Agent. Create a comprehensive prompt for guiding the planning agents based on the collected requirements.
            Your prompt is used by the planning agents to generate a plan. Please make sure it is clear, concise, and includes all necessary details needed by it if it is an automation agent to implement and automate the tasks by itself.
            Structure the prompt to help planning agents create detailed, actionable plans about how it is going to implement and automate the task by itself.
            Your prompt must also make the planning agents say the limitations of the it in automating the task and the potential risks involved in automating it.
            Based on the task description and details, generate a detailed prompt to instruct the Planning Agent to say how it is going to acheive the task by itself and what are all the credentials and details it needs from the user for acheiving the task by itself without human intervention.
            Please include any tools, platforms, or constraints mentioned by the user in the task prompt.
            If the user specified human intervention, include that in the prompt. Clarify the points where human intervention is required.
            Ensure that the generated prompt clearly communicates the desired outcome and the steps needed to automate the task.Your Prompt must consider the requirements specified by the user.
            """),
            ("human", f"Requirements:\n{requirements}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        print(f"Generated planning prompt based on requirements")
        return response.content

    async def generate_agent_plan(self, agent_id: str, prompt: str) -> AgentPlan:
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are {agent_id},You are going to automate the task given by the user and implement it by yourself..say me whatever access or credentials and sources you need from me to carry on the task by yourself.

            General Planning:
            Based on the provided prompt, outline how you are going to automate the user’s task.Keep in mind that you are not instructing the user to do something, you are going to do the task by yourself, so you are just informing the user about how are you going to do the work by yourself step by step and what you require from the user for doing your work.
            Break down the task into smaller automatable steps.
            Identify if there are any dependencies or specific conditions where automation may fail and how you are going to handle these failures.
            Consider the tools, platforms, or software mentioned by the user in his requirements.
          Check the automation feasibility, perform human intervention check,tell about how you are going to automate the process and Specify tools, permissions and access needed by you, and expected results for each step.
            For each action, clarify the technical approach (e.g., API calls, script execution, UI interaction).
                       """),
            ("human", f"Generated Prompt: {prompt}")
        ])
        
        chain = agent_prompt | self.llm
        response = await chain.ainvoke({})
        
        plan = AgentPlan(
            agent_id=agent_id,
            steps=[{"step": i+1, "description": step.strip()} 
                   for i, step in enumerate(response.content.split("\n")) 
                   if step.strip()]
        )
        
        print(f"\n{agent_id} Proposal:")
        for step in plan.steps:
            print(f"   Step {step['step']}: {step['description']}")
        print()
        
        return plan

    async def evaluate_plans(self, plans: List[AgentPlan]) -> AgentPlan:
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the Evaluation Agent. Review the provided plans and:
            1. Assign a confidence score (0-1) to each plan and give a summary of confidence score assgnes to each plan.
            2. Select the best plan based on comprehensiveness, feasibility, and alignment with requirements..these are the requirements specified by the user.
            3.Identify if any of the requirements is violated or not considered by the plans, then mention it. 
            4. Provide reasoning for your selection
            
            Format your response as:
            SELECTED_PLAN: [Agent ID]
            CONFIDENCE_SCORE: [Score]
            REASONING: [Your detailed explanation]
            """),
            ("human", "\n\n".join([
                f"Plan {plan.agent_id}:\n" + 
                "\n".join([f"Step {step['step']}: {step['description']}" 
                          for step in plan.steps])
                for plan in plans
            ]))
        ])
        
        chain = evaluation_prompt | self.llm
        response = await chain.ainvoke({})
        print(f"\n:scales: Evaluation Agent:\n{response.content}\n")
        
        selected_plan = next(
            plan for plan in plans 
            if plan.agent_id in response.content.split("SELECTED_PLAN:")[1].split("\n")[0]
        )
        selected_plan.confidence_score = float(
            response.content.split("CONFIDENCE_SCORE:")[1].split("\n")[0]
        )
        
        return selected_plan

    async def create_planning_graph(self, final_plan: AgentPlan):
        graph = StateGraph(dict)
        
        for step in final_plan.steps:
            graph.add_node(str(step['step']), 
                          lambda state, step=step: self._execute_step(state, step))
        
        steps = final_plan.steps
        for i in range(len(steps) - 1):
            graph.add_edge(str(steps[i]['step']), str(steps[i+1]['step']))
        
        graph.add_edge(START, str(steps[0]['step']))
        graph.add_edge(str(steps[-1]['step']), END)
        
        return graph.compile()

    def _execute_step(self, state: Dict, step: Dict[str, Any]) -> Dict:
        logger.info(f"Executing step {step['step']}: {step['description']}")
        return state

    async def main(self):
        # Collect requirements
        requirements = await self.requirements_collector.collect_requirements()
        
        # Generate prompt for planning agents
        planning_prompt = await self.generate_prompt_for_agents(requirements)
        
        # Generate plans from multiple agents
        agent_plans = await asyncio.gather(*[
            self.generate_agent_plan(f"Agent_{i}", planning_prompt)
            for i in range(3)
        ])
        
        # Evaluate and select best plan
        best_plan = await self.evaluate_plans(agent_plans)
        
        print("\nSelected Plan")
        print(f"Agent: {best_plan.agent_id}")
        print(f"Confidence Score: {best_plan.confidence_score}\n")
        for step in best_plan.steps:
            print(f"Step {step['step']}: {step['description']}")

async def run_planner():
    planner = MultiAgentPlanner()
    await planner.main()

if __name__ == "__main__":
    asyncio.run(run_planner())