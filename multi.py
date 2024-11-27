import asyncio
import logging
import os
import re
from typing import List, Dict, Any, Optional
from langchain.schema import AIMessage
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
# Data Models
class AgentPlan(BaseModel):
    agent_id: str
    steps: List[Dict[str, Any]]
    confidence_score: Optional[float] = None


class PlanningState(BaseModel):
    user_input: str
    num_agents: int = 3
    agent_plans: List[AgentPlan] = []
    evaluated_plan: Optional[AgentPlan] = None
    final_plan: Optional[Dict[str, Any]] = None


class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id: str, action: callable = None):
        self.nodes[node_id] = action

    def add_edge(self, from_node: str, to_node: str):
        self.edges.append((from_node, to_node))

    def display_graph(self):
        for edge in self.edges:
            logger.info(f"{edge[0]} -> {edge[1]}")
        logger.info("Graph visualization logic can be extended here.")


class RequirementsCollector:
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        self.template = ChatPromptTemplate.from_messages([
            ("system", """
                You are a highly skilled agent responsible for gathering detailed requirements about the user's task. 
                Your goal is to fully understand the task by asking a series of essential questions, evaluating its feasibility, 
                and exploring the core components of the task. The questions must include, but are not limited to, the following:
                
                1. What is the task? Describe it clearly and in detail.
                2. Where is the task to be carried out? Is location relevant?
                3. Why is this task important? What is the motivation behind it?
                4. What other factors should be considered? Is there anything else that might affect this task?
                5. How will this task be completed? What steps are involved?
                6. Evaluate whether the task is feasible. Are there any limitations or resources required that could affect its execution?
                
                Your process should include:
                - Asking the user for clarification when answers are vague or incomplete.
                - Evaluating the feasibility of the task from multiple angles (e.g., resources, timeline, risks).
                - Guiding the user through clarifying details if they do not understand certain questions or if they say "Don't know".
                - Ensuring all questions are answered in sufficient detail before concluding the requirements-gathering session.
                - Once all essential questions are answered and feasibility is assessed, conclude with "REQUIREMENTS GATHERED."

                Remember: Do not proceed without ensuring that every question is thoroughly answered and evaluated.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.history_store = {}

    def _get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    async def collect_requirements(self) -> str:
        session_id = "requirements_session"
        chain = RunnableWithMessageHistory(
            runnable=self.template | self.llm,
            get_session_history=lambda: self._get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history"
        )

        collected_info = []
        print("Hello! What task would you like help planning?")

        while True:
            user_input = input("User: ").strip()
            response = await chain.ainvoke({"input": user_input})
            print(f"Requirements Collector: {response.content}")
            collected_info.append(response.content)

            # If the agent signals that the requirements are gathered, break the loop.
            if "REQUIREMENTS GATHERED" in response.content:
                return "\n".join(collected_info)


class MultiAgentPlanner:
    def __init__(self, llm_model: str = "gpt-4", num_agents: int = 3):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.5)
        self.requirements_collector = RequirementsCollector(llm_model)
        self.num_agents = num_agents

    async def generate_prompt_for_agents(self, requirements: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([(
            "system", f"""
                You are the Prompt Generator Agent. Based on the requirements provided:
                {requirements}
                Generate a detailed prompt for planning agents to create actionable, comprehensive plans. Include:
                - Clear steps
                - Risks and mitigations
                - Logical sequences
            """)
        ])
        response = await self.llm.ainvoke(prompt_template.format())
        return response.content

    async def generate_agent_plan(self, agent_id: str, prompt: str) -> AgentPlan:
        agent_prompt = (
            f"You are Agent {agent_id}. Develop a comprehensive, detailed plan "
            f"for the following task: {prompt}\n\n"
            "Provide your plan with clear, actionable steps. "
            "Include specific details, potential challenges, "
            "and the reasoning behind each step."
        )

        response = await self.llm.ainvoke(agent_prompt)

        # Check if the response is an AIMessage and extract its content
        if isinstance(response, AIMessage):
            response_text = response.content  # Extract the string content from AIMessage
        else:
            response_text = str(response)  # Fallback for unexpected types              

        # Now split the response_text as it is guaranteed to be a string
        steps = [
            {"step": i + 1, "description": step.strip()}
            for i, step in enumerate(response_text.split("\n"))
            if step.strip()
        ]

        # Ensure arguments are passed as keyword arguments
        return AgentPlan(agent_id=agent_id, steps=steps)

    async def evaluate_plans(self, agent_plans: List[AgentPlan]) -> AgentPlan:
        eval_prompt = ChatPromptTemplate.from_messages([(
            "system", """
                You are the Evaluation Agent. Compare the following plans:
                Assign a confidence score (0-1) and select the best plan. Provide reasons for your selection.
            """),
            ("human", "\n\n".join([  # List plans for the AI to evaluate
                f"Plan {plan.agent_id}:\n" + "\n".join([f"Step {step['step']}: {step['description']}" for step in plan.steps])
                for plan in agent_plans
            ]))
        ])

        # Send the evaluation request to the LLM
        response = await self.llm.ainvoke(eval_prompt.format())

        # Log the full response to inspect its structure
        logger.info(f"Evaluation Response: {response}")

        # Ensure response is an AIMessage, then access its content
        if isinstance(response, AIMessage):
            response_text = response.content  # Get string content from the AIMessage
        else:
            response_text = str(response)  # If it's not an AIMessage, treat it as a string

        # Log the raw response for debugging
        logger.info(f"Raw response text: {response_text}")

        # Use regex to extract SELECTED_PLAN
        selected_plan_match = re.search(r"SELECTED_PLAN:\s*([^\n]+)", response_text)
        if not selected_plan_match:
            logger.error("SELECTED_PLAN section not found in response.")
            logger.error(f"Response received: {response_text}")
            raise ValueError("Unable to extract selected plan from response.")

        selected_agent_id = selected_plan_match.group(1).strip()

        # Use regex to extract CONFIDENCE_SCORE
        confidence_score_match = re.search(r"CONFIDENCE_SCORE:\s*([\d.]+)", response_text)
        if not confidence_score_match:
            logger.error("CONFIDENCE_SCORE section not found in response.")
            logger.error(f"Response received: {response_text}")
            raise ValueError("Unable to extract confidence score from response.")

        confidence_score = float(confidence_score_match.group(1))

        # Find the selected plan in the list of agent plans
        selected_plan = next(
            (plan for plan in agent_plans if plan.agent_id == selected_agent_id),
            None
        )
        if not selected_plan:
            logger.error(f"Agent ID '{selected_agent_id}' not found in agent plans.")
            raise ValueError("Unable to match selected plan to an agent.")

        selected_plan.confidence_score = confidence_score
        logger.info(f"Selected Plan: {selected_plan}")
        return selected_plan

    async def automate_plan(self, best_plan: AgentPlan) -> AgentPlan:
        agent_plans_str = "\n".join([  # Prepare the plan string
            f"Plan {plan.agent_id}:\n" + "\n".join([f"Step {step['step']}: {step['description']}" for step in plan.steps])
            for plan in best_plan.steps
        ])
        automate_prompt = ChatPromptTemplate.from_messages([(
            "system", f"""
                You are the Automation Agent. Automate the steps in the following plan:
                Plan {best_plan.agent_id}:\n
                {best_plan.steps}
                For each plan, provide a detailed explanation of how to implement each step. For each step, include:
                1. A solution or action to automate the step.
                2. Any tools, systems, or resources required.
                3. Potential challenges and how to overcome them.
            """)
        ])
        response = await self.llm.ainvoke(automate_prompt.format(agent_plans=agent_plans_str))
        automated_steps = [{"step": i + 1, "description": step.strip()} for i, step in enumerate(response.split("\n")) if step.strip()]
        return AgentPlan(agent_id=best_plan.agent_id, steps=automated_steps)

    def create_state_graph(self, agent_plans: List[AgentPlan]) -> StateGraph:
        state_graph = StateGraph()
        for plan in agent_plans:
            state_graph.add_node(plan.agent_id)
            for step in plan.steps:
                state_graph.add_edge(plan.agent_id, step['description'])
        return state_graph

    async def run_planner(self):
        requirements = await self.requirements_collector.collect_requirements()
        plan_prompt = await self.generate_prompt_for_agents(requirements)
        agent_plans = [await self.generate_agent_plan(f"Agent-{i}", plan_prompt) for i in range(self.num_agents)]
        best_plan = await self.evaluate_plans(agent_plans)
        state_graph = self.create_state_graph(agent_plans)
        logger.info(f"Generated State Graph: {state_graph.display_graph()}")
        automated_plan = await self.automate_plan(best_plan)
        return automated_plan

if __name__ == "__main__":
    asyncio.run(MultiAgentPlanner().run_planner())
