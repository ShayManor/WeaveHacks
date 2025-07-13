import os

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional, Type


class DynamicTool(BaseTool):
    """Wrapper to convert your existing tools to CrewAI tools"""
    name: str = "dynamic_tool"
    description: str = "A dynamically loaded tool"
    execute_func: Any = None

    def _run(self, **kwargs) -> str:
        """Execute the wrapped tool function"""
        if self.execute_func:
            return str(self.execute_func(**kwargs))
        return "Tool not properly initialized"


class HomeMateCrewIntegration:
    def __init__(self, homemate_agent):
        self.homemate_agent = homemate_agent
        self.crews = {}
        self.agents = {}
        self.crewai_tools = self._convert_tools_to_crewai()

    def _convert_tools_to_crewai(self) -> Dict[str, DynamicTool]:
        """Convert existing HomeMate tools to CrewAI tools"""
        crewai_tools = {}

        for tool_name, tool_func in self.homemate_agent.tools.items():
            # Get tool description from docstring or default
            description = tool_func.__doc__ or f"Execute {tool_name} action"

            crewai_tool = DynamicTool(
                name=tool_name,
                description=description.strip(),
                execute_func=tool_func
            )
            crewai_tools[tool_name] = crewai_tool

        return crewai_tools

    def create_agent(self, config: Dict) -> Agent:
        """Dynamically create a CrewAI agent"""
        # Get tools for this agent
        tool_names = config.get("tools", [])
        agent_tools = [self.crewai_tools[name] for name in tool_names if name in self.crewai_tools]

        agent = Agent(
            role=config.get("role", "Assistant"),
            goal=config.get("goal", "Help the user"),
            backstory=config.get("backstory", "I am a helpful AI assistant"),
            tools=agent_tools,
            verbose=True,
            allow_delegation=config.get("allow_delegation", False),
            llm=self._get_llm_config()
        )

        # Store agent for later reference
        agent_id = config.get("id", f"agent_{len(self.agents)}")
        self.agents[agent_id] = agent

        return agent

    def _get_llm_config(self):
        """Get LLM configuration for CrewAI agents"""
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=self.homemate_agent.model,
            api_key=os.getenv("ANTHROPIC_API_KEY")
            )

    def create_crew(self, config: Dict) -> Crew:
        """Dynamically create a crew with agents and tasks"""
        crew_id = config.get("id", f"crew_{len(self.crews)}")

        # Create agents for this crew
        agents = []
        for agent_config in config.get("agents", []):
            agent = self.create_agent(agent_config)
            agents.append(agent)

        # Create tasks
        tasks = []
        for task_config in config.get("tasks", []):
            # Find the agent for this task
            agent_id = task_config.get("agent_id")
            agent = self.agents.get(agent_id, agents[0] if agents else None)

            task = Task(
                description=task_config.get("description", ""),
                expected_output=task_config.get("expected_output", ""),
                agent=agent,
                tools=task_config.get("tools", [])
            )
            tasks.append(task)

        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential if config.get("sequential", True) else Process.hierarchical,
            verbose=True
        )

        self.crews[crew_id] = crew
        return crew

    def execute_crew(self, crew_id: str, inputs: Dict = None) -> str:
        """Execute a crew and return results"""
        if crew_id not in self.crews:
            return f"Crew {crew_id} not found"

        crew = self.crews[crew_id]
        result = crew.kickoff(inputs=inputs or {})
        return str(result)