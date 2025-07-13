import json
from typing import Dict, Any

# Global crew integration instance (will be set by HomeMateAgent)
crew_integration = None


def execute(**kwargs) -> str:
    """Manage CrewAI agents and crews dynamically.

    Actions:
      - create_agent: add an agent with role, goal, tools
      Ex:
      result = execute(
        action="create_agent",
        config={
            "role": "Scheduler",
            "goal": "Keep the team’s calendar up to date",
            "tools": ["calendar", "email"]
        }
      )
      - create_crew:  group agents into a crew with tasks
      Ex:
      result = execute(
        action="create_crew",
        config={
            "id": "event_planning_crew",
            "agents": ["Scheduler", "Notifier"],
            "tasks": ["gather_availability", "send_invitations"]
        }
      )
      - execute_crew: run a crew by ID with given inputs
      - create_tool:  register a new tool module at runtime
    Returns:
      Status message or new identifier.
    """
    action = kwargs.get("action", "")
    config = kwargs.get("config", {})

    if not crew_integration:
        return "Error: CrewAI integration not initialized"

    try:
        if action == "create_agent":
            agent = crew_integration.create_agent(config)
            return f"Created agent: {config.get('role', 'Assistant')}"

        elif action == "create_crew":
            crew = crew_integration.create_crew(config)
            return f"Created crew: {config.get('id', 'new_crew')}"

        elif action == "execute_crew":
            crew_id = config.get("crew_id")
            inputs = config.get("inputs", {})
            result = crew_integration.execute_crew(crew_id, inputs)
            return f"Crew execution result: {result}"

        elif action == "create_tool":
            # Dynamically create a new tool
            tool_name = config.get("name")
            tool_code = config.get("code")

            # Save the tool code to a file
            with open(f"services/{tool_name}.py", "w") as f:
                f.write(tool_code)

            # Reload tools
            crew_integration.homemate_agent.tools = crew_integration.homemate_agent._load_tools()
            crew_integration.crewai_tools = crew_integration._convert_tools_to_crewai()

            return f"Created and loaded new tool: {tool_name}"

        else:
            return f"Unknown action: {action}"

    except Exception as e:
        return f"Error: {str(e)}"