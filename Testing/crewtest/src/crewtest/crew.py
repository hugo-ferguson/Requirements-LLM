from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Crewtest():
    """Crewtest crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    # @agent
    # def researcher(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['researcher'], # type: ignore[index]
    #         verbose=True
    #     )

    # @agent
    # def reporting_analyst(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['reporting_analyst'], # type: ignore[index]
    #         verbose=True
    #     )
    # @tool("secret_calculation_tool")
    # def secret_calculation_tool(question: str) -> str:
    #     """
    #     A tool to calculate a secret number based on two input integers. "
    #     "The tool takes two whitespace-separated integers as input, performs a secret calculation, and returns the result as a string.
    #     """
    #     # Parse the input argument to extract the two integers
    #     try:
    #         num1_str, num2_str = argument.split()
    #         num1 = int(num1_str)
    #         num2 = int(num2_str)
    #     except ValueError:
    #         return "Invalid input format. Please provide two whitespace-separated integers."

    #     # Perform a calculation
    #     result = (num1 - 1)  * (num2 + 1)

    #     # Return the result as a string
    #     return str(result)

    @tool("secret_calculation_tool") # This string MUST match tasks.yaml
    def secret_calculation_tool(self, argument: str) -> str:
        """
        Calculates a secret number from two whitespace-separated integers.
        """
        try:
            num1, num2 = map(int, argument.split())
            return str((num1 - 1) * (num2 + 1))
        except Exception:
            return "Please provide two integers separated by a space."

    @agent
    def random_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['random_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    # @task
    # def research_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['research_task'], # type: ignore[index]
    #     )

    # @task
    # def reporting_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['reporting_task'], # type: ignore[index]
    #         output_file='report.md'
    #     )
    @task
    def random_numbers_task(self) -> Task:
        return Task(
            config=self.tasks_config['random_numbers_task'], # type: ignore[index]
        )
        

    @crew
    def crew(self) -> Crew:
        """Creates the Crewtest crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
