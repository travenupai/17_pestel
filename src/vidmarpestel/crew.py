from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai_tools import (SerperDevTool, ScrapeWebsiteTool, ScrapeElementFromWebsiteTool)
from vidmarpestel.my_llm import MyLLM
load_dotenv()
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
scrape_element_tool = ScrapeElementFromWebsiteTool()

# Uncomment the following line to use an example of a custom tool
# from vidmarpestel.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class VidmarpestelCrew():
	"""Vidmarpestel crew"""

	@agent
	def political_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['political_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=MyLLM.gpt4o_mini_2024_07_18
		)

	@agent
	def economic_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['economic_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=MyLLM.gpt4o_mini_2024_07_18
		)

	@agent
	def social_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['social_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=MyLLM.gpt4o_mini_2024_07_18,
		)

	@agent
	def technological_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['technological_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=MyLLM.gpt4o_mini_2024_07_18,
		)
  
	@agent
	def environmental_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['environmental_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=MyLLM.gpt4o_mini_2024_07_18,
		)
  
	@agent
	def legal_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['legal_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			verbose=True,
			memory=True,
			allow_delegation=False,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente

			llm=MyLLM.gpt4o_mini_2024_07_18,
		)
		



	@task
	def political_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['political_analysis'],
   			output_file='political_agent.md'
		)

	@task
	def economic_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['economic_analysis'],
   			output_file='economic_agent.md'
		)

	@task
	def social_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['social_analysis'],
   			output_file='social_agent.md'
		)
  
	@task
	def technological_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['technological_analysis'],
   			output_file='technological_agent.md'
		)
  
	@task
	def environmental_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['environmental_analysis'],
   			output_file='environmental_agent.md'
		)
  
	@task
	def legal_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['legal_analysis'],
			output_file='legal_agent.md'
		)


	@crew
	def crew(self) -> Crew:
		return Crew(
			agents=self.agents,  # Automatic collection of agents from decorated methods
			tasks=self.tasks,     # Automatic collection of tasks from decorated methods
			process=Process.sequential,
			verbose=True
        )


