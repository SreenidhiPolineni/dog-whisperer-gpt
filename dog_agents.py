from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatOpenAI

# Replace this with your OpenRouter API key
import os
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="mistralai/mixtral-8x7b-instruct"  # or another model from OpenRouter
)

therapist = Agent(
    role="Dog Therapist",
    goal="Calm and guide dog owners based on bark emotion",
    backstory="Expert in canine behavior and emotional responses",
    llm=llm
)

vet = Agent(
    role="Vet Advisor",
    goal="Provide medical suggestions for distress barks",
    backstory="Veterinary expert analyzing vocal stress cues",
    llm=llm
)

def get_advice(emotion):
    task = Task(
        description=f"A dog bark was detected as {emotion}. What advice would you give the owner?",
        expected_output="One paragraph of helpful, empathetic advice.",
        agent=therapist if emotion != "growl" else vet
    )
    crew = Crew(agents=[therapist, vet], tasks=[task])
    return crew.kickoff()

