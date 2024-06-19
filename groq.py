from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "tLlama3-7ob-8192"  # Adjust based on available model
os.environ["OPENAI_API_KEY"] = "Token"

email = "hey my guy, it's your friend Steve, wanna go for lunch tomorrow?"
is_verbose = True

classifier = Agent(
    role = "email classifier",
    goal = "accurately classify emails based on their importance. give every email one of these ratings",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly",
    verbose = is_verbose,
    allow_delegation = False
)

responder = Agent(
    role = "email responder",
    goal = "Based on the importance of the email, write a concise and simple response. If the email is important, the response should be as well",
    backstory = "You are an AI assistant whose only job is to write short responses to emails based on their classification",
    verbose = is_verbose,
    allow_delegation = False
)

classify_email = Task(
    description = f"Classify the following email: '{email}'",
    agent = classifier,
    expected_output = "One of these three options: 'important', 'casual', or 'spam'"
)

respond_to_email = Task(
    description = f"Respond to the email: '{email}' based on the importance provided by the 'classifier' agent.",
    agent = responder,
    expected_output = "a very concise response to the email based on the importance provided by the 'classifier' agent."
)

crew = Crew(
    agents = [classifier, responder],
    tasks = [classify_email, respond_to_email],
    verbose = is_verbose,
    process = Process.sequential
)

output = crew.kickoff()
print(output)
