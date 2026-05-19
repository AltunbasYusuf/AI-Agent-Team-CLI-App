import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# !!!!! Create .env file and load environment variables (Your API Key must be here) !!!!!
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key,
    temperature=0.5
)

@tool("File Writing Tool")
def save_code_to_file(filename: str, code_content: str) -> str:
    """Saves the generated code or text to a file on the local disk."""
    try:
        clean_content = code_content.replace("```python", "").replace("```", "")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(clean_content.strip())
        return f"SUCCESS: Code successfully written to '{filename}'."
    except Exception as e:
        return f"ERROR: An error occurred while writing the file: {str(e)}"

# --- AGENTS ---
pm_agent = Agent(
    role="Senior Product Manager",
    goal="Analyze requests and prepare technical specification documents.",
    backstory="You think like a software architect and create perfect roadmaps.",
    llm=gemini_llm,
    verbose=True
)

developer_agent = Agent(
    role="Senior Python Developer",
    goal="Write the most optimized code based on specifications or QA reports and save it to disk.",
    backstory="You write clean code, follow SOLID principles, and never forget to save your code using the 'File Writing Tool'.",
    tools=[save_code_to_file],
    llm=gemini_llm,
    verbose=True
)

qa_agent = Agent(
    role="Senior QA Test Engineer",
    goal="Review the code, find bugs, and make the final decision.",
    backstory="You are strict. If there is even a single bug in the code, you absolutely end your report with the word 'REJECTED'. If the code is flawless, you end your report with the word 'APPROVED'.",
    llm=gemini_llm,
    verbose=True
)

if __name__ == "__main__":
    project_request = "A CLI-based budget tracking application where users can add expenses, categorize them, and get a total expense report."
    file_name = "budget_tracker.py"
    max_epoch = 3 # Maximum number of attempts to prevent infinite loops

    print("🚀 EPOCH 0: Initial Development Process Starting...")
    
    # PHASE 1: INITIAL CREATION (PM -> DEV -> QA)
    task_analysis = Task(
        description=f"Analyze this request: '{project_request}'. Prepare detailed technical specifications.",
        expected_output="Technical specification document for the developer.",
        agent=pm_agent
    )

    task_coding = Task(
        description=f"Take the analysis. Write the Python code. Save the code as '{file_name}' using the 'File Writing Tool'.",
        expected_output="The written code and file saving confirmation.",
        agent=developer_agent,
        context=[task_analysis]
    )

    task_testing = Task(
        description="Review the code written by the developer. Are there any bugs?",
        expected_output="Test report and final decision (APPROVED or REJECTED).",
        agent=qa_agent,
        context=[task_coding],
        human_input=True # <--- HUMAN IN THE LOOP!
    )

    initial_crew = Crew(
        agents=[pm_agent, developer_agent, qa_agent],
        tasks=[task_analysis, task_coding, task_testing],
        process=Process.sequential,
    )
    
    # Getting the first result and converting it to string
    qa_result = str(initial_crew.kickoff())
    print("\n--- EPOCH 0 COMPLETED. QA REPORT: ---")
    print(qa_result)

    # PHASE 2: SELF-HEALING LOOP
    epoch_counter = 1
    
    # Changed the trigger word to "REJECTED"
    while "REJECTED" in qa_result and epoch_counter <= max_epoch:
        print(f"\n⚠️ QA REJECTED THE CODE! STARTING EPOCH {epoch_counter} (Bug Fixing Phase)...")
        
        # Creating a new rescue team consisting only of Dev and QA
        task_fix = Task(
            description=(
                f"The QA agent rejected the previous code. Here is the QA's error report:\n{qa_result}\n\n"
                f"Please fix your code according to this report and overwrite the '{file_name}' file using the 'File Writing Tool'."
            ),
            expected_output="Fixed code and file saving confirmation.",
            agent=developer_agent
        )

        task_retest = Task(
            description="The developer updated the code. Review the new code. If bugs persist, write 'REJECTED', if flawless, write 'APPROVED'.",
            expected_output="Updated test report and decision (APPROVED or REJECTED).",
            agent=qa_agent,
            context=[task_fix]
        )

        fix_crew = Crew(
            agents=[developer_agent, qa_agent],
            tasks=[task_fix, task_retest],
            process=Process.sequential,
        )

        # Run the fix crew and get the new QA report
        qa_result = str(fix_crew.kickoff())
        print(f"\n--- EPOCH {epoch_counter} COMPLETED. NEW QA REPORT: ---")
        print(qa_result)
        
        epoch_counter += 1

    # LOOP RESULT
    print("\n================ PROCESS FINISHED ================")
    if "APPROVED" in qa_result:
        print("✅ Success: The code passed QA approval and has been saved!")
    else:
        print(f"❌ Failed: Reached maximum attempts ({max_epoch}), but QA still hasn't approved. There is a stubborn bug in the code.")