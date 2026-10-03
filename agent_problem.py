from crewai import Agent, Task

from agent_common import JSON_SPEC, guideline_block


def build_agent(llm):
    return Agent(
        role="Problem & Objectives Analyst",
        goal="Test whether the research gap, questions, objectives and hypotheses are clear, novel and logically connected.",
        backstory="You are a senior researcher who sits on proposal-defence panels and is known for "
                  "spotting vague problem statements and objectives that do not match the research questions.",
        llm=llm, allow_delegation=False, max_iter=2, max_retry_limit=1,
        respect_context_window=True, verbose=False)


def build_task(agent, ctx, callback=None, parallel=True):
    desc = (
        f"Document type: {ctx['doc_type']} ({ctx['level']}).\n"
        f"{guideline_block(ctx['hits']['problem'])}\n\n"
        f"EXCERPT (introduction / problem / questions / objectives):\n{ctx['excerpts']['problem']}\n\n"
        "Task: evaluate (1) is the research gap real and evidenced, (2) are questions/hypotheses specific "
        "and testable, (3) are objectives SMART and traceable to the questions, (4) is the contribution "
        "appropriate for the degree level (MS vs PhD).\n" + JSON_SPEC)
    return Task(name="problem", description=desc, expected_output="A JSON review object.",
                agent=agent, async_execution=parallel, callback=callback)
