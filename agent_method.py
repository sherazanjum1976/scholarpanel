from crewai import Agent, Task

from agent_common import JSON_SPEC, guideline_block


def build_agent(llm):
    return Agent(
        role="Methodology & Ethics Evaluator",
        goal="Check that design, data, sampling and analysis fit the research questions and that ethics/plagiarism requirements are addressed.",
        backstory="You are a research-methods expert and ethics-committee member who rejects designs "
                  "that cannot answer the stated questions.",
        llm=llm, allow_delegation=False, max_iter=2, max_retry_limit=1,
        respect_context_window=True, verbose=False)


def build_task(agent, ctx, callback=None, parallel=True):
    desc = (
        f"Document type: {ctx['doc_type']} ({ctx['level']}).\n"
        f"{guideline_block(ctx['hits']['method'])}\n\n"
        f"EXCERPT (methodology / ethics / timeline):\n{ctx['excerpts']['method']}\n\n"
        "Task: evaluate (1) design-to-question fit, (2) data sources, sampling and sample size justification, "
        "(3) analysis techniques and validity/reliability, (4) ethical approval, consent, data protection "
        "and plagiarism-check statements, (5) feasibility of the timeline.\n" + JSON_SPEC)
    return Task(name="method", description=desc, expected_output="A JSON review object.",
                agent=agent, async_execution=parallel, callback=callback)
