from crewai import Agent, Task

from agent_common import JSON_SPEC, guideline_block


def build_agent(llm):
    return Agent(
        role="Academic Writing Reviewer",
        goal="Assess clarity, academic tone, literature-review depth and citation consistency.",
        backstory="You are a journal editor and academic writing mentor who gives precise, actionable "
                  "language and citation feedback.",
        llm=llm, allow_delegation=False, max_iter=2, max_retry_limit=1,
        respect_context_window=True, verbose=False)


def build_task(agent, ctx, callback=None, parallel=True):
    st = ctx["stats"]
    desc = (
        f"Document type: {ctx['doc_type']} ({ctx['level']}).\n"
        f"MEASURED STATS (computed by code, treat as facts): {st}\n"
        f"{guideline_block(ctx['hits']['writing'])}\n\n"
        f"WRITING SAMPLE (abstract / introduction / literature):\n{ctx['excerpts']['writing']}\n\n"
        "Task: evaluate clarity and concision, academic tone, critical (not merely descriptive) literature "
        "synthesis, recency/adequacy of citations, and citation-style consistency.\n" + JSON_SPEC)
    return Task(name="writing", description=desc, expected_output="A JSON review object.",
                agent=agent, async_execution=parallel, callback=callback)
