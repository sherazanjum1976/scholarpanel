from crewai import Agent, Task

from agent_common import JSON_SPEC, guideline_block


def build_agent(llm):
    return Agent(
        role="Document Structure Auditor",
        goal="Verify that a postgraduate synopsis/thesis follows the required format and section checklist.",
        backstory="You are a graduate-office reviewer who has checked hundreds of synopses and theses "
                  "against the university's formatting and section requirements.",
        llm=llm, allow_delegation=False, max_iter=2, max_retry_limit=1,
        respect_context_window=True, verbose=False)


def build_task(agent, ctx, callback=None, parallel=True):
    missing = [r["label"] for r in ctx["checks"] if not r["present"]]
    desc = (
        f"Document type: {ctx['doc_type']} ({ctx['level']}). Automated check found these REQUIRED sections "
        f"missing: {missing or 'none'}. Section coverage: {round(ctx['coverage'] * 100)}%.\n"
        f"{guideline_block(ctx['hits']['structure'])}\n\n"
        f"DOCUMENT OPENING EXCERPT:\n{ctx['excerpts']['structure']}\n\n"
        "Task: judge structural completeness and format compliance (section order, title/abstract quality, "
        "front matter, references presence). Treat the automated missing-section list as factual.\n" + JSON_SPEC)
    return Task(name="structure", description=desc, expected_output="A JSON review object.",
                agent=agent, async_execution=parallel, callback=callback)
