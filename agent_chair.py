from crewai import Agent, Task


def build_agent(llm):
    return Agent(
        role="Panel Chair",
        goal="Merge the four reviewer reports, resolve conflicts and produce a prioritised, actionable revision roadmap.",
        backstory="You chair the graduate studies review committee. You are fair, concise and always "
                  "turn criticism into concrete next steps for the student.",
        llm=llm, allow_delegation=False, max_iter=2, max_retry_limit=1,
        respect_context_window=True, verbose=False)


def build_task(agent, ctx, context_tasks, callback=None):
    desc = (
        f"You received four independent JSON reviews (structure, problem/objectives, methodology/ethics, "
        f"writing) of a {ctx['doc_type']} ({ctx['level']}). Automated section coverage: "
        f"{round(ctx['coverage'] * 100)}%.\n\n"
        "Write the panel report in Markdown with EXACTLY these sections:\n"
        "## Panel Summary\n(3-4 sentences: overall readiness, strongest point, biggest risk)\n"
        "## Conflicts & Cross-Cutting Themes\n(bullets: where reviewers agreed/disagreed and how you resolved it; max 4)\n"
        "## Prioritised Revision Roadmap\n(numbered list, max 8 items, each starting with [P1], [P2] or [P3], "
        "then the action, then in parentheses which reviewer raised it)\n"
        "## Likely Questions at Defence\n(max 4 bullets)\n\n"
        "Rules: be specific to this document; do not invent facts; do NOT state a numeric overall score or a "
        "final decision (the system computes those); keep the whole report under 450 words.")
    return Task(name="chair", description=desc, context=context_tasks,
                expected_output="A Markdown panel report with the four required sections.",
                agent=agent, async_execution=False, callback=callback)
