from Practice_28 import retrieve_context, build_user_prompt, ask_llm

q = "Why is RAG more reliable than asking a language model directly?"
prompt = build_user_prompt(q, retrieve_context(q))

for temp in [0.1, 1.0]:
    print("=" * 20, f"temperature = {temp}", "=" * 20)
    for run in range(2):
        print(f"\nRun {run + 1}:")
        print(ask_llm(prompt, temperature=temp))
    print()