from core.rag import ask_regulation

def decision_ai(row, regulation_context, llm):
    """
    Decision AI
    - 위험도 판단
    - 규정 근거 요약
    - 조치사항 제안
    """

    prompt = f"""
    Situation data:
    {row}

    Related regulation context:
    {regulation_context}

    1. Determine the risk level (Low / Medium / High)
    2. Explain the basis using regulations
    3. Recommend immediate actions
    """

    decision_text = ask_regulation(prompt, llm)

    return {
        "risk_level": "High",
        "basis": decision_text,
        "action": "Immediately stop work and enforce safety measures."
    }
