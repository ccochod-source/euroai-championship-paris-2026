import asyncio
import json


PHASE = "Veille et vérifications Unlearning"
WORK = (
    ("vision", "Search recent peer-reviewed machine unlearning for vision classifiers, prioritizing forget-only, label-free methods. Verify benchmark/data assumptions from papers and source. Rank three genuinely distinct options for this ViT."),
    ("ssd", "Verify the official SSD and LF-SSD papers and repos, especially how the importance ratio is computed. Design and test a small Torch implementation for label-free dampening on forget images and permitted synthetic controls, with bounded head-only or late-layer edits. Share source as an attachment if tested."),
    ("jit", "Read the JiT paper and independent work on forgetting by local smoothness. Check what labels and controls are needed and derive an original, low-cost ViT adaptation from the paper. Verify the key math with a toy Torch test, then provide practical GPU settings."),
    ("distillation", "Research view-ensemble distillation and data-free retention for image classifiers. Propose and numerically test fast final-layer ridge targets with mixed-image anchors or OOF teacher estimates. Prefer a design that uses already cached penultimate features and preserves the same architecture."),
    ("calibration", "Research calibration and selective editing based on margins, class-conditional or low-rank mappings. Derive a closed-form head-only update to modulate the strongest confidence while preserving control predictions and alternatives. Compare with plain logit temperature. Verify assumptions and give an immediately testable hyperparameter grid."),
    ("evaluation", "Look for evidence and best practices on unlearning evaluation, using published papers and challenge reports. Examine how our unlearning loss/KL proxy can misrank candidates, including teacher temperature, hidden-view pooling, proxy labels, scale of retention and public-vs-private split. Provide concrete offline regression cases and proposed checks."),
    ("competition", "Review public organizer repositories/docs, accessible issues and publicly documented solutions of comparable competitions. Confirm allowed data, the Unlearning scoring rules and practical checkpoint requirements with links. Propose isolated, local smoke tests for submission validation and retry logic using fixtures or mocks."),
)

CONTEXT = """Clément is competing in the European AI Championship at Epitech. His request is for six to ten agents to « faire la veille sur tout Internet pour récupérer le plus d'informations sur le learning et sur comment ça se passe ». We are focusing on the machine Unlearning track. Public kit https://huggingface.co/datasets/SprintML/unlearning and public setup https://huggingface.co/datasets/SprintML/hackathon. Search public GitHub, papers, technical docs and public issues using targeted read-only access. Any tests you run must be isolated, local model/metric tests or harmless synthetic, mocked validation tests. No exploitation, vulnerability reproduction, live probing, submissions, writes to third-party services, private tickets, secrets, private scores or datasets. Respect source licenses and verify what really ran.

The supplied model is ViT-B/16 with LayerScale, 100 classes, and 3500 unlabeled forget images. Inputs are 224x224 RGB, NCHW float32 in [-1,1]. The retain set and retrained reference models are unavailable. The official forget utility combines gold CE distribution and per-image KL at T=2 for clean and hidden altered views. Retention uses hidden-control accuracy and KL. Below 70% hidden-control accuracy scores zero. Public ranking is 30% of the forget set. Official checkpoints must strict-load as a finite float32 state_dict under 450 MB. Official scores known so far: unchanged 0, temperature 1.5 0.000135, class-conditional head ridge 0.0082, head-delta scaling 1.07 0.0099 and 1.10 0.0101. Our simulator uses synthetic teachers, transformed views and CutMix-like controls. It can compare settings quickly from cached final features but cannot establish the official score. Parent owns the A100, code integration and all submissions.

Report in French. Time-box to about 8 minutes and prioritize findings usable today. Provide 2–5 verified source URLs, assumptions, expected cost, one first test with concrete settings and a summary of what you actually verified. For a tested code artifact, return its attachment URL; no weights or datasets."""

RESULT = {
    "type": "object",
    "properties": {
        "finding": {"type": "string"},
        "test": {"type": "string"},
        "settings": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "artifact": {"type": "string"},
    },
    "required": ["finding", "test", "settings", "sources", "artifact"],
}


async def research(label, assignment):
    try:
        result = await agent(
            f"{CONTEXT}\n\nYour assignment: {assignment}",
            phase=PHASE, schema=RESULT, label=label,
        )
    except WorkflowAgentError as error:
        log(f"{label}: {error}")
        return {"topic": label, "error": str(error)}
    log(f"{label}: findings received")
    return {"topic": label, **result}


async def main():
    await register_workflow({
        "name": "Veille Unlearning Eugenia",
        "description": "Vérifier sept pistes de recherche, d’évaluation et de validation",
        "soft_time_limit_minutes": 8,
        "phases": [{"title": PHASE, "detail": "Comparer les sources publiques et tester les idées en local",
                    "labels": [label for label, _ in WORK]}],
    })
    log(f"Starting {len(WORK)} independent research tasks")
    results = await parallel([
        lambda label=label, assignment=assignment: research(label, assignment)
        for label, assignment in WORK
    ])
    log(f"{sum('error' not in result for result in results)}/{len(WORK)} tasks returned findings")
    print(json.dumps(results, ensure_ascii=False))


asyncio.run(main())
