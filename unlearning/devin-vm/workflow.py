import asyncio
import json


PHASE = "Unlearning experiments"
WORK = (
    ("evaluation", "Implement evaluation.py with reproducible tuning/held-out indices, light validation transforms and an original-versus-candidate comparison of top-1 agreement, tempered KL, confidence and pseudo-label loss statistics. Include a way to evaluate a user-provided permitted control array. Do not assume a known source dataset or ground-truth labels. Document how to read the proxy metrics."),
    ("head", "Implement head.py for a fast targeted classifier-head update. Try a feature-space ridge regression or small gradient-based fit toward class-conditional soft targets. Use the frozen original ViT features, regularization and adjustable strength. Prioritize a stable initial candidate with a small parameter change and include a proposed sweep."),
    ("gradient", "Implement gradient.py for a bounded update to the last ViT block(s) and/or head. Use an objective that reduces original pseudo-label confidence on forget samples while regularizing weight drift and preserving useful output structure through distillation where appropriate. Use minibatches, clipping and a small sweep. Avoid unconstrained ascent."),
    ("targets", "Implement targets.py for augmentation-derived soft prediction targets and a lightweight weight update to fit them. Generate several mild, plausible unseen views, aggregate original-model evidence for plausible alternative classes, and preserve the structure of the prediction distribution. Include a calibration or fit-strength sweep."),
    ("influence", "Implement influence.py for an approximate influence/forget-only gradient update or low-rank inverse-curvature correction. Explain the approximation and keep memory and trainable parameter count small. Propose a norm/interpolation sweep and clearly identify conditions under which the correction may hurt retention."),
    ("validation", "Implement validation.py to validate submission schema/size/finite float32 tensors and strict checkpoint loading. Design GPU smoke and Slurm instructions for training2666, partition dc-gpu, reservation cispahack, 8 CPU cores and one A100 requested. Include reproducibility and failure diagnostics. Do not schedule jobs or submit models yourself."),
)

CONTEXT = """Clément's request: « prépare un plan qui utilisera plusieurs sous-agents que tu lanceras, en sachant que tu seras l'agent orchestrateur dans l'histoire. » You own one work stream in the approved implementation, to maximize team Eugenia's Unlearning result at the Paris hackathon.

Official starter: https://huggingface.co/datasets/SprintML/unlearning/resolve/main/task_template.py . Setup: https://huggingface.co/datasets/SprintML/hackathon . Frozen original is a ViT-B/16, 100 classes and LayerScale. 3500 unlabeled RGB uint8 images are provided, NHWC, 224 x 224; original preprocessing is fp32 NCHW in [-1,1]. Baseline original ran on JURECA A100 and scored 0. Official score: forget loss statistics and temperature-2 prediction KL to retrained gold, equally across originals and unseen transformed copies, times hidden-control retention. Below 70% hidden-control accuracy scores zero. Public split is 30%, holdout 70%. Gold, control labels and private leaderboard inputs are unavailable. Do not infer a confirmed score from proxy metrics.

Use a separate work directory and only permitted public starter materials. The parent owns JURECA, GPU jobs and submissions. Do not request or access the team token or private data, submit any models, or push to the organizer repository. Work against Torch and timm from the official requirements, with no additional required pip dependencies. Test a tiny synthetic case and report honestly what you ran. Do not wait for the full checkpoint/dataset download to test core math. Time-box your work for the on-site hackathon.

Method streams should expose fit(model, forget_images, *, strength, seed), which returns the modified model. The model is a loaded ViT and on the intended device, forget_images is the permitted training subset in its original uint8 NHWC format on CPU, and any preprocessing should match task_template.py. `strength` is a documented positive float and `seed` an int. Provide safe defaults, candidate sweep values and simple CPU/GPU instructions. Do not modify the official task_template.py. Evaluation and validation streams own their corresponding modules and CLI/interfaces only.

There is no writable shared project remote. Put only your source and minimal usage instructions on a local git branch named for your stream, export that branch with `git bundle create`, and deliver the bundle to the parent using upload_attachment (or your child message attachment). Return the attachment URL from your upload tool as `bundle_url` so the orchestrator can import your branch. If delivery is blocked, return runnable source as `fallback_source`; do not include weights or datasets. Report your result through the structured output tool with summary, branch, bundle_url, test, settings and fallback_source. Deliver a working first version promptly; record any known limitations."""

RESULT = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in (
        "summary", "branch", "bundle_url", "test", "settings", "fallback_source"
    )},
    "required": ["summary", "branch", "bundle_url", "test", "settings", "fallback_source"],
}


async def execute(label, assignment):
    try:
        result = await agent(
            f"{CONTEXT}\n\nYour stream is {label}. {assignment}",
            phase=PHASE,
            label=label,
            schema=RESULT,
        )
    except WorkflowAgentError as error:
        log(f"{label} ended without a result: {error}")
        return {"stream": label, "error": str(error)}
    log(f"{label} delivered its first version")
    return {"stream": label, **result}


async def main():
    await register_workflow({
        "name": "Eugenia Unlearning",
        "description": "Four candidate methods, shared evaluation, and submission validation",
        "product": "European AI Championship Paris",
        "soft_time_limit_minutes": 12,
        "phases": [{"title": PHASE, "detail": "Develop and validate independent work streams", "labels": [label for label, _ in WORK]}],
    })
    log("Starting six parallel work streams. The parent owns JURECA and the submission queue.")
    results = await parallel([
        lambda label=label, assignment=assignment: execute(label, assignment)
        for label, assignment in WORK
    ])
    log(f"{sum('error' not in item for item in results)} of {len(WORK)} work streams delivered results")
    print(json.dumps(results, ensure_ascii=False))


asyncio.run(main())
