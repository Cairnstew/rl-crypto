import { spawn, execFileSync } from "node:child_process";
import { homedir } from "node:os";
import { join, resolve, sep } from "node:path";

/**
 * rl-model-lab: an offline, out-of-sample model improvement loop.
 *
 * It does not promise profit. It measures whether the current models beat a
 * constant baseline AND a simple momentum policy across walk-forward folds,
 * after fees and slippage, and only promotes a candidate that survives a
 * reserved final holdout.
 *
 * The Python engine lives at tools/model_lab.py. Artifacts are written outside
 * the repo by default (~/.local/share/rl-crypto/model-lab) so experiments never
 * pollute data/ or models/.
 *
 * Environment knobs:
 *   RL_CRYPTO_ROOT                    override repo root detection
 *   RL_CRYPTO_LAB_DIR                 override the artifact directory
 *   RL_MODEL_LAB_ALLOW_ANY_CANDIDATE  set to 1 to allow candidates outside LAB_DIR
 *   RL_CRYPTO_QUIET                   set by this tool; honour it in the shellHook
 *                                     to suppress the dev-shell banner at source
 */

const ACTIONS = ["status", "evaluate", "sweep", "promote", "report", "help"] as const;
type Action = (typeof ACTIONS)[number];

const MAX_OUTPUT_BYTES = 50 * 1024 * 1024;
const KILL_GRACE_MS = 5_000;

const TIMEOUTS_MS: Record<Action, number> = {
    sweep: 900_000,
    evaluate: 900_000,
    promote: 300_000,
    status: 180_000,
    report: 180_000,
    help: 60_000,
};

// Which arguments each action actually consumes. Anything else that is set
// gets reported back as ignored instead of silently dropped.
const USED_ARGS: Record<Action, ReadonlySet<string>> = {
    help: new Set(["action"]),
    status: new Set(["action", "tickers", "json"]),
    report: new Set(["action", "limit", "json"]),
    promote: new Set([
        "action",
        "candidate",
        "ticker",
        "tickers",
        "targets",
        "threshold",
        "position",
        "feeBps",
        "slippageBps",
        "apply",
        "holdoutStart",
        "seed",
        "championRetrain",
        "json",
    ]),
    sweep: new Set([
        "action",
        "tickers",
        "targets",
        "folds",
        "threshold",
        "position",
        "feeBps",
        "slippageBps",
        "featureSets",
        "modelSets",
        "horizons",
        "maxConfigs",
        "primaryConfig",
        "primaryTarget",
        "holdoutStart",
        "seed",
        "alpha",
        "bootstrapSamples",
        "bootstrapBlock",
        "limit",
        "json",
    ]),
    evaluate: new Set([
        "action",
        "tickers",
        "targets",
        "folds",
        "threshold",
        "position",
        "feeBps",
        "slippageBps",
        "featureSet",
        "modelSet",
        "horizon",
        "primaryTarget",
        "holdoutStart",
        "seed",
        "alpha",
        "bootstrapSamples",
        "bootstrapBlock",
        "controls",
        "shiftBars",
        "fullArchive",
        "thresholds",
        "costLevels",
        "canaryLeak",
        "limit",
        "json",
    ]),
};

type LabArgs = {
    action: string;
    tickers?: string;
    targets?: string;
    featureSet?: string;
    modelSet?: string;
    horizon?: number;
    folds?: number;
    threshold?: number;
    position?: string;
    feeBps?: number;
    slippageBps?: number;
    featureSets?: string;
    modelSets?: string;
    horizons?: string;
    maxConfigs?: number;
    candidate?: string;
    ticker?: string;
    apply?: boolean;
    holdoutStart?: string;
    seed?: number;
    alpha?: number;
    bootstrapSamples?: number;
    bootstrapBlock?: number;
    controls?: string;
    shiftBars?: number;
    fullArchive?: boolean;
    primaryTarget?: string;
    primaryConfig?: string;
    thresholds?: string;
    costLevels?: string;
    canaryLeak?: string;
    championRetrain?: boolean;
    json?: boolean;
    limit?: number;
};

type ToolContext = {
    abort?: AbortSignal;
    directory?: string;
    worktree?: string;
};

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

function git(cwd: string): string | undefined {
    try {
        const out = execFileSync("git", ["rev-parse", "--show-toplevel"], {
            cwd,
            encoding: "utf-8",
            stdio: ["ignore", "pipe", "ignore"],
        }).trim();
        return out || undefined;
    } catch {
        return undefined;
    }
}

function repoRoot(ctx?: ToolContext): string {
    if (process.env.RL_CRYPTO_ROOT) return resolve(process.env.RL_CRYPTO_ROOT);
    const candidates = [ctx?.worktree, ctx?.directory, process.cwd()].filter(
        (d): d is string => !!d && d !== "/",
    );
    for (const dir of candidates) {
        const root = git(dir);
        if (root) return root;
    }
    return resolve(candidates[0] ?? ".");
}

function labDir(): string {
    return resolve(
        process.env.RL_CRYPTO_LAB_DIR ||
            join(
                process.env.XDG_DATA_HOME || join(homedir(), ".local", "share"),
                "rl-crypto",
                "model-lab",
            ),
    );
}

// Fallback only: prefer RL_CRYPTO_QUIET in the shellHook.
const BANNER =
    /^(rl-crypto dev shell|  uv run python -m pytest|  uv run python tools\/build_notebook|  uv run rl-pipeline|  ruff check src\/)/;

function clean(output: string): string {
    return output
        .split("\n")
        .filter((line) => !BANNER.test(line))
        .join("\n")
        .trim();
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function checkNumber(
    name: string,
    value: number | undefined,
    opts: {
        min?: number;
        max?: number;
        integer?: boolean;
        exclusiveMin?: boolean;
        exclusiveMax?: boolean;
    },
): string | undefined {
    if (value === undefined) return undefined;
    if (typeof value !== "number" || !Number.isFinite(value))
        return `${name} must be a finite number.`;
    if (opts.integer && !Number.isInteger(value)) return `${name} must be an integer.`;
    if (opts.min !== undefined && (opts.exclusiveMin ? value <= opts.min : value < opts.min)) {
        return `${name} must be ${opts.exclusiveMin ? ">" : ">="} ${opts.min}.`;
    }
    if (opts.max !== undefined && (opts.exclusiveMax ? value >= opts.max : value > opts.max)) {
        return `${name} must be ${opts.exclusiveMax ? "<" : "<="} ${opts.max}.`;
    }
    return undefined;
}

function validate(args: LabArgs): string | undefined {
    const errors = [
        checkNumber("folds", args.folds, { min: 2, integer: true }),
        checkNumber("threshold", args.threshold, {
            min: 0,
            max: 1,
            exclusiveMin: true,
            exclusiveMax: true,
        }),
        checkNumber("feeBps", args.feeBps, { min: 0 }),
        checkNumber("slippageBps", args.slippageBps, { min: 0 }),
        checkNumber("horizon", args.horizon, { min: 1, integer: true }),
        checkNumber("maxConfigs", args.maxConfigs, { min: 1, integer: true }),
        checkNumber("limit", args.limit, { min: 1, integer: true }),
    ].filter(Boolean);
    if (args.position !== undefined && !["long-only", "long-short"].includes(args.position)) {
        errors.push("position must be 'long-only' or 'long-short'.");
    }
    if (args.modelSet !== undefined && !["reference", "baseline"].includes(args.modelSet)) {
        errors.push("modelSet must be 'reference' or 'baseline'.");
    }
    return errors.length ? errors.join(" ") : undefined;
}

function ignoredArgs(action: Action, args: LabArgs): string[] {
    const used = USED_ARGS[action];
    return Object.entries(args)
        .filter(
            ([key, value]) =>
                value !== undefined && value !== "" && value !== false && !used.has(key),
        )
        .map(([key]) => key);
}

const normList = (value?: string) => (value ? value.trim().replace(/\s+/g, " ") : undefined);

// ---------------------------------------------------------------------------
// Promote safety: apply requires a prior dry run of the same candidate in this
// session, so a write to models/ is never the first thing that happens.
// ---------------------------------------------------------------------------

const dryRunPassed = new Set<string>();

// ---------------------------------------------------------------------------
// Subprocess runner
// ---------------------------------------------------------------------------

type RunResult = {
    stdout: string;
    stderr: string;
    code: number | null;
    signal: NodeJS.Signals | null;
    timedOut: boolean;
    aborted: boolean;
    truncated: boolean;
    spawnError?: string;
};

function run(
    cmd: string,
    argv: string[],
    opts: { cwd: string; env: NodeJS.ProcessEnv; timeoutMs: number; abort?: AbortSignal },
): Promise<RunResult> {
    return new Promise((resolvePromise) => {
        const result: RunResult = {
            stdout: "",
            stderr: "",
            code: null,
            signal: null,
            timedOut: false,
            aborted: false,
            truncated: false,
        };

        // detached => own process group, so we can kill nix + uv + python together.
        const child = spawn(cmd, argv, {
            cwd: opts.cwd,
            env: opts.env,
            stdio: ["ignore", "pipe", "pipe"],
            detached: true,
        });

        let bytes = 0;
        let killTimer: NodeJS.Timeout | undefined;

        const killGroup = (sig: NodeJS.Signals) => {
            if (child.pid === undefined) return;
            try {
                process.kill(-child.pid, sig);
            } catch {
                try {
                    child.kill(sig);
                } catch {
                    /* already gone */
                }
            }
        };

        const terminate = () => {
            killGroup("SIGTERM");
            killTimer = setTimeout(() => killGroup("SIGKILL"), KILL_GRACE_MS);
        };

        const collect = (key: "stdout" | "stderr") => (chunk: Buffer) => {
            bytes += chunk.length;
            if (bytes > MAX_OUTPUT_BYTES) {
                if (!result.truncated) {
                    result.truncated = true;
                    terminate();
                }
                return;
            }
            result[key] += chunk.toString("utf-8");
        };
        child.stdout!.on("data", collect("stdout"));
        child.stderr!.on("data", collect("stderr"));

        const timer = setTimeout(() => {
            result.timedOut = true;
            terminate();
        }, opts.timeoutMs);

        const onAbort = () => {
            result.aborted = true;
            terminate();
        };
        if (opts.abort) {
            if (opts.abort.aborted) onAbort();
            else opts.abort.addEventListener("abort", onAbort, { once: true });
        }

        const finish = () => {
            clearTimeout(timer);
            if (killTimer) clearTimeout(killTimer);
            opts.abort?.removeEventListener("abort", onAbort);
            resolvePromise(result);
        };

        child.on("error", (err) => {
            result.spawnError = err.message;
            finish();
        });
        child.on("close", (code, signal) => {
            result.code = code;
            result.signal = signal;
            finish();
        });
    });
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

export default {
    description:
        "Offline walk-forward model lab for rl-crypto. Actions: status (deployed bundles/trust/freshness), evaluate (out-of-sample metrics + after-cost economic proxy vs constant and momentum baselines), sweep (controlled feature/model/horizon experiments, writes candidates), promote (validate a candidate on a reserved final holdout; dry run unless apply=true, and apply requires a prior dry run of the same candidate), report (experiment ledger), help (full argument reference). It measures honestly and does not guarantee profit. Artifacts live under ~/.local/share/rl-crypto/model-lab.",
    args: {
        action: {
            type: "string",
            description: "One of: status, evaluate, sweep, promote, report, help.",
        },
        tickers: {
            type: "string",
            description:
                "Ticker(s), comma- or space-separated (e.g. 'BTC/USD ETH/USD'). Default: all for evaluate/status, BTC/ETH/SOL for sweep.",
        },
        targets: {
            type: "string",
            description:
                "Target(s), comma- or space-separated. Default: all four. Use fwd_direction for the trade-relevant model.",
        },
        featureSet: {
            type: "string",
            description:
                "evaluate: all | market | core | price, or comma-separated feature groups. Default all.",
        },
        modelSet: {
            type: "string",
            description:
                "evaluate: reference (ridge/logistic) | baseline (mean/base_rate). Default reference.",
        },
        horizon: {
            type: "number",
            description:
                "evaluate: forward horizon in bars (integer >= 1). Default 1. The economic proxy is only honest at horizon 1.",
        },
        folds: {
            type: "number",
            description: "Walk-forward folds (integer >= 2). Default 4.",
        },
        threshold: {
            type: "number",
            description:
                "Direction probability needed to take a long (and, long-short, 1-threshold for a short). Strictly between 0 and 1. Default 0.55.",
        },
        position: {
            type: "string",
            description: "long-only | long-short. Default long-only.",
        },
        feeBps: {
            type: "number",
            description: "Per-side fee in basis points (>= 0). Default 10.",
        },
        slippageBps: {
            type: "number",
            description: "Per-side slippage in basis points (>= 0). Default 5.",
        },
        featureSets: {
            type: "string",
            description:
                "sweep: comma-separated feature sets (all,market,core,price). Default all,market,core,price.",
        },
        modelSets: {
            type: "string",
            description:
                "sweep: comma-separated model sets (reference,baseline). Default reference,baseline.",
        },
        horizons: {
            type: "string",
            description:
                "sweep: comma- or space-separated horizons. Default 1. Keep 1 for the economic proxy.",
        },
        maxConfigs: {
            type: "number",
            description: "sweep: cap the config grid. Default 16.",
        },
        candidate: {
            type: "string",
            description:
                "promote: path to a candidate bundle JSON (usually written by sweep). Must live under the model-lab artifact directory.",
        },
        ticker: {
            type: "string",
            description:
                "promote: ticker for the candidate if it cannot be inferred from the filename.",
        },
        apply: {
            type: "boolean",
            description:
                "promote: actually write models/<slug>.json after a passing gate. Default false (dry run). Requires a prior dry run of the same candidate in this session.",
        },
        holdoutStart: {
            type: "string",
            description:
                "Freeze the holdout at this ISO timestamp. promote apply requires it.",
        },
        seed: {
            type: "number",
            description: "Controls and bootstrap seed. Default 0.",
        },
        alpha: {
            type: "number",
            description:
                "evaluate/sweep: CI alpha, already spent for this trial by the driver.",
        },
        bootstrapSamples: {
            type: "number",
            description: "evaluate/sweep: block-bootstrap draws.",
        },
        bootstrapBlock: {
            type: "number",
            description: "evaluate/sweep: bars per bootstrap block.",
        },
        controls: {
            type: "string",
            description: "evaluate: none | shuffle | shift | both.",
        },
        shiftBars: {
            type: "number",
            description: "evaluate: shift width for the shift control.",
        },
        fullArchive: {
            type: "boolean",
            description: "evaluate: diagnostic full-archive run (never verdict-eligible).",
        },
        primaryTarget: {
            type: "string",
            description: "Target of record for the pooled paired improvement.",
        },
        primaryConfig: {
            type: "string",
            description: "sweep: the only verdict-eligible config (e.g. all+reference+h1).",
        },
        thresholds: {
            type: "string",
            description: "evaluate: comma/space thresholds to sweep in one fit.",
        },
        costLevels: {
            type: "string",
            description: "evaluate: fee:slippage pairs to sweep in one fit, e.g. 10:5,0:0.",
        },
        canaryLeak: {
            type: "string",
            description: "evaluate: positive-canary only; inject this feature from the label.",
        },
        championRetrain: {
            type: "boolean",
            description:
                "promote: retrain the champion on the pre-cutoff slice (default true).",
        },
        json: {
            type: "boolean",
            description: "Return the machine-readable payload instead of the table.",
        },
        limit: {
            type: "number",
            description: "Max table/ledger rows. Default 20 (evaluate/sweep), 10 (report).",
        },
    },

    async execute(args: LabArgs, ctx?: ToolContext) {
        const action = (args.action || "").toLowerCase() as Action;
        if (!ACTIONS.includes(action)) {
            return `rl-model-lab: action must be one of ${ACTIONS.join(", ")}.`;
        }

        const invalid = validate(args);
        if (invalid) return `rl-model-lab: ${invalid}`;

        const notes: string[] = [];
        const ignored = ignoredArgs(action, args);
        if (ignored.length) {
            notes.push(`note: ignored args not used by '${action}': ${ignored.join(", ")}.`);
        }
        if (action === "evaluate" && args.horizon !== undefined && args.horizon !== 1) {
            notes.push(
                "note: horizon != 1, so the after-cost economic proxy is not honest for this run.",
            );
        }
        if (
            action === "sweep" &&
            args.horizons &&
            normList(args.horizons)!
                .split(/[\s,]+/)
                .some((h) => h !== "1")
        ) {
            notes.push(
                "note: horizons other than 1 make the after-cost economic proxy unreliable.",
            );
        }

        const argv: string[] = [action];
        const add = (flag: string, value: string | number | boolean | undefined) => {
            if (value === undefined || value === false || value === "") return;
            if (value === true) {
                argv.push(flag);
                return;
            }
            argv.push(flag, String(value));
        };

        const economics = () => {
            add("--threshold", args.threshold ?? 0.55);
            add("--position", args.position ?? "long-only");
            add("--fee-bps", args.feeBps ?? 10);
            add("--slippage-bps", args.slippageBps ?? 5);
        };
        const statistics = () => {
            add("--primary-target", args.primaryTarget);
            add("--holdout-start", args.holdoutStart);
            add("--seed", args.seed);
            add("--alpha", args.alpha);
            add("--bootstrap-samples", args.bootstrapSamples);
            add("--bootstrap-block", args.bootstrapBlock);
        };

        let promoteKey: string | undefined;

        switch (action) {
            case "help":
                // Prints the argument reference; takes no flags.
                break;

            case "status":
                add("--tickers", normList(args.tickers));
                add("--json", args.json);
                break;

            case "report":
                add("--limit", args.limit ?? 10);
                add("--json", args.json);
                break;

            case "promote": {
                if (!args.candidate) return "rl-model-lab: promote requires candidate.";
                const candidate = resolve(args.candidate);
                const lab = labDir();
                const inLab = candidate === lab || candidate.startsWith(lab + sep);
                if (!inLab && process.env.RL_MODEL_LAB_ALLOW_ANY_CANDIDATE !== "1") {
                    return `rl-model-lab: candidate must be under ${lab} (set RL_MODEL_LAB_ALLOW_ANY_CANDIDATE=1 to override).`;
                }
                promoteKey = `${candidate}::${args.ticker ?? ""}`;
                if (args.apply && !dryRunPassed.has(promoteKey)) {
                    return "rl-model-lab: apply=true requires a prior dry run (apply=false) of this same candidate in this session. Run the dry run first and review the gate result.";
                }
                add("--candidate", candidate);
                add("--ticker", args.ticker);
                add("--tickers", normList(args.tickers));
                add("--targets", normList(args.targets));
                economics();
                add("--holdout-start", args.holdoutStart);
                add("--seed", args.seed);
                if (args.championRetrain === false) add("--no-champion-retrain");
                add("--apply", args.apply);
                add("--json", args.json);
                break;
            }

            case "sweep":
                add("--tickers", normList(args.tickers));
                add("--targets", normList(args.targets));
                add("--folds", args.folds ?? 4);
                economics();
                statistics();
                add("--primary-config", args.primaryConfig);
                add("--feature-sets", args.featureSets ?? "all,market,core,price");
                add("--model-sets", args.modelSets ?? "reference,baseline");
                add("--horizons", normList(args.horizons) ?? "1");
                add("--max-configs", args.maxConfigs ?? 16);
                add("--limit", args.limit ?? 20);
                add("--json", args.json);
                break;

            case "evaluate":
                add("--tickers", normList(args.tickers));
                add("--targets", normList(args.targets));
                add("--folds", args.folds ?? 4);
                economics();
                statistics();
                add("--feature-set", args.featureSet ?? "all");
                add("--model-set", args.modelSet ?? "reference");
                add("--horizon", args.horizon ?? 1);
                add("--controls", args.controls);
                add("--shift-bars", args.shiftBars);
                add("--full-archive", args.fullArchive);
                add("--thresholds", args.thresholds);
                add("--cost-levels", args.costLevels);
                add("--canary-leak", args.canaryLeak);
                add("--limit", args.limit ?? 20);
                add("--json", args.json);
                break;
        }

        const repo = repoRoot(ctx);
        const script = join(repo, "tools", "model_lab.py");
        const env: NodeJS.ProcessEnv = {
            ...process.env,
            RL_CRYPTO_ROOT: repo,
            RL_CRYPTO_QUIET: "1",
        };

        // No shell: pass argv straight through. Skip `nix develop` if we are
        // already inside the dev shell.
        const inShell = !!process.env.IN_NIX_SHELL;
        const [cmd, cmdArgs] = inShell
            ? ["uv", ["run", "python", script, ...argv]]
            : ["nix", ["develop", "--command", "uv", "run", "python", script, ...argv]];

        const res = await run(cmd, cmdArgs, {
            cwd: repo,
            env,
            timeoutMs: TIMEOUTS_MS[action],
            abort: ctx?.abort,
        });

        const stdout = clean(res.stdout);
        const stderr = res.stderr.trim();
        const withNotes = (body: string) => [...notes, body].filter(Boolean).join("\n");

        if (res.spawnError) {
            return withNotes(`rl-model-lab failed to start '${cmd}': ${res.spawnError}`);
        }
        if (res.aborted) {
            return withNotes(
                `rl-model-lab: '${action}' was cancelled.${stdout ? `\nPartial output:\n${stdout}` : ""}`,
            );
        }
        if (res.timedOut) {
            const mins = Math.round(TIMEOUTS_MS[action] / 60_000);
            return withNotes(
                `rl-model-lab: '${action}' timed out after ${mins} min and was killed.` +
                    ` Try fewer tickers, fewer folds, or a smaller maxConfigs.` +
                    (stdout ? `\nPartial output:\n${stdout}` : ""),
            );
        }
        if (res.truncated) {
            return withNotes(
                `rl-model-lab: output exceeded ${MAX_OUTPUT_BYTES / 1024 / 1024} MB and the run was stopped. Use a smaller limit or json=false.`,
            );
        }
        if (res.code !== 0) {
            return withNotes(
                `rl-model-lab failed (exit ${res.code ?? res.signal}):\n${stdout}\n${stderr}`.trim(),
            );
        }

        if (promoteKey && !args.apply) dryRunPassed.add(promoteKey);
        return withNotes(stdout || "rl-model-lab: no output.");
    },
};
