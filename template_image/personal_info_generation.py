"""
End-to-end synthetic dataset generator (research-safe).

What this script does:
- Loads a YAML config describing:
  - modules to run
  - distribution specs for fields
  - portrait index integration (JSONL)
  - output profile (which fields to export, optional renaming)
- Generates N rows
- Writes CSV

Design goals:
- Make the YAML config the main interface (user edits it)
- Keep code modular (modules are building blocks)
- Support distribution constraints:
  - categorical with "quota" (approx exact counts over N)
  - uniform_int, normal, truncated_normal with i.i.d. sampling
- Integrate portrait index WITHOUT per-person "closest match":
  - use coarse buckets + round-robin for dataset balancing

This is intentionally NOT a document/ID generator.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import string

import numpy as np
import yaml
from faker import Faker

fake = Faker()


# ============================================================
# 1) IO helpers
# ============================================================

def load_yaml(path: str) -> dict:
    """Load YAML config (human editable, supports comments)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[dict]:
    """
    Load JSONL file into a list of dicts.
    Each line must be valid JSON.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# 2) Generic distribution sampling
# ============================================================

def weighted_choice(values: list, probs: list):
    """One draw from categorical distribution."""
    return random.choices(values, weights=probs, k=1)[0]


def sample_from_dist(dist_spec: dict) -> Any:
    """
    Sample a value from a distribution spec.

    Supported types:
      - uniform_int: integer in [min, max]
      - uniform_float: float in [min, max]
      - normal: Gaussian float
      - truncated_normal: Gaussian with rejection sampling in [min, max]
      - categorical: one of values with probs

    dist_spec format:
      dist:
        type: <type>
        params: {...}
        enforce: sample|quota   (quota handled separately for categorical)
    """
    t = dist_spec["type"]
    p = dist_spec.get("params", {}) or {}

    if t == "uniform_int":
        return random.randint(int(p["min"]), int(p["max"]))

    if t == "uniform_float":
        return random.uniform(float(p["min"]), float(p["max"]))

    if t == "normal":
        return float(np.random.normal(loc=float(p["mean"]), scale=float(p["std"])))

    if t == "truncated_normal":
        mean, std = float(p["mean"]), float(p["std"])
        lo, hi = float(p["min"]), float(p["max"])
        # Rejection sampling: try multiple times to land in bounds
        for _ in range(10000):
            x = float(np.random.normal(loc=mean, scale=std))
            if lo <= x <= hi:
                return x
        # Fallback: clamp if repeated rejection
        x = float(np.random.normal(loc=mean, scale=std))
        return max(lo, min(hi, x))

    if t == "categorical":
        values = p["values"]
        probs = p["probs"]
        return weighted_choice(values, probs)

    raise ValueError(f"Unsupported dist type: {t}")


def build_quota_stream(values: list, probs: list, n: int) -> list:
    """
    Build a length-n list that matches the categorical distribution by counts (as closely as possible).
    Example: probs=[0.5,0.5], n=10 -> 5 and 5 exactly.
    With rounding, remainder is distributed to largest fractional parts.
    """
    # raw expected counts
    raw = [float(p) * n for p in probs]
    counts = [int(math.floor(x)) for x in raw]
    remainder = n - sum(counts)

    # distribute remainder to largest fractional parts
    fracs = sorted([(i, raw[i] - counts[i]) for i in range(len(values))],
                   key=lambda x: x[1], reverse=True)
    for k in range(remainder):
        counts[fracs[k][0]] += 1

    stream = []
    for v, c in zip(values, counts):
        stream.extend([v] * c)
    random.shuffle(stream)
    return stream


# ============================================================
# 3) Date helpers
# ============================================================

def add_years_safe(d: date, years: int) -> date:
    """Adds years while handling Feb 29 safely."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)

def random_date_between(start: date, end: date) -> date:
    if end < start:
        start, end = end, start
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def format_date_choices(d: date, formats: list[dict]) -> str:
    patterns = [x["pattern"] for x in formats]
    probs = [x["prob"] for x in formats]
    pattern = random.choices(patterns, weights=probs, k=1)[0]
    return d.strftime(pattern)


def make_from_pattern(pattern: str) -> str:
    """
    Replace tokens:
      {LETTERS:n}
      {DIGITS:n}
    """
    out = pattern
    while "{LETTERS:" in out:
        n = int(out.split("{LETTERS:")[1].split("}")[0])
        letters = ''.join(random.choices(string.ascii_uppercase, k=n))
        out = out.replace(f"{{LETTERS:{n}}}", letters, 1)
    while "{DIGITS:" in out:
        n = int(out.split("{DIGITS:")[1].split("}")[0])
        digits = ''.join(random.choices(string.digits, k=n))
        out = out.replace(f"{{DIGITS:{n}}}", digits, 1)
    return out


def random_date_between(start: date, end: date) -> date:
    """Uniform random date between start and end (inclusive)."""
    if end < start:
        start, end = end, start
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def dob_from_age(age: int) -> date:
    """Pick a random DOB in the birth year consistent with integer age (approx)."""
    today = date.today()
    birth_year = today.year - age
    return random_date_between(date(birth_year, 1, 1), date(birth_year, 12, 31))


def format_date_choices(d: date, formats: List[dict]) -> str:
    """Format date by selecting one of several strftime patterns by probability."""
    patterns = [x["pattern"] for x in formats]
    probs = [x["prob"] for x in formats]
    return d.strftime(weighted_choice(patterns, probs))


def apply_capitalization(s: str, rule: str) -> str:
    """Simple name capitalization rules."""
    if rule == "first_letter_upper":
        return s[:1].upper() + s[1:].lower() if s else s
    if rule == "all_upper":
        return s.upper()
    if rule == "all_lower":
        return s.lower()
    if rule == "title_case":
        return s.title()
    return s


# ============================================================
# 4) Portrait bucketed + round-robin assignment
# ============================================================

def age_to_bin(age: float, edges: List[float]) -> str:
    """
    Convert age (float/int) to a bin label based on edges.
    edges like [0, 13, 18, 25, 35, 45, 60, 200].
    Returns a string label like "18-25".
    """
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= age < hi:
            return f"{int(lo)}-{int(hi)}"
    # fallback (shouldn't happen if last edge is big)
    return f"{int(edges[-2])}+"

def build_portrait_buckets(
    portrait_rows: List[dict],
    bucket_keys: List[str],
    age_binning_enabled: bool,
    age_edges: List[float],
) -> Tuple[Dict[Tuple[Any, ...], deque], List[str]]:
    """
    Build portrait buckets for round-robin selection.

    - bucket_keys: e.g., ["gender","ethnicity"] (COARSE)
    - if age binning enabled, add age_bin to bucket key

    Returns:
      buckets: dict[bucket_tuple] -> deque(images)
      all_images: list of all image filenames for fallback
    """
    buckets = defaultdict(list)
    all_images = []

    for r in portrait_rows:
        img = r.get("image")
        if not img:
            continue
        all_images.append(img)

        key_parts = []
        for k in bucket_keys:
            key_parts.append(r.get(k))
        if age_binning_enabled:
            key_parts.append(age_to_bin(float(r.get("age", 0.0)), age_edges))

        buckets[tuple(key_parts)].append(img)

    # shuffle each bucket and convert to deque for round-robin popping
    buckets_deque = {}
    for k, imgs in buckets.items():
        random.shuffle(imgs)
        buckets_deque[k] = deque(imgs)

    return buckets_deque, all_images


def select_portrait_bucket_round_robin(
    *,
    buckets: Dict[Tuple[Any, ...], deque],
    all_images: List[str],
    desired_key: Tuple[Any, ...],
    fallback: str,
) -> str:
    """
    Bucketed + round-robin selection:
      - Try to pop from the desired bucket
      - If empty/missing, fallback to:
          - random_global: random image from all_images
          - any_bucket: pop from any non-empty bucket
    """
    q = buckets.get(desired_key)
    if q and len(q) > 0:
        return q.popleft()

    if fallback == "any_bucket":
        # Find any bucket that still has items and pop from it
        for _, dq in buckets.items():
            if dq:
                return dq.popleft()

    # default fallback: random from all images
    return random.choice(all_images) if all_images else "portrait_placeholder.png"

from collections import defaultdict, deque
import random
from typing import Any, Dict, List, Tuple

# -------------------------
# Helpers reused from earlier
# -------------------------
def age_to_bin(age: float, edges: List[float]) -> str:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= age < hi:
            return f"{int(lo)}-{int(hi)}"
    return f"{int(edges[-2])}+"

def age_bin_center(bin_label: str) -> float:
    if '-' in bin_label:
        a, b = bin_label.split('-')
        return (int(a) + int(b)) / 2.0
    if bin_label.endswith('+'):
        a = int(bin_label[:-1])
        return float(a) + 10.0
    return float(bin_label)

def find_nearest_nonempty_bin(desired_bin: str, bins_available: List[str]) -> List[str]:
    desired_center = age_bin_center(desired_bin)
    def dist_key(b):
        return abs(age_bin_center(b) - desired_center)
    return sorted(bins_available, key=dist_key)

# -------------------------
# Build nested buckets (unchanged)
# -------------------------
def build_portrait_nested_buckets(
    portrait_rows: List[dict],
    bucket_keys: List[str],
    age_binning_enabled: bool,
    age_edges: List[float]
) -> Tuple[Dict[Tuple[Any, ...], Dict[str, deque]], List[str]]:
    nested = defaultdict(lambda: defaultdict(list))
    all_images = []

    for r in portrait_rows:
        img = r.get("image")
        if not img:
            continue
        all_images.append(img)
        outer = tuple(r.get(k) for k in bucket_keys)
        if age_binning_enabled:
            bin_label = age_to_bin(float(r.get("age", 0.0)), age_edges)
        else:
            bin_label = "all_ages"
        nested[outer][bin_label].append(img)

    nested_deques: Dict[Tuple[Any, ...], Dict[str, deque]] = {}
    for outer_k, inner_map in nested.items():
        nested_deques[outer_k] = {}
        for bin_label, imgs in inner_map.items():
            random.shuffle(imgs)
            nested_deques[outer_k][bin_label] = deque(imgs)

    return nested_deques, all_images

# -------------------------
# Rotation-based selector (reuses images)
# -------------------------
def _rotate_and_peek(dq: deque) -> str:
    """
    Return the leftmost element but rotate the deque so the element is consumed and
    appended to the right — this creates round-robin reuse.
    """
    if not dq:
        return None
    # peek leftmost
    val = dq[0]
    # rotate left by 1 so next call returns the next element (effectively moves pointer)
    dq.rotate(-1)
    return val

def _try_group_rotate(group_inner: Dict[str, deque], desired_bin: str) -> str:
    """
    Attempt to select an image from a group's inner bins using rotation (reuse).
    Returns the selected image filename or None.
    """
    # 1) desired bin exists
    dq = group_inner.get(desired_bin)
    if dq and len(dq) > 0:
        return _rotate_and_peek(dq)

    # 2) nearest non-empty bin
    bins_available = [b for b, q in group_inner.items() if q and len(q) > 0]
    if bins_available:
        ordered = find_nearest_nonempty_bin(desired_bin, bins_available)
        for b in ordered:
            dq2 = group_inner.get(b)
            if dq2 and len(dq2) > 0:
                return _rotate_and_peek(dq2)

    # 3) any non-empty bin (rotate from first found)
    for b, q in group_inner.items():
        if q and len(q) > 0:
            return _rotate_and_peek(q)

    return None

def select_portrait_by_gender_race_agebin_rotate(
    *,
    nested_buckets: Dict[Tuple[Any, ...], Dict[str, deque]],
    all_images: List[str],
    desired_gender: str,
    desired_ethnicity: str,
    desired_age: float,
    age_binning_enabled: bool,
    age_edges: List[float],
    fallback: str = "random_global"
) -> str:
    """
    Rotation (reuse) version of the selector.

    Behavior:
    - Try exact (gender, ethnicity) + desired age bin, reusing images by rotation.
    - If empty, search nearest age bins, then any bin within group.
    - If no images for (gender, ethnicity), try gender-only groups similarly.
    - Then fallback to any_bucket or random_global.
    """
    outer_key = (desired_gender, desired_ethnicity)
    desired_bin = age_to_bin(float(desired_age), age_edges) if age_binning_enabled else "all_ages"

    # helper to attempt group selection
    def try_group(key):
        inner = nested_buckets.get(key)
        if not inner:
            return None
        return _try_group_rotate(inner, desired_bin)

    # 1) exact gender+ethnicity
    img = try_group(outer_key)
    if img:
        return img

    # 2) gender-only (iterate over keys with matching gender)
    for key in list(nested_buckets.keys()):
        if key[0] == desired_gender:
            img = try_group(key)
            if img:
                return img

    # 3) fallback across any buckets (if requested)
    if fallback == "any_bucket":
        for key in list(nested_buckets.keys()):
            img = try_group(key)
            if img:
                return img

    # 4) final fallback: random global
    if all_images:
        return random.choice(all_images)
    return "portrait_placeholder.png"


# ============================================================
# 5) Module system
# ============================================================

@dataclass
class QuotaStreams:
    """
    Stores per-field quota streams for categorical variables.
    Keyed by (module_name, field_name).
    """
    streams: Dict[Tuple[str, str], List[Any]]

    def get(self, module: str, field: str, idx: int) -> Any:
        return self.streams[(module, field)][idx]


def prepare_quota_streams(cfg: dict, selected_modules: List[str], n: int) -> QuotaStreams:
    """
    Precompute quota streams for categorical distributions with enforce=quota.
    This ensures dataset-level distribution satisfaction for those fields.
    """
    streams: Dict[Tuple[str, str], List[Any]] = {}
    modules_cfg = cfg["modules"]

    for m in selected_modules:
        mcfg = modules_cfg[m]
        params = mcfg.get("params", {}) or {}

        # scan params to find fields that look like:
        #   <field>:
        #     dist: { type: categorical, ..., enforce: quota }
        for field_name, field_cfg in params.items():
            if not isinstance(field_cfg, dict):
                continue
            dist = field_cfg.get("dist")
            if not isinstance(dist, dict):
                continue
            if dist.get("type") == "categorical" and dist.get("enforce") == "quota":
                values = dist["params"]["values"]
                probs = dist["params"]["probs"]
                streams[(m, field_name)] = build_quota_stream(values, probs, n)

    return QuotaStreams(streams=streams)


def resolve_module_order(cfg: dict, selected_modules: List[str]) -> List[str]:
    """
    Topologically order modules by requires/provides.
    This keeps the pipeline stable even if users reorder in config.
    """
    modules_cfg = cfg["modules"]

    provides = {m: set(modules_cfg[m].get("provides", [])) for m in selected_modules}
    requires = {m: set(modules_cfg[m].get("requires", [])) for m in selected_modules}

    # map field -> modules providing it
    providers: Dict[str, List[str]] = defaultdict(list)
    for m in selected_modules:
        for f in provides[m]:
            providers[f].append(m)

    # build deps: m depends on modules that provide its requirements
    deps = {m: set() for m in selected_modules}
    for m in selected_modules:
        for r in requires[m]:
            # if requirement isn't provided by another module, error
            provs = providers.get(r, [])
            if not provs:
                raise ValueError(f"Module '{m}' requires '{r}' but no selected module provides it.")
            for p in provs:
                if p != m:
                    deps[m].add(p)

    # Kahn’s algorithm
    indeg = {m: len(deps[m]) for m in selected_modules}
    queue = [m for m in selected_modules if indeg[m] == 0]
    order = []

    while queue:
        x = queue.pop(0)
        order.append(x)
        for m in selected_modules:
            if x in deps[m]:
                deps[m].remove(x)
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)

    if len(order) != len(selected_modules):
        raise ValueError("Cycle detected in module dependencies. Check provides/requires in config.")
    return order


def run_module(
    module_name: str,
    cfg: dict,
    ctx: dict,
    row_idx: int,
    quota: QuotaStreams,
    portrait_state: dict
) -> None:
    """
    Execute one module; update ctx dict in-place.

    - row_idx is used for quota streams (so each row consumes the next quota value)
    - portrait_state holds buckets/all_images for portrait selection
    """
    mcfg = cfg["modules"][module_name]
    params = mcfg.get("params", {}) or {}

    # -------------------------
    # person_core
    # -------------------------
    if module_name == "person_core":
        cap = params.get("name_capitalization", "first_letter_upper")

        # gender: quota or sample
        gender_dist = params["gender"]["dist"]
        if gender_dist.get("enforce") == "quota":
            gender = quota.get("person_core", "gender", row_idx)
        else:
            gender = sample_from_dist(gender_dist)

        # ethnicity: quota or sample
        eth_dist = params["ethnicity"]["dist"]
        if eth_dist.get("enforce") == "quota":
            ethnicity = quota.get("person_core", "ethnicity", row_idx)
        else:
            ethnicity = sample_from_dist(eth_dist)

        # names: use Faker; optionally conditioned on gender for variety
        if gender == "male":
            first = fake.first_name_male()
        elif gender == "female":
            first = fake.first_name_female()
        else:
            first = fake.first_name()
        last = fake.last_name()

        ctx["gender"] = gender
        ctx["ethnicity"] = ethnicity
        ctx["first_name"] = apply_capitalization(first, cap)
        ctx["last_name"] = apply_capitalization(last, cap)
        return

    # -------------------------
    # dob
    # -------------------------
    if module_name == "dob":
        age_dist = params["age"]["dist"]
        age = int(sample_from_dist(age_dist))
        dob = dob_from_age(age)

        ctx["age"] = age
        ctx["dob_iso"] = dob.isoformat()
        ctx["dob"] = format_date_choices(dob, params["dob_formats"])
        return

    if module_name == "issue_location":
        p = cfg["modules"]["issue_location"].get("params", {})
        values = p.get("values", [])
        probs = p.get("probs", [])
    
        if not values:
            ctx["issue_place"] = None
            return
    
        ctx["issue_place"] = random.choices(values, weights=probs, k=1)[0]
        return
    
    if module_name == "svk_numbers":
        p = cfg["modules"]["svk_numbers"].get("params", {})
    
        number_pattern = p.get("number_pattern", "{LETTERS:2}{DIGITS:6}")
        id_number_pattern = p.get("id_number_pattern", "{DIGITS:6}/{DIGITS:4}")
        sig_len = int(p.get("signature_prefix_len", 5))
    
        ctx["doc_number"] = make_from_pattern(number_pattern)
        ctx["local_id_number"] = make_from_pattern(id_number_pattern)
    
        first_name = ctx.get("first_name", "")
        ctx["signature"] = first_name[:sig_len]
    
        return
    
    if module_name == "doc_dates":
        p = cfg["modules"]["doc_dates"].get("params", {}) or {}
    
        # issue date: pick a random date within the last X years
        years_back = random.randint(int(p["issue_years_back_min"]), int(p["issue_years_back_max"]))
        today = date.today()
        earliest = date(today.year - years_back, today.month, today.day)
        issue = random_date_between(earliest, today)
    
        # expiry date: add a random number of years
        expiry_years = random.randint(int(p["expiry_years_min"]), int(p["expiry_years_max"]))
        expiry = add_years_safe(issue, expiry_years)
    
        fmts = p.get("formats", [{"pattern": "%Y-%m-%d", "prob": 1.0}])
        ctx["issue_date"] = format_date_choices(issue, fmts)
        ctx["expiry_date"] = format_date_choices(expiry, fmts)
        return


    # -------------------------
    # traits
    # -------------------------
    if module_name == "traits":
        for field in ["eye_color", "hair_color", "hair_length", "bald"]:
            dist = params[field]["dist"]
            val = sample_from_dist(dist)
            # make bald a rounded float for nicer CSV
            if field == "bald":
                val = round(float(val), 3)
            ctx[field] = val
        return

    # -------------------------
    # portrait_from_index  (bucketed + round-robin)
    # -------------------------
    if module_name == "portrait_from_index":
        pa = cfg["portrait_assignment"]
        bucket_keys = pa["bucket_keys"]
        age_cfg = pa.get("age_binning", {}) or {}
        age_enabled = bool(age_cfg.get("enabled", False))
        age_edges = age_cfg.get("bins", [])
        fallback = pa.get("fallback", "random_global")
        strategy = pa.get("strategy", "bucket_round_robin")

        # Build desired key tuple from ctx:
        key_parts = [ctx.get(k) for k in bucket_keys]
        if age_enabled:
            key_parts.append(age_to_bin(float(ctx.get("age", 0.0)), age_edges))
        desired_key = tuple(key_parts)

        if strategy == "random":
            ctx["portrait"] = random.choice(portrait_state["all_images"])
            return

        # Default strategy: bucket_round_robin
        img = select_portrait_by_gender_race_agebin_rotate(
            nested_buckets=portrait_state["nested_buckets"],
            all_images=portrait_state["all_images"],
            desired_gender=ctx.get("gender"),
            desired_ethnicity=ctx.get("ethnicity"),
            desired_age=ctx.get("age", 30),
            age_binning_enabled=bool(pa.get("age_binning", {}).get("enabled", False)),
            age_edges=pa.get("age_binning", {}).get("bins", []),
            fallback=pa.get("fallback","random_global")
        )
        ctx["portrait"] = img


        #ctx["portrait"] = select_portrait_bucket_round_robin(
        #    buckets=portrait_state["buckets"],
        #    all_images=portrait_state["all_images"],
        #    desired_key=desired_key,
        #    fallback=fallback,
        #)
        return

    raise ValueError(f"No implementation for module '{module_name}'")


# ============================================================
# 6) Output rendering + CSV writing
# ============================================================
def render_profile(ctx: dict, profile: dict) -> dict:
    """
    Create an output dict according to profile:
      - fields: which canonical fields to export (subset & order)
      - key_map: rename canonical -> output keys
      - formats: optional output-time formatting tweaks
    This version also supports:
      - formats.dob_date_pattern (strftime override)
      - formats.gender_encoding (map canonical -> output encoding)
    """
    fields = profile.get("fields", [])
    key_map = profile.get("key_map", {}) or {}
    fmt = profile.get("formats", {}) or {}

    # Optional format overrides
    dob_pattern = fmt.get("dob_date_pattern")
    # gender_encoding might be like: { "male": "M", "female": "F", "nonbinary": "X" }
    gender_map = fmt.get("gender_encoding", {}) or {}

    out = {}
    for f in fields:
        out_key = key_map.get(f, f)
        val = ctx.get(f)

        # 1) DOB formatting override (if dob_iso available)
        if f == "dob" and dob_pattern and ctx.get("dob_iso"):
            try:
                d = datetime.fromisoformat(ctx["dob_iso"]).date()
                val = d.strftime(dob_pattern)
            except Exception:
                # fallback: keep existing val (if any)
                pass

        # 2) Gender encoding mapping
        if f == "gender" and isinstance(val, str):
            # be liberal with capitalization: map using lowercase keys
            mapped = gender_map.get(val)
            if mapped is None:
                mapped = gender_map.get(val.lower())
            if mapped is None:
                mapped = gender_map.get(val.capitalize())
            # if found mapping, use it; otherwise keep original value
            if mapped is not None:
                val = mapped

        out[out_key] = val

    # If you want blank columns to appear in CSV, return out (includes None).
    # If you prefer to drop None fields (previous behavior), uncomment next line:
    # return {k: v for k, v in out.items() if v is not None}

    return out


def render_profile1(ctx: dict, profile: dict) -> dict:
    """
    Create an output dict according to profile:
      - fields: which canonical fields to export (subset/order)
      - key_map: rename fields
      - formats: optional output-time formatting tweaks
    """
    fields = profile.get("fields", [])
    key_map = profile.get("key_map", {}) or {}
    fmt = profile.get("formats", {}) or {}

    # Optional: override DOB string formatting at export time
    dob_pattern = fmt.get("dob_date_pattern")

    out = {}
    for f in fields:
        out_key = key_map.get(f, f)
        val = ctx.get(f)

        # Example format override: DOB from dob_iso -> dob_pattern
        if f == "dob" and dob_pattern and ctx.get("dob_iso"):
            d = datetime.fromisoformat(ctx["dob_iso"]).date()
            val = d.strftime(dob_pattern)

        out[out_key] = val

    # drop Nones
    return {k: v for k, v in out.items() if v is not None}


def write_csv(path: str, rows: List[dict]) -> None:
    """Write rows to CSV with union-of-keys columns (stable, sorted)."""
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ============================================================
# 7) CLI entry point
# ============================================================

def show_capabilities(cfg: dict) -> None:
    """Print what the system provides (modules + profiles)."""
    print("Modules available:")
    for name, m in cfg["modules"].items():
        print(f" - {name}: {m.get('description','')}")
        print(f"   provides: {m.get('provides', [])}")
        print(f"   requires: {m.get('requires', [])}")
    print("\nOutput profiles available:")
    for name, p in cfg["output_profiles"].items():
        print(f" - {name}: {p.get('description','')}")
        print(f"   fields: {p.get('fields', [])}")


def pii_generation(cfg):

    if __debug__:
        show_capabilities(cfg)

    # Seed control (reproducibility)
    seed = cfg["ui"].get("default_seed", 42) 
    random.seed(seed)
    np.random.seed(seed)

    selected_modules = list(cfg["ui"]["default_modules"])

    # Validate module names
    for m in selected_modules:
        if m not in cfg["modules"]:
            raise ValueError(f"Unknown module '{m}'. Use --show to list modules.")

    # Resolve order based on requires/provides
    ordered_modules = resolve_module_order(cfg, selected_modules)

    # Choose output profile
    profile_name = cfg["ui"].get("default_profile")
    if profile_name not in cfg["output_profiles"]:
        raise ValueError(f"Unknown profile '{profile_name}'. Use --show to list profiles.")
    profile = cfg["output_profiles"][profile_name]

    # Precompute quota streams (for enforce=quota categorical fields)
    number_samples = cfg.get('number_samples', 10)
    quota = prepare_quota_streams(cfg, selected_modules, number_samples)


    # Load portrait index and build buckets once (if portrait module is used)
    portrait_state = {"buckets": {}, "all_images": []}
    if "portrait_from_index" in selected_modules:
        pa = cfg["portrait_assignment"]
        portrait_rows = load_jsonl(pa["jsonl_path"])
        age_cfg = pa.get("age_binning", {}) or {}

        nested_buckets, all_images = build_portrait_nested_buckets(
            portrait_rows=portrait_rows,
            bucket_keys=pa["bucket_keys"],
            age_binning_enabled=bool(age_cfg.get("enabled", False)),
            age_edges=age_cfg.get("bins", []),
        )
        portrait_state = {"nested_buckets": nested_buckets, "all_images": all_images}

#        buckets, all_images = build_portrait_buckets(
#            portrait_rows=portrait_rows,
#            bucket_keys=pa["bucket_keys"],
#            age_binning_enabled=bool(age_cfg.get("enabled", False)),
#            age_edges=age_cfg.get("bins", []),
#        )
#        portrait_state["buckets"] = buckets
#        portrait_state["all_images"] = all_images

    # Generate rows
    rows = []
    for i in range(number_samples):
        ctx = {}
        for m in ordered_modules:
            run_module(
                module_name=m,
                cfg=cfg,
                ctx=ctx,
                row_idx=i,
                quota=quota,
                portrait_state=portrait_state
            )
        rows.append(render_profile(ctx, profile))

    return rows

