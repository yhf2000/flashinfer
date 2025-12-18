from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


_TREE_KERNEL_CASES = [
    # Minimal sizes / pagination boundaries.
    dict(
        batch_size=1,
        seq_len=1,
        page_size=1,
        group_size=1,
        head_dim=64,
        num_kv_heads=1,
        max_tree_height=2,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=1,
        seq_len=2,
        page_size=16,  # page_size > seq_len
        group_size=1,
        head_dim=64,
        num_kv_heads=4,
        max_tree_height=5,  # 4 ancestors => exactly one packed word
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=1,
        seq_len=3,
        page_size=2,
        group_size=2,
        head_dim=64,
        num_kv_heads=4,
        max_tree_height=6,  # 5 ancestors => crosses 4-byte packing boundary
        pos_encoding_mode="NONE",
    ),
    # Depth == 1 (no ancestors): mask degenerates to identity.
    dict(
        batch_size=1,
        seq_len=16,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=1,
        pos_encoding_mode="NONE",
        max_depth=1,
        max_width=254,
    ),
    # Non power-of-two / remainder pages.
    dict(
        batch_size=2,
        seq_len=17,
        page_size=7,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=2,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=2,
        seq_len=17,
        page_size=16,
        group_size=4,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=2,
        seq_len=33,
        page_size=16,
        group_size=2,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=10,  # another packing boundary
        pos_encoding_mode="NONE",
    ),
    # Stress long sequences / many pages.
    dict(
        batch_size=1,
        seq_len=128,
        page_size=1,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=2,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=1,
        seq_len=257,
        page_size=1,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=18,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=2,
        seq_len=512,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=5,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=1,
        seq_len=1024,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=2,
        pos_encoding_mode="NONE",
        max_width=64,
    ),
    # Wide levels to push inner_id close to uint8 max (<= 254; reserve 255).
    dict(
        batch_size=1,
        seq_len=300,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=5,
        pos_encoding_mode="NONE",
        max_depth=2,
        max_width=254,
    ),
    # Larger batch size.
    dict(
        batch_size=16,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
    ),
    # Stress group_size / GQA and head_dim.
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=8,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=13,
        pos_encoding_mode="NONE",
    ),
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=4,
        head_dim=256,
        num_kv_heads=4,
        max_tree_height=14,
        pos_encoding_mode="NONE",
    ),
    # Non-contiguous tree_info/KV variants.
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
        tree_info_variant="word_stride2",
    ),
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
        tree_info_variant="entry_stride2",
    ),
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
        kv_variant="noncontig_head",
    ),
    # Extra sche_len words (kernel should ignore padding).
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
        extra_sche_words=7,
    ),
    # Position_id near 23-bit packing boundary (but still in-range).
    dict(
        batch_size=2,
        seq_len=64,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=9,
        pos_encoding_mode="NONE",
        pos_base=(1 << 22) - 64,
    ),
    # RoPE cases: limited set to avoid exploding JIT variants.
    dict(
        batch_size=2,
        seq_len=33,
        page_size=16,
        group_size=2,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=10,
        pos_encoding_mode="ROPE_LLAMA",
    ),
    dict(
        batch_size=1,
        seq_len=257,
        page_size=16,
        group_size=1,
        head_dim=128,
        num_kv_heads=4,
        max_tree_height=18,
        pos_encoding_mode="ROPE_LLAMA",
        pos_base=4096,
    ),
]


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    # Warm up only the JIT variants that are actually used by cases below, to keep compilation
    # overhead reasonable while still avoiding per-test compile spikes.
    # (group_size/num_heads/seq_len/page_size are runtime parameters and do not affect JIT URI.)
    cases = _TREE_KERNEL_CASES
    tree_variants = set()
    prefill_head_dims = set()
    for c in cases:
        head_dim = int(c["head_dim"])
        max_tree_height = int(c["max_tree_height"])
        twpq = 1 + ((max_tree_height - 1 + 3) // 4)
        pos_mode = 0 if c.get("pos_encoding_mode", "NONE") == "NONE" else 1
        tree_variants.add((head_dim, pos_mode, twpq))
        prefill_head_dims.add(head_dim)

    specs = []
    for head_dim, pos_mode, twpq in sorted(tree_variants):
        specs.append(
            flashinfer.jit.gen_batch_tree_module(
                "fa2",
                torch.float16,
                torch.float16,
                torch.float16,
                torch.int32,
                head_dim,
                head_dim,
                twpq,
                pos_mode,
                False,  # use_sliding_window
                False,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )
        )
    for head_dim in sorted(prefill_head_dims):
        specs.append(
            flashinfer.prefill.gen_batch_prefill_module(
                "fa2",
                torch.float16,
                torch.float16,
                torch.float16,
                torch.int32,
                head_dim,
                head_dim,
                0,  # pos_encoding_mode NONE (baseline uses pre-rotated q/k for ROPE cases)
                False,  # use_sliding_window
                False,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )
        )
    flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


@dataclass(frozen=True)
class _TreeSequence:
    # Token-major arrays for a single sequence (length = seq_len).
    position_id: List[int]
    inner_id: List[int]
    # ancestors[t][d] is the ancestor inner_id at distance (d+1) in position_id space.
    # i.e. if diff = q_pos - kv_pos == d+1, we compare kv_inner with ancestors[q][d].
    ancestors: List[List[int]]


def _random_widths_sum_to(
    *,
    seq_len: int,
    max_depth: int,
    max_width: int,
    rng: torch.Generator,
) -> List[int]:
    assert seq_len >= 1
    max_depth = min(max_depth, seq_len)
    if max_width <= 0 or max_width > 254:
        raise ValueError("max_width must be in [1, 254] to keep inner_id in uint8 and reserve 255")

    min_depth = max(1, (seq_len + max_width - 1) // max_width)
    if min_depth > max_depth:
        raise ValueError(
            f"Infeasible (seq_len={seq_len}, max_depth={max_depth}, max_width={max_width})"
        )

    depth = int(torch.randint(min_depth, max_depth + 1, (1,), generator=rng).item())
    widths = [1] * depth
    remaining = seq_len - depth
    # Randomly distribute remaining tokens, respecting max_width.
    while remaining > 0:
        p = int(torch.randint(0, depth, (1,), generator=rng).item())
        if widths[p] < max_width:
            widths[p] += 1
            remaining -= 1
    return widths


def _generate_random_tree_sequence(
    *,
    seq_len: int,
    max_depth: int,
    max_width: int,
    rng: torch.Generator,
) -> _TreeSequence:
    """
    Generate a random tree whose edges only connect adjacent position_id levels.

    - position_id is the level index (0..depth-1), and there can be multiple nodes per level.
    - inner_id is unique within each position_id level and fits in uint8 (<= 254).
    - Each node at level p>0 picks a random parent from level p-1, producing a unique ancestor
      chain for every position distance.
    """
    widths = _random_widths_sum_to(
        seq_len=seq_len, max_depth=max_depth, max_width=max_width, rng=rng
    )
    depth = len(widths)
    invalid_inner = 255  # reserved "no ancestor" marker (max_width <= 254 ensures no collision)

    @dataclass
    class _Node:
        pos: int
        inner: int
        parent: "_Node | None"
        anc: List[int]  # nearest-first ancestor inner_id list

    levels: List[List[_Node]] = []
    for p, w in enumerate(widths):
        cur: List[_Node] = []
        if p == 0:
            for inner in range(w):
                cur.append(_Node(pos=p, inner=inner, parent=None, anc=[]))
        else:
            prev = levels[p - 1]
            prev_w = len(prev)
            for inner in range(w):
                parent_idx = int(torch.randint(0, prev_w, (1,), generator=rng).item())
                parent = prev[parent_idx]
                cur.append(_Node(pos=p, inner=inner, parent=parent, anc=[parent.inner] + parent.anc))
        levels.append(cur)

    nodes = [n for lvl in levels for n in lvl]
    assert len(nodes) == seq_len

    position_id = [n.pos for n in nodes]
    inner_id = [n.inner for n in nodes]

    # Build full ancestor arrays up to max possible distance (depth - 1).
    full_anc_len = max(0, depth - 1)
    ancestors: List[List[int]] = []
    for n in nodes:
        a = n.anc[:full_anc_len]
        if len(a) < full_anc_len:
            a = a + [invalid_inner] * (full_anc_len - len(a))
        ancestors.append(a)
    return _TreeSequence(position_id=position_id, inner_id=inner_id, ancestors=ancestors)


def _generate_structured_tree_sequence_with_main_path(
    *,
    depth: int,
    width: int,
    main_inner_by_pos: List[int],
) -> _TreeSequence:
    """
    Build a deterministic tree to stress ancestor byte packing.

    - There are `width` nodes per level (position_id), inner_id = 0..width-1.
    - Most nodes follow a "same-inner" chain (parent inner_id stays constant across levels).
    - One special node per level (inner_id = main_inner_by_pos[pos]) follows a main path whose
      parent at pos-1 is main_inner_by_pos[pos-1], so its ancestor inner_ids vary with diff.
    """
    if depth <= 0:
        raise ValueError("depth must be > 0")
    if width <= 0 or width > 254:
        raise ValueError("width must be in [1, 254] to keep inner_id in uint8 and reserve 255")
    if len(main_inner_by_pos) != depth:
        raise ValueError("main_inner_by_pos must have length == depth")
    if any((x < 0 or x >= width) for x in main_inner_by_pos):
        raise ValueError("main_inner_by_pos entries must be in [0, width)")

    invalid_inner = 255
    full_anc_len = max(0, depth - 1)

    position_id: List[int] = []
    inner_id: List[int] = []
    ancestors: List[List[int]] = []
    for p in range(depth):
        main_inner = int(main_inner_by_pos[p])
        for j in range(width):
            position_id.append(p)
            inner_id.append(j)
            anc: List[int] = []
            for d in range(full_anc_len):
                if d >= p:
                    anc.append(invalid_inner)
                    continue
                if j == main_inner:
                    anc.append(int(main_inner_by_pos[p - 1 - d]))
                else:
                    anc.append(j)
            ancestors.append(anc)
    return _TreeSequence(position_id=position_id, inner_id=inner_id, ancestors=ancestors)


def _build_tree_info_pages(
    *,
    batch_trees: List[_TreeSequence],
    max_tree_height: int,
    page_size: int,
    pos_base: int = 0,
    extra_sche_words: int = 0,
    tree_info_variant: str = "contiguous",  # contiguous | word_stride2 | entry_stride2
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build `tree_info` (paged like KV-cache) and `pos_ids` (per token).

    tree_info layout (uint32):
      word0: [is_build(1b) | token_id(31b)]                (not used by attention)
      word1: [is_delete(1b) | is_ghost(1b) | position_id(22b) | inner_id(8b)]
      word2..: packed ancestor bytes (4 per uint32), nearest-first
    """
    if max_tree_height <= 0:
        raise ValueError("max_tree_height must be >= 1")
    anc_array_len = max(0, int(max_tree_height) - 1)

    batch_size = len(batch_trees)
    seq_len = len(batch_trees[0].position_id)
    assert all(len(t.position_id) == seq_len for t in batch_trees)

    anc_words = (anc_array_len + 3) // 4
    max_sche_len = 2 + anc_words + extra_sche_words  # word0 + word1 + anc_words + padding
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    padded_len = num_pages_per_seq * page_size

    # Token-major padded tensors, then reshape to pages.
    # Use int64 for construction because some ops (e.g. arange for uint32) are not available on CUDA.
    tree_info_tok = torch.zeros(
        (batch_size, padded_len, max_sche_len), device=device, dtype=torch.int64
    )
    pos_ids = torch.zeros((batch_size, padded_len), device=device, dtype=torch.int32)

    # is_delete/is_ghost are 0 for randomized tests (see focused tests below).
    is_delete = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int64)
    is_ghost = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int64)

    for b, t in enumerate(batch_trees):
        pos = torch.tensor(t.position_id, device=device, dtype=torch.int32) + int(pos_base)
        inner = torch.tensor(t.inner_id, device=device, dtype=torch.int32)
        pos_ids[b, :seq_len] = pos

        # word0: is_build=1, token_id=token_idx within the sequence.
        token_id = torch.arange(seq_len, device=device, dtype=torch.int64)
        tree_info_tok[b, :seq_len, 0] = (1 << 31) | (token_id & 0x7FFF_FFFF)

        # word1: [is_delete|position_id|inner_id]
        tree_info_tok[b, :seq_len, 1] = (
            (is_delete[b] << 31)
            | (is_ghost[b] << 30)
            | ((pos.to(torch.int64) & ((1 << 22) - 1)) << 8)
            | (inner.to(torch.int64) & 0xFF)
        )

        # Ancestors: truncate/pad to anc_array_len, pack 4 bytes per uint32.
        # ancestors[t][d] corresponds to diff == d+1.
        anc_full = torch.tensor(t.ancestors, device=device, dtype=torch.int32)  # [L, full_anc_len]
        if anc_full.numel() == 0:
            anc = torch.full((seq_len, anc_array_len), 255, device=device, dtype=torch.int32)
        else:
            anc = anc_full[:, :anc_array_len]
            if anc.shape[1] < anc_array_len:
                anc = torch.cat(
                    [
                        anc,
                        torch.full(
                            (seq_len, anc_array_len - anc.shape[1]),
                            255,
                            device=device,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=1,
                )
        anc = anc.to(torch.int64)
        anc_padded = torch.zeros((seq_len, anc_words * 4), device=device, dtype=torch.int64)
        anc_padded[:, :anc_array_len] = anc
        anc_padded = anc_padded.view(seq_len, anc_words, 4)
        anc_words_u32 = (
            (anc_padded[:, :, 0] & 0xFF)
            | ((anc_padded[:, :, 1] & 0xFF) << 8)
            | ((anc_padded[:, :, 2] & 0xFF) << 16)
            | ((anc_padded[:, :, 3] & 0xFF) << 24)
        )
        tree_info_tok[b, :seq_len, 2 : 2 + anc_words] = anc_words_u32

    # Reshape to [total_pages, page_size, max_sche_len]
    tree_info_pages = tree_info_tok.to(torch.uint32).view(
        batch_size, num_pages_per_seq, page_size, max_sche_len
    )
    tree_info_pages = tree_info_pages.reshape(
        batch_size * num_pages_per_seq, page_size, max_sche_len
    )

    if tree_info_variant == "contiguous":
        pass
    elif tree_info_variant == "word_stride2":
        tmp = torch.empty(
            (tree_info_pages.shape[0], tree_info_pages.shape[1], tree_info_pages.shape[2] * 2),
            device=device,
            dtype=torch.uint32,
        )
        tmp[:, :, ::2] = tree_info_pages
        tree_info_pages = tmp[:, :, ::2]
    elif tree_info_variant == "entry_stride2":
        tmp = torch.empty(
            (tree_info_pages.shape[0], tree_info_pages.shape[1] * 2, tree_info_pages.shape[2]),
            device=device,
            dtype=torch.uint32,
        )
        tmp[:, ::2, :] = tree_info_pages
        tree_info_pages = tmp[:, ::2, :]
    else:
        raise ValueError(f"Unknown tree_info_variant: {tree_info_variant}")

    pos_ids = pos_ids[:, :seq_len].reshape(batch_size * seq_len)
    return tree_info_pages, pos_ids


def _pack_paged_kv_from_tokens(
    *,
    k_tok: torch.Tensor,  # [B*L, H_KV, D]
    v_tok: torch.Tensor,  # [B*L, H_KV, D]
    batch_size: int,
    seq_len: int,
    page_size: int,
    kv_variant: str = "contiguous",  # contiguous | noncontig_head
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = k_tok.shape[1]
    head_dim = k_tok.shape[2]
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    padded_len = num_pages_per_seq * page_size

    k_padded = torch.zeros(
        (batch_size, padded_len, num_kv_heads, head_dim), device=k_tok.device, dtype=k_tok.dtype
    )
    v_padded = torch.zeros_like(k_padded)
    k_padded[:, :seq_len] = k_tok.view(batch_size, seq_len, num_kv_heads, head_dim)
    v_padded[:, :seq_len] = v_tok.view(batch_size, seq_len, num_kv_heads, head_dim)

    k_cache = k_padded.view(batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim)
    v_cache = v_padded.view(batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim)
    k_cache = k_cache.reshape(batch_size * num_pages_per_seq, page_size, num_kv_heads, head_dim)
    v_cache = v_cache.reshape(batch_size * num_pages_per_seq, page_size, num_kv_heads, head_dim)
    if kv_variant == "contiguous":
        return k_cache, v_cache
    if kv_variant == "noncontig_head":
        # Make the head dimension strided (similar to other non-contig KV tests in this repo).
        tmp_k = torch.empty(
            (k_cache.shape[0], k_cache.shape[1], k_cache.shape[2] * 2, k_cache.shape[3]),
            device=k_cache.device,
            dtype=k_cache.dtype,
        )
        tmp_v = torch.empty_like(tmp_k)
        tmp_k[:, :, ::2, :] = k_cache
        tmp_v[:, :, ::2, :] = v_cache
        return tmp_k[:, :, ::2, :], tmp_v[:, :, ::2, :]
    raise ValueError(f"Unknown kv_variant: {kv_variant}")
    return k_cache, v_cache


def _tree_mask_dense(
    *,
    batch_trees: List[_TreeSequence],
    max_tree_height: int,
    pos_base: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Dense custom mask (flattened) matching tree_mask rules."""
    anc_array_len = max(0, int(max_tree_height) - 1)
    batch_size = len(batch_trees)
    seq_len = len(batch_trees[0].position_id)
    pos = torch.stack(
        [
            torch.tensor(t.position_id, device=device, dtype=torch.int32) + int(pos_base)
            for t in batch_trees
        ],
        dim=0,
    )  # [B, L]
    inner = torch.stack(
        [torch.tensor(t.inner_id, device=device, dtype=torch.int32) for t in batch_trees], dim=0
    )  # [B, L]

    invalid_inner = 255
    # Ancestors can have different "depth-1" across sequences; build a fixed [B, L, anc_array_len]
    # tensor by truncating/padding per sequence.
    anc = torch.full(
        (batch_size, seq_len, anc_array_len),
        invalid_inner,
        device=device,
        dtype=torch.int32,
    )
    for b, t in enumerate(batch_trees):
        anc_b = torch.tensor(t.ancestors, device=device, dtype=torch.int32)
        if anc_b.numel() == 0:
            continue
        anc_b = anc_b[:, :anc_array_len]
        anc[b, :, : anc_b.shape[1]] = anc_b

    # is_delete is all 0 in randomized tests.
    is_delete = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)

    pos_q = pos[:, :, None]  # [B, Q, 1]
    pos_k = pos[:, None, :]  # [B, 1, K]
    inner_q = inner[:, :, None]
    inner_k = inner[:, None, :]

    valid = (~is_delete[:, None, :]) & (pos_k <= pos_q)
    diff = pos_q - pos_k  # >=0 when valid
    valid = valid & (diff <= anc_array_len)

    diff0 = diff == 0
    match0 = diff0 & (inner_k == inner_q)

    if anc_array_len == 0:
        # MAX_TREE_HEIGHT == 1: only allow diff == 0 (no ancestor bytes are meaningful/available).
        mask = valid & match0
        return mask.reshape(-1)

    diffgt0 = diff > 0
    # Lookup ancestor inner_id for each (q,k) by selecting anc[b, q, diff-1].
    #
    # Avoid expanding to [B, Q, K, A] (which is memory-heavy for large seq_len). Instead, treat
    # anc as a flattened 1D array and use a linearized index:
    #   flat_idx = ((b * L + q) * A) + (diff-1)
    diff_slot = (diff - 1).clamp(min=0, max=anc_array_len - 1).to(torch.int64)  # [B,Q,K]
    row_id = (
        torch.arange(batch_size, device=device, dtype=torch.int64)[:, None] * seq_len
        + torch.arange(seq_len, device=device, dtype=torch.int64)[None, :]
    )  # [B,Q]
    row_base = (row_id * anc_array_len)[:, :, None]  # [B,Q,1]
    flat_idx = (row_base + diff_slot).reshape(-1)
    anc_flat = anc.to(torch.int64).reshape(-1)
    anc_lookup = torch.take(anc_flat, flat_idx).reshape(batch_size, seq_len, seq_len)
    match1 = diffgt0 & (anc_lookup.to(torch.int32) == inner_k)

    mask = valid & (match0 | match1)
    return mask.reshape(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
@pytest.mark.parametrize(
    "case",
    _TREE_KERNEL_CASES,
    ids=lambda c: "_".join(
        f"{k}={c.get(k)}"
        for k in [
            "batch_size",
            "seq_len",
            "page_size",
            "group_size",
            "head_dim",
            "num_kv_heads",
            "max_tree_height",
            "tree_info_variant",
            "kv_variant",
            "extra_sche_words",
            "pos_base",
            "pos_encoding_mode",
        ]
        if k in c
    ),
)
def test_batch_tree_with_paged_kv_cache_against_prefill_custom_mask(case):
    torch.manual_seed(0)
    batch_size = int(case["batch_size"])
    seq_len = int(case["seq_len"])
    page_size = int(case["page_size"])
    group_size = int(case["group_size"])
    head_dim = int(case["head_dim"])
    num_kv_heads = int(case["num_kv_heads"])
    max_tree_height = int(case["max_tree_height"])
    pos_encoding_mode = case.get("pos_encoding_mode", "NONE")
    tree_info_variant = case.get("tree_info_variant", "contiguous")
    kv_variant = case.get("kv_variant", "contiguous")
    extra_sche_words = int(case.get("extra_sche_words", 0))
    pos_base = int(case.get("pos_base", 0))
    max_depth = int(case.get("max_depth", 16))
    max_width = int(case.get("max_width", 32))

    rng = torch.Generator(device="cpu").manual_seed(
        2025
        + batch_size * 10000
        + seq_len * 100
        + page_size * 10
        + group_size
        + (0 if pos_encoding_mode == "NONE" else 1)
        + max_tree_height * 3
    )

    device = torch.device("cuda:0")
    dtype = torch.float16
    num_qo_heads = num_kv_heads * group_size

    # Random tree per sequence (position_id depth <= 16 keeps position_id small; inner_id <= 254).
    batch_trees = [
        _generate_random_tree_sequence(
            seq_len=seq_len, max_depth=max_depth, max_width=max_width, rng=rng
        )
        for _ in range(batch_size)
    ]

    # Build tree_info pages and per-token position ids.
    tree_info, pos_ids = _build_tree_info_pages(
        batch_trees=batch_trees,
        max_tree_height=max_tree_height,
        page_size=page_size,
        pos_base=pos_base,
        extra_sche_words=extra_sche_words,
        tree_info_variant=tree_info_variant,
        device=device,
    )

    # Build dense custom mask for baseline prefill kernel.
    custom_mask = _tree_mask_dense(
        batch_trees=batch_trees,
        max_tree_height=max_tree_height,
        pos_base=pos_base,
        device=device,
    )

    # Tokens (unpadded), then pack to paged KV cache.
    q_raw = torch.randn(
        (batch_size * seq_len, num_qo_heads, head_dim), device=device, dtype=dtype
    )
    k_raw = torch.randn(
        (batch_size * seq_len, num_kv_heads, head_dim), device=device, dtype=dtype
    )
    v_raw = torch.randn_like(k_raw)

    # For ROPE_LLAMA, tree kernel applies RoPE internally based on tree_info.position_id.
    # Baseline uses pre-rotated (q,k) and runs prefill with pos_encoding_mode="NONE".
    if pos_encoding_mode == "ROPE_LLAMA":
        q_base = q_raw.clone()
        k_base = k_raw.clone()
        flashinfer.apply_rope_pos_ids_inplace(
            q_base,
            k_base,
            pos_ids.to(torch.int32),
            rotary_dim=head_dim,
            interleave=False,
            rope_scale=1.0,
            rope_theta=1e4,
        )
        k_cache_base, v_cache = _pack_paged_kv_from_tokens(
            k_tok=k_base,
            v_tok=v_raw,
            batch_size=batch_size,
            seq_len=seq_len,
            page_size=page_size,
            kv_variant=kv_variant,
        )
        q_tree = q_raw
        k_cache_tree, _ = _pack_paged_kv_from_tokens(
            k_tok=k_raw,
            v_tok=v_raw,
            batch_size=batch_size,
            seq_len=seq_len,
            page_size=page_size,
            kv_variant=kv_variant,
        )
        prefill_pos_mode = "NONE"
        tree_pos_mode = "ROPE_LLAMA"
    else:
        q_base = q_raw
        q_tree = q_raw
        k_cache_tree, v_cache = _pack_paged_kv_from_tokens(
            k_tok=k_raw,
            v_tok=v_raw,
            batch_size=batch_size,
            seq_len=seq_len,
            page_size=page_size,
            kv_variant=kv_variant,
        )
        k_cache_base = k_cache_tree
        prefill_pos_mode = "NONE"
        tree_pos_mode = "NONE"

    # Paged KV indptr/indices (contiguous pages per sequence).
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * seq_len
    )
    paged_kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * num_pages_per_seq
    )
    paged_kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.full(
        (batch_size,),
        (seq_len - 1) % page_size + 1,
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Baseline: batch prefill + dense custom mask.
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    prefill.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode=prefill_pos_mode,
    )
    o_base, lse_base = prefill.run(
        q_base, (k_cache_base, v_cache), return_lse=True
    )

    # Tree kernel: tree_mask driven by tree_info.
    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_tree_height=max_tree_height,
        causal=False,
        pos_encoding_mode=tree_pos_mode,
    )
    o_tree, lse_tree = tree.run(q_tree, (k_cache_tree, v_cache), return_lse=True)

    torch.testing.assert_close(o_tree, o_base, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse_tree, lse_base, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
def test_batch_tree_mask_respects_is_delete():
    """
    Focused test for the is_delete rule:
      if kv.is_delete == 1 => masked (false)
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim = 128
    batch_size = 1
    seq_len = 16
    page_size = 16
    max_tree_height = 9
    num_kv_heads = 4
    num_qo_heads = 4

    rng = torch.Generator(device="cpu").manual_seed(0)
    # Ensure there exists at least one token with position_id > 0 so we can delete it without
    # fully masking its own query row.
    for _ in range(50):
        tree_seq = _generate_random_tree_sequence(
            seq_len=seq_len, max_depth=8, max_width=8, rng=rng
        )
        if any(p > 0 for p in tree_seq.position_id):
            break
    else:
        raise RuntimeError("Failed to generate a tree with depth >= 2 for is_delete test")
    tree_info, _ = _build_tree_info_pages(
        batch_trees=[tree_seq], max_tree_height=max_tree_height, page_size=page_size, device=device
    )

    # Mark a key token (with position_id > 0) as deleted in tree_info word1.
    # This avoids the degenerate case where the deleted token has no ancestors and would fully
    # mask its own query row (potential NaNs in softmax).
    del_idx = next(i for i, p in enumerate(tree_seq.position_id) if p > 0)
    page = del_idx // page_size
    entry = del_idx % page_size
    # Some uint32 bitwise ops are not implemented on CUDA; flip the bit in int64 then cast back.
    tree_info_i64 = tree_info.to(torch.int64)
    tree_info_i64[page, entry, 1] = tree_info_i64[page, entry, 1] | (1 << 31)
    tree_info = tree_info_i64.to(torch.uint32)

    # Build custom mask baseline with the same deletion (mask out token 0 for all queries).
    custom_mask = _tree_mask_dense(
        batch_trees=[tree_seq], max_tree_height=max_tree_height, device=device
    ).view(1, seq_len, seq_len)
    custom_mask[:, :, del_idx] = False
    custom_mask = custom_mask.reshape(-1)

    q = torch.randn((seq_len, num_qo_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_cache, v_cache = _pack_paged_kv_from_tokens(
        k_tok=k, v_tok=v, batch_size=batch_size, seq_len=seq_len, page_size=page_size
    )

    qo_indptr = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(1, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([seq_len], device=device, dtype=torch.int32)

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    prefill.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode="NONE",
    )
    o_base = prefill.run(q, (k_cache, v_cache))

    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        max_tree_height=max_tree_height,
    )
    o_tree = tree.run(q, (k_cache, v_cache))

    torch.testing.assert_close(o_tree, o_base, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
def test_batch_tree_mask_respects_is_ghost():
    """
    Focused test for the is_ghost rule:
      if (kv.is_delete | kv.is_ghost) != 0 => masked (false)
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim = 128
    batch_size = 1
    seq_len = 16
    page_size = 16
    max_tree_height = 9
    num_kv_heads = 4
    num_qo_heads = 4

    rng = torch.Generator(device="cpu").manual_seed(1)
    for _ in range(50):
        tree_seq = _generate_random_tree_sequence(
            seq_len=seq_len, max_depth=8, max_width=8, rng=rng
        )
        if any(p > 0 for p in tree_seq.position_id):
            break
    else:
        raise RuntimeError("Failed to generate a tree with depth >= 2 for is_ghost test")

    tree_info, _ = _build_tree_info_pages(
        batch_trees=[tree_seq], max_tree_height=max_tree_height, page_size=page_size, device=device
    )

    # Mark a key token (with position_id > 0) as ghost in tree_info word1.
    ghost_idx = next(i for i, p in enumerate(tree_seq.position_id) if p > 0)
    page = ghost_idx // page_size
    entry = ghost_idx % page_size
    tree_info_i64 = tree_info.to(torch.int64)
    tree_info_i64[page, entry, 1] = tree_info_i64[page, entry, 1] | (1 << 30)
    tree_info = tree_info_i64.to(torch.uint32)

    custom_mask = _tree_mask_dense(
        batch_trees=[tree_seq], max_tree_height=max_tree_height, device=device
    ).view(1, seq_len, seq_len)
    custom_mask[:, :, ghost_idx] = False
    custom_mask = custom_mask.reshape(-1)

    q = torch.randn((seq_len, num_qo_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_cache, v_cache = _pack_paged_kv_from_tokens(
        k_tok=k, v_tok=v, batch_size=batch_size, seq_len=seq_len, page_size=page_size
    )

    qo_indptr = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(1, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([seq_len], device=device, dtype=torch.int32)

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    prefill.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode="NONE",
    )
    o_base = prefill.run(q, (k_cache, v_cache))

    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        max_tree_height=max_tree_height,
    )
    o_tree = tree.run(q, (k_cache, v_cache))

    torch.testing.assert_close(o_tree, o_base, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
def test_batch_tree_mask_with_padding_tokens_in_last_page():
    """
    Stress the behavior when the last page is only partially filled.
    The kernel should treat padded entries as out-of-range and never read garbage tree_info.
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim = 128
    batch_size = 2
    seq_len = 33
    page_size = 16  # last page has 1 token
    max_tree_height = 10  # crosses a 4-byte packing boundary
    num_kv_heads = 4
    num_qo_heads = 8

    rng = torch.Generator(device="cpu").manual_seed(123)
    batch_trees = [
        _generate_random_tree_sequence(seq_len=seq_len, max_depth=16, max_width=32, rng=rng)
        for _ in range(batch_size)
    ]
    tree_info, pos_ids = _build_tree_info_pages(
        batch_trees=batch_trees,
        max_tree_height=max_tree_height,
        page_size=page_size,
        device=device,
    )
    custom_mask = _tree_mask_dense(
        batch_trees=batch_trees, max_tree_height=max_tree_height, device=device
    )

    q = torch.randn((batch_size * seq_len, num_qo_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch_size * seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)

    # RoPE path sanity: use moderate positions.
    q_base = q.clone()
    k_base = k.clone()
    flashinfer.apply_rope_pos_ids_inplace(
        q_base,
        k_base,
        pos_ids.to(torch.int32),
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=1e4,
    )

    k_cache_base, v_cache = _pack_paged_kv_from_tokens(
        k_tok=k_base,
        v_tok=v,
        batch_size=batch_size,
        seq_len=seq_len,
        page_size=page_size,
    )
    k_cache_tree, _ = _pack_paged_kv_from_tokens(
        k_tok=k,
        v_tok=v,
        batch_size=batch_size,
        seq_len=seq_len,
        page_size=page_size,
    )

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * seq_len
    )
    paged_kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * num_pages_per_seq
    )
    paged_kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.full(
        (batch_size,),
        (seq_len - 1) % page_size + 1,
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    prefill.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode="NONE",
    )
    o_base = prefill.run(q_base, (k_cache_base, v_cache))

    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="ROPE_LLAMA",
        max_tree_height=max_tree_height,
    )
    o_tree = tree.run(q, (k_cache_tree, v_cache))
    torch.testing.assert_close(o_tree, o_base, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
def test_batch_tree_mask_structured_main_path_expected_mean():
    """
    Deterministic mask correctness test that does not rely on the prefill baseline.

    Use q=k=0 so all allowed logits are identical => attention is uniform over allowed keys.
    With v[token] filled with token_id, the output equals the mean token_id over allowed keys.
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim = 256  # reuse an already-warmed JIT head_dim
    batch_size = 1
    width = 8
    depth = 12  # diff spans 1..11 (crosses 4-byte packing boundaries at 4/5 and 8/9)
    seq_len = width * depth
    page_size = 7  # deliberately not dividing seq_len
    max_tree_height = 14  # larger than depth to exercise padding within packed words
    num_kv_heads = 1
    num_qo_heads = 1

    main_inner_by_pos = [int((p * 3 + 1) % width) for p in range(depth)]
    tree_seq = _generate_structured_tree_sequence_with_main_path(
        depth=depth, width=width, main_inner_by_pos=main_inner_by_pos
    )
    tree_info, _ = _build_tree_info_pages(
        batch_trees=[tree_seq],
        max_tree_height=max_tree_height,
        page_size=page_size,
        device=device,
    )

    q = torch.zeros((seq_len, num_qo_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    token_ids = torch.arange(seq_len, device=device, dtype=dtype)
    v = token_ids.view(seq_len, 1, 1).expand(seq_len, num_kv_heads, head_dim).contiguous()

    k_cache, v_cache = _pack_paged_kv_from_tokens(
        k_tok=k, v_tok=v, batch_size=batch_size, seq_len=seq_len, page_size=page_size
    )

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    qo_indptr = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, num_pages_per_seq], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(num_pages_per_seq, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor(
        [seq_len - (num_pages_per_seq - 1) * page_size], device=device, dtype=torch.int32
    )

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        max_tree_height=max_tree_height,
    )
    o = tree.run(q, (k_cache, v_cache))

    # Check main-path queries at multiple levels, including around packing boundaries.
    check_levels = [0, 1, 4, 5, 8, 9, depth - 1]
    for p in check_levels:
        inner = int(main_inner_by_pos[p])
        q_token_id = p * width + inner
        allowed_ids = [t * width + int(main_inner_by_pos[t]) for t in range(p + 1)]
        expected = float(sum(allowed_ids)) / float(len(allowed_ids))
        got = float(o[q_token_id, 0, 0].item())
        assert abs(got - expected) <= 2e-2, (p, got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for tree kernel tests")
def test_batch_tree_mask_structured_max_diff_truncates():
    """
    Verify that runtime `anc_array_len` (max_diff) truncates the mask even if tree_info contains
    more packed ancestor bytes.
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim = 128
    batch_size = 1
    width = 8
    depth = 12
    seq_len = width * depth
    page_size = 7
    max_tree_height_storage = 14  # tree_info contains up to 13 diffs (padded past depth-1)
    max_tree_height_runtime = 8  # kernel should only allow diff in [0, 7] (8 keys incl. self)
    num_kv_heads = 1
    num_qo_heads = 1

    main_inner_by_pos = [int((p * 3 + 1) % width) for p in range(depth)]
    tree_seq = _generate_structured_tree_sequence_with_main_path(
        depth=depth, width=width, main_inner_by_pos=main_inner_by_pos
    )
    tree_info, _ = _build_tree_info_pages(
        batch_trees=[tree_seq],
        max_tree_height=max_tree_height_storage,
        page_size=page_size,
        device=device,
    )

    q = torch.zeros((seq_len, num_qo_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    token_ids = torch.arange(seq_len, device=device, dtype=dtype)
    v = token_ids.view(seq_len, 1, 1).expand(seq_len, num_kv_heads, head_dim).contiguous()

    k_cache, v_cache = _pack_paged_kv_from_tokens(
        k_tok=k, v_tok=v, batch_size=batch_size, seq_len=seq_len, page_size=page_size
    )

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    qo_indptr = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, num_pages_per_seq], device=device, dtype=torch.int32)
    paged_kv_indices = torch.arange(num_pages_per_seq, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor(
        [seq_len - (num_pages_per_seq - 1) * page_size], device=device, dtype=torch.int32
    )

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    tree = flashinfer.BatchTreeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    tree.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        max_tree_height=max_tree_height_runtime,
    )
    o = tree.run(q, (k_cache, v_cache))

    # For a deep query on the main path, only the last (max_tree_height_runtime - 1) ancestors plus
    # itself should remain.
    p = depth - 1
    inner = int(main_inner_by_pos[p])
    q_token_id = p * width + inner
    start_level = max(0, p - (max_tree_height_runtime - 1))
    allowed_ids = [t * width + int(main_inner_by_pos[t]) for t in range(start_level, p + 1)]
    expected = float(sum(allowed_ids)) / float(len(allowed_ids))
    got = float(o[q_token_id, 0, 0].item())
    assert abs(got - expected) <= 2e-2, (got, expected)
