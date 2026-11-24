"""Arrange a control-flow graph as nested loops and conditionals driven by a block-number variable.

    structure(succ) -> tree, or None when the graph is irreducible

`succ[b]` lists the blocks that block b can go to next (none = the program ends there); block 0 is the entry.
Every block ends by storing its successor's number in a variable `pc` (END when it has none). The tree says where
each block's code goes:

    ("block", b)             run block b
    ("guard", b, [items])    if pc == b: run the items (they start with block b, or with the loop headed by b)
    ("loop", b, [items])     repeat the items, which start with block b, for as long as pc == b afterwards

Why this instead of one loop around a switch: a shader compiler has to rediscover the real control flow from a
switch on a variable that every block writes, which takes it minutes on shaders with a thousand blocks. Here the
loops are loops, and the guards are tests against a value that was just assigned, which fold at once.

How it is built (reducible graphs only, i.e. every loop has a single entry):
  - Each natural loop becomes a "loop" item. Leaving it needs no break: pc simply no longer names the header.
  - Inside a loop (and at the top level) the members are nested by dominance: after a member come the members it
    immediately dominates, each behind a guard, in an order where every forward edge points later. A jump to a
    member that isn't below the current one falls out of the guards it is in, to the level where a later guard
    matches. A jump out of the loop falls out of all of them.
A block is never skipped or run twice in one pass, so any path through the graph is followed exactly.
"""
import collections

END = -1


def _order(succ):
    """Reverse postorder from block 0, the retreating edges of that walk, and the predecessors of each block."""
    state, order, back = {0: 1}, [], []
    stack = [(0, iter(succ[0]))]
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in state:
                state[w] = 1
                stack.append((w, iter(succ[w])))
                break
            if state[w] == 1:
                back.append((v, w))
        else:
            state[v] = 2
            order.append(v)
            stack.pop()
    rpo = order[::-1]
    preds = collections.defaultdict(list)
    for v in rpo:
        for w in succ[v]:
            preds[w].append(v)
    return rpo, back, preds


def _dominators(rpo, preds):
    """Immediate dominator of each reachable block (Cooper, Harvey and Kennedy)."""
    pos = {v: i for i, v in enumerate(rpo)}
    idom = {rpo[0]: rpo[0]}
    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = None
            for p in preds[v]:
                if p not in idom:
                    continue
                if new is None:
                    new = p
                    continue
                a, b = p, new
                while a != b:
                    while pos[a] > pos[b]:
                        a = idom[a]
                    while pos[b] > pos[a]:
                        b = idom[b]
                new = a
            if idom.get(v) != new:
                idom[v] = new
                changed = True
    return idom, pos


def structure(succ):
    rpo, back, preds = _order(succ)
    idom, pos = _dominators(rpo, preds)

    def dominates(a, b):
        while pos[b] > pos[a]:
            b = idom[b]
        return a == b

    if not all(dominates(w, v) for v, w in back):
        return None                                   # a loop with two entries: no nesting expresses it

    # Natural loops, and the innermost loop each block is in (named by its header; None = not in a loop).
    bodies = {}
    for v, h in back:
        body = bodies.setdefault(h, {h})
        work = [v]
        while work:
            x = work.pop()
            if x not in body:
                body.add(x)
                work.extend(preds[x])
    inner = {}
    for h in sorted(bodies, key=lambda h: len(bodies[h])):
        for x in bodies[h]:
            inner.setdefault(x, h)
    outer = {}                                        # loop -> the loop around it
    for h in bodies:
        around = [g for g in bodies if g != h and h in bodies[g]]
        outer[h] = min(around, key=lambda g: len(bodies[g])) if around else None

    def member(x, region):
        """The member of `region` that block x belongs to: x itself, or the loop inside the region holding it.
        Members are ("b", block) and ("l", header)."""
        loop = inner.get(x)
        if loop == region:
            return ("b", x)
        while outer[loop] != region:
            loop = outer[loop]
        return ("l", loop)

    # Dominance among the members of each region. A member's parent is the member holding the immediate
    # dominator of its first block; that dominator is always inside the same region.
    children = collections.defaultdict(list)
    for x in rpo[1:]:
        region = inner.get(x)
        if region == x:                               # a loop header: it is placed as its loop, in the outer region
            children[member(idom[x], outer[x])].append(("l", x))
        else:
            children[member(idom[x], region)].append(("b", x))

    # Flatten dominated tail continuations out of enclosing guards and build iteratively so long sequential
    # diamond/loop chains stay O(1) deep instead of nesting O(N) OpSelectionMerge constructs (and never hit
    # Python's recursion limit on multi-thousand-block shaders).
    # Hoisting the last child c_last of a guarded member m into m's enclosing list is exact: m's head strictly
    # dominates c_last's head, so no block outside m's subtree can jump to c_last's head. When m's guard is skipped
    # (pc != m_head), pc cannot equal c_last_head either, so c_last's guard is skipped too; when m's guard runs,
    # c_last runs right after m's earlier children, in the same order as before.
    max_guard_depth = 16
    root = []
    start = ("l", 0) if 0 in bodies else ("b", 0)
    stack = [("emit", root, None, start, 0)]
    while stack:
        task = stack.pop()
        if task[0] == "open_guard":
            _, parent_list, m, depth = task
            gbody = []
            parent_list.append(("guard", m[1], gbody))
            stack.append(("emit", gbody, parent_list, m, depth))
            continue
        _, out_list, parent_list, m, depth = task
        kind, x = m
        below = sorted(children[m], key=lambda c: pos[c[1]])
        if kind == "l":
            lbody = []
            out_list.append(("loop", x, lbody))
        else:
            out_list.append(("block", x))
        if kind == "b" and len(below) == 1 and list(succ[x]) == [below[0][1]]:
            stack.append(("emit", out_list, parent_list, below[0], depth))
        else:
            hoist = (parent_list is not None and below
                     and (len(below) >= 2 or kind == "l" or depth >= max_guard_depth))
            if hoist:
                stack.append(("open_guard", parent_list, below[-1], depth))
                inner = below[:-1]
            else:
                inner = below
            for c in reversed(inner):
                stack.append(("open_guard", out_list, c, depth + 1))
        if kind == "l":
            stack.append(("emit", lbody, None, ("b", x), 0))
    return root



def depth(tree):
    """Nesting depth of a tree from structure(): a block is depth 0 and each guard or loop adds one."""
    best = 0
    stack = [(tree, 0)]
    while stack:
        items, d = stack.pop()
        for it in items:
            if it[0] == "block":
                best = max(best, d)
            else:
                best = max(best, d + 1)
                stack.append((it[2], d + 1))
    return best


def _sccs(nodes, adj):
    """Iterative Tarjan's algorithm over `nodes` with adjacency `adj`; returns SCCs with >= 2 nodes."""
    index, low, on_stack, stack, out = {}, {}, set(), [], []
    nxt = 0
    for start in nodes:
        if start in index:
            continue
        index[start] = low[start] = nxt
        nxt += 1
        stack.append(start)
        on_stack.add(start)
        call = [(start, iter(adj[start]))]
        while call:
            v, it = call[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = nxt
                    nxt += 1
                    stack.append(w)
                    on_stack.add(w)
                    call.append((w, iter(adj[w])))
                    break
                if w in on_stack and index[w] < low[v]:
                    low[v] = index[w]
            else:
                if low[v] == index[v]:
                    comp = []
                    while True:
                        x = stack.pop()
                        on_stack.remove(x)
                        comp.append(x)
                        if x == v:
                            break
                    if len(comp) >= 2:
                        out.append(set(comp))
                call.pop()
                if call:
                    p = call[-1][0]
                    if low[v] < low[p]:
                        low[p] = low[v]
    return out


def split_irreducible(succ, max_factor=2, max_splits=64):
    """Make an irreducible CFG reducible by controlled node splitting (cloning the closed region R reachable from a secondary entry e without passing through primary entry h
    in an irreducible SCC, and redirecting e's external predecessors to clone[e]).

    Returns `(new_succ, orig_of)` where `orig_of[b]` is the original block index that block `b` executes
    and `new_succ[b]` preserves the exact positional successor order of `succ[orig_of[b]]`.
    Returns `(succ, list(range(len(succ))))` unchanged if `succ` is already reducible, or `None` if
    splitting would exceed `max_factor` * len(succ) blocks or `max_splits` duplications.
    """
    n0 = len(succ)
    if n0 == 0:
        return succ, []
    max_blocks = min(int(n0 * max_factor), n0 + max_splits)
    g = None
    orig = None
    while True:
        cur = succ if g is None else g
        rpo, back, preds = _order(cur)
        idom, pos = _dominators(rpo, preds)

        def dominates(a, b):
            while pos[b] > pos[a]:
                b = idom[b]
            return a == b

        if all(dominates(w, v) for v, w in back):
            if g is None:
                return succ, list(range(n0))
            return g, orig
        if g is None:
            g = [list(s) for s in succ]
            orig = list(range(n0))
        if len(g) >= max_blocks:
            return None

        # Strip dominating back-edges so nested irreducible cycles inside reducible loops surface as SCCs.
        adj = {v: [w for w in g[v] if not dominates(w, v)] for v in rpo}
        sccs = _sccs(rpo, adj)
        if not sccs:
            return None
        # Pick the smallest irreducible SCC (tie-break by earliest RPO position).
        scc = min(sccs, key=lambda c: (len(c), min(pos[x] for x in c)))
        entries = [x for x in scc if x == 0 or any(p not in scc for p in preds[x])]
        # Choose a pair (h, e) of SCC entries (e != h, e != 0) minimizing |R|, where R is the closed region of
        # SCC nodes reachable from e without passing through h. Cloning all of R at once keeps internal cycles
        # inside the clone instead of re-entering the original SCC and unrolling infinitely.
        best = None
        for h in entries:
            for e in entries:
                if e == h or e == 0:
                    continue
                region = {e}
                work = [e]
                while work:
                    u = work.pop()
                    for w in g[u]:
                        if w in scc and w != h and w not in region:
                            region.add(w)
                            work.append(w)
                key = (len(region), pos[h], -pos[e])
                if best is None or key < best[0]:
                    best = (key, h, e, region)
        if best is None:
            return None
        _, h, e, region = best
        if len(g) + len(region) > max_blocks:
            return None
        # Order cloned nodes deterministically by RPO position.
        reg_ordered = sorted(region, key=lambda x: pos[x])
        clone_of = {}
        for u in reg_ordered:
            clone_of[u] = len(g)
            orig.append(orig[u])
            g.append([])
        for u in reg_ordered:
            g[clone_of[u]] = [clone_of[w] if w in clone_of else w for w in g[u]]
        ext_preds = {p for p in preds[e] if p not in scc}
        ce = clone_of[e]
        for p in ext_preds:
            g[p] = [ce if w == e else w for w in g[p]]

def run(tree, choose, limit):
    """Interpret a tree: the blocks it executes, in order, stopping after `limit`. `choose(b)` gives the block
    that b goes to next (END to finish) and stands for whatever b stores in pc."""
    trace, pc = [], [0]
    stack = [("items", tree, 0)]
    while stack:
        frame = stack[-1]
        if frame[0] == "items":
            _, items, idx = frame
            if idx >= len(items):
                stack.pop()
                continue
            stack[-1] = ("items", items, idx + 1)
            it = items[idx]
            if it[0] == "block":
                if len(trace) >= limit:
                    break
                trace.append(it[1])
                pc[0] = choose(it[1])
            elif it[0] == "guard":
                if pc[0] == it[1]:
                    stack.append(("items", it[2], 0))
            else:
                stack.append(("loop", it[1], it[2]))
                stack.append(("items", it[2], 0))
        else:
            _, head, body = frame
            if pc[0] == head:
                stack.append(("items", body, 0))
            else:
                stack.pop()
    return trace
